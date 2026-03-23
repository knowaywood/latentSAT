from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from latentsat.model import LatentSATConfig, LatentSATModel, build_model
from latentsat.utils.verify import VerifyList


@dataclass
class ListSample:
    clauses: list[list[int]]
    answer: list[int]
    satisfiable: bool


class ListDataset(Dataset[ListSample]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.samples: list[ListSample] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(
                    ListSample(
                        clauses=obj["clauses"],
                        answer=[int(x) for x in obj["answer"]],
                        satisfiable=bool(obj["satisfiable"]),
                    )
                )
        if not self.samples:
            raise ValueError(f"No samples found in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ListSample:
        return self.samples[index]


def collate_fn(batch: Sequence[ListSample]) -> list[ListSample]:
    return list(batch)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunningMeanBaseline:
    def __init__(self, momentum: float = 0.9) -> None:
        self.momentum = momentum
        self.value = 0.0
        self.initialized = False

    def update(self, rewards: Tensor) -> Tensor:
        batch_mean = rewards.mean().item()
        if not self.initialized:
            self.value = batch_mean
            self.initialized = True
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * batch_mean
        return rewards.new_full(rewards.shape, self.value)


class RLTrainer:
    def __init__(
        self,
        model: LatentSATModel,
        lr: float,
        weight_decay: float,
        grad_clip: float,
        supervised_weight: float,
        rl_weight: float,
        entropy_coef: float,
        baseline_mix: float,
        clause_reward_weight: float,
        awr_weight: float,
        awr_temp: float,
        save_dir: str | Path,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.grad_clip = grad_clip
        self.supervised_weight = supervised_weight
        self.rl_weight = rl_weight
        self.entropy_coef = entropy_coef
        self.baseline_mix = baseline_mix
        self.clause_reward_weight = clause_reward_weight
        self.awr_weight = awr_weight
        self.awr_temp = max(1e-3, float(awr_temp))
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        self.baseline = RunningMeanBaseline()
        self.verify = VerifyList()
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = max(1, int(epsilon_decay_steps))
        self.global_step = 0

    def _num_vars_from_clauses(self, clauses: Sequence[Sequence[int]]) -> int:
        return max((abs(lit) for clause in clauses for lit in clause), default=0)

    def _clause_satisfaction_ratio(
        self, clauses: Sequence[Sequence[int]], assignment_bits: Sequence[int]
    ) -> float:
        if not clauses:
            return 0.0
        satisfied = 0
        for clause in clauses:
            clause_ok = False
            for lit in clause:
                idx = abs(lit) - 1
                if idx < 0 or idx >= len(assignment_bits):
                    continue
                val = assignment_bits[idx]
                if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                    clause_ok = True
                    break
            if clause_ok:
                satisfied += 1
        return satisfied / max(len(clauses), 1)

    def _current_epsilon(self) -> float:
        progress = min(1.0, self.global_step / self.epsilon_decay_steps)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def _canonicalize_output(
        self, sat_value: int, assignment_bits: Sequence[int]
    ) -> list[int]:
        if int(sat_value) == 0:
            return [0]
        return [1, *[int(x) for x in assignment_bits]]

    def _supervised_loss(
        self, sample: ListSample, output: object
    ) -> tuple[Tensor, dict[str, float]]:
        target = torch.tensor(
            sample.answer,
            dtype=output.sat_logits.dtype,
            device=self.device,
        )

        sat_target = target[:1].view_as(output.sat_logits[0])
        sat_loss = F.binary_cross_entropy_with_logits(output.sat_logits[0], sat_target)

        assignment_loss = torch.zeros((), device=self.device, dtype=sat_loss.dtype)
        if len(sample.answer) > 1:
            assign_target = target[1:].view(-1)
            assign_logits = output.assignment_logits[0, : assign_target.numel()]
            assignment_loss = F.binary_cross_entropy_with_logits(
                assign_logits,
                assign_target,
            )

        stop_loss = F.binary_cross_entropy(
            output.halt_probs.max(dim=1).values,
            torch.ones(1, device=self.device, dtype=output.halt_probs.dtype),
        )
        loss = sat_loss + assignment_loss + 0.1 * stop_loss
        metrics = {
            "sat_loss": float(sat_loss.item()),
            "assignment_loss": float(assignment_loss.item()),
            "stop_loss": float(stop_loss.item()),
        }
        return loss, metrics

    def _compute_reward(
        self,
        sample: ListSample,
        predicted_output: Sequence[int],
        stop_prob: Tensor,
    ) -> float:
        num_vars = self._num_vars_from_clauses(sample.clauses)
        prediction = [int(x) for x in predicted_output]
        verify_ok = self.verify(
            sample.clauses, num_vars, prediction, sample.satisfiable
        )
        sat_ok = self.verify.verify_sat(prediction, sample.satisfiable)

        if verify_ok:
            reward = 1.5
        elif sat_ok and sample.satisfiable and prediction[0] == 1:
            ratio = self._clause_satisfaction_ratio(
                sample.clauses, prediction[1 : num_vars + 1]
            )
            # SAT label correct but assignment wrong: keep a gradient signal,
            # but make it clearly lower than exact correctness.
            reward = -0.4 + self.clause_reward_weight * ratio
        else:
            reward = -1.0

        if (
            sample.satisfiable
            and prediction[0] == 1
            and len(prediction) >= num_vars + 1
            and num_vars > 0
        ):
            ratio = self._clause_satisfaction_ratio(
                sample.clauses, prediction[1 : num_vars + 1]
            )
            reward += 0.2 * self.clause_reward_weight * ratio

        # Keep a weak preference for decoding readiness.
        reward += 0.01 * float(stop_prob.item())
        return reward

    def _awr_loss(
        self, sample: ListSample, output: object, sampled_output: Sequence[int]
    ) -> Tensor:
        sat_target = torch.tensor(
            [int(sampled_output[0])],
            device=self.device,
            dtype=output.sat_logits.dtype,
        ).view_as(output.sat_logits[0])
        sat_loss = F.binary_cross_entropy_with_logits(output.sat_logits[0], sat_target)

        assignment_loss = torch.zeros((), device=self.device, dtype=sat_loss.dtype)
        num_vars = self._num_vars_from_clauses(sample.clauses)
        if int(sampled_output[0]) == 1 and num_vars > 0 and len(sampled_output) > 1:
            max_len = min(num_vars, len(sampled_output) - 1)
            assign_target = torch.tensor(
                sampled_output[1 : max_len + 1],
                device=self.device,
                dtype=output.assignment_logits.dtype,
            ).view(-1)
            assign_logits = output.assignment_logits[0, : assign_target.numel()]
            assignment_loss = F.binary_cross_entropy_with_logits(
                assign_logits,
                assign_target,
            )
        return sat_loss + assignment_loss

    def _sample_policy(
        self, sample: ListSample
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        list[int],
        float,
        float,
        int,
        bool,
        dict[str, float],
    ]:
        output = self.model(sample.clauses)
        epsilon = self._current_epsilon()
        explore = random.random() < epsilon

        sat_dist = torch.distributions.Bernoulli(logits=output.sat_logits[0])
        if explore:
            sat_action = torch.randint(
                0,
                2,
                sat_dist.logits.shape,
                device=self.device,
                dtype=sat_dist.logits.dtype,
            )
        else:
            sat_action = sat_dist.sample()
        sat_log_prob = sat_dist.log_prob(sat_action).sum()
        sat_entropy = sat_dist.entropy().sum()

        num_vars = self._num_vars_from_clauses(sample.clauses)
        assignment_log_prob = torch.zeros((), device=self.device)
        assignment_entropy = torch.zeros((), device=self.device)
        assignment_bits: list[int] = []
        sampled_sat = int(sat_action.item())
        if sampled_sat == 1 and num_vars > 0:
            assign_logits = output.assignment_logits[0, :num_vars]
            assign_dist = torch.distributions.Bernoulli(logits=assign_logits)
            if explore:
                assign_action = torch.randint(
                    0,
                    2,
                    assign_logits.shape,
                    device=self.device,
                    dtype=assign_logits.dtype,
                )
            else:
                assign_action = assign_dist.sample()
            assignment_log_prob = assign_dist.log_prob(assign_action).sum()
            assignment_entropy = assign_dist.entropy().sum()
            assignment_bits = [int(x) for x in assign_action.tolist()]

        sampled_output = self._canonicalize_output(sampled_sat, assignment_bits)
        greedy_output = self.model.decode_solution(
            output.sat_logits,
            output.assignment_logits,
            num_vars=num_vars,
        )[0]
        greedy_output = self._canonicalize_output(
            greedy_output[0],
            greedy_output[1:],
        )
        stop_prob = output.halt_probs.max(dim=1).values[0]
        reward = self._compute_reward(sample, sampled_output, stop_prob)
        greedy_reward = self._compute_reward(sample, greedy_output, stop_prob)
        supervised_loss, supervised_metrics = self._supervised_loss(sample, output)
        awr_loss = self._awr_loss(sample, output, sampled_output)
        log_prob = sat_log_prob + assignment_log_prob
        entropy = sat_entropy + assignment_entropy
        return (
            log_prob,
            entropy,
            stop_prob,
            supervised_loss,
            awr_loss,
            sampled_output,
            reward,
            greedy_reward,
            int(output.halt_steps[0].item()),
            explore,
            supervised_metrics,
        )

    def train_epoch(
        self, dataloader: DataLoader[list[ListSample]], epoch: int, log_every: int
    ) -> None:
        self.model.train()
        for step, batch in enumerate(dataloader, start=1):
            log_probs: list[Tensor] = []
            entropies: list[Tensor] = []
            rewards_list: list[float] = []
            greedy_rewards_list: list[float] = []
            stop_probs: list[Tensor] = []
            supervised_losses: list[Tensor] = []
            awr_losses: list[Tensor] = []
            samples_out: list[list[int]] = []
            halt_steps: list[int] = []
            explore_count = 0
            metric_buffer: list[dict[str, float]] = []

            for sample in batch:
                self.global_step += 1
                (
                    log_prob,
                    entropy,
                    stop_prob,
                    supervised_loss,
                    awr_loss,
                    sampled_output,
                    reward,
                    greedy_reward,
                    halt_step,
                    explored,
                    supervised_metrics,
                ) = self._sample_policy(sample)
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards_list.append(reward)
                greedy_rewards_list.append(greedy_reward)
                stop_probs.append(stop_prob)
                supervised_losses.append(supervised_loss)
                awr_losses.append(awr_loss)
                samples_out.append(sampled_output)
                halt_steps.append(halt_step)
                metric_buffer.append(supervised_metrics)
                if explored:
                    explore_count += 1

            rewards = torch.tensor(
                rewards_list, dtype=torch.float32, device=self.device
            )
            greedy_rewards = torch.tensor(
                greedy_rewards_list, dtype=torch.float32, device=self.device
            )
            mean_baseline = self.baseline.update(rewards)
            baseline = (
                self.baseline_mix * greedy_rewards
                + (1.0 - self.baseline_mix) * mean_baseline
            )
            raw_advantages = rewards - baseline
            adv_std = raw_advantages.std(unbiased=False).clamp_min(1e-4)
            advantages = torch.clamp(raw_advantages / adv_std, min=-3.0, max=3.0)
            log_prob_tensor = torch.stack(log_probs)
            entropy_tensor = torch.stack(entropies)
            stop_prob_tensor = torch.stack(stop_probs)
            supervised_loss_tensor = torch.stack(supervised_losses)
            awr_loss_tensor = torch.stack(awr_losses)

            policy_loss = -(advantages.detach() * log_prob_tensor).mean()
            supervised_loss = supervised_loss_tensor.mean()
            awr_weights = torch.exp(
                torch.clamp(advantages, min=-2.0, max=2.0) / self.awr_temp
            ).detach()
            awr_loss = (awr_weights * awr_loss_tensor).sum() / (awr_weights.sum() + 1e-8)
            entropy_bonus = entropy_tensor.mean()
            stop_penalty = (1.0 - stop_prob_tensor).mean()
            loss = self.supervised_weight * supervised_loss
            loss = loss + self.rl_weight * policy_loss
            loss = loss + self.awr_weight * awr_loss
            loss = loss + 0.05 * stop_penalty
            loss = loss - self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if step % log_every == 0 or step == 1:
                mean_metrics = {
                    key: sum(m[key] for m in metric_buffer) / len(metric_buffer)
                    for key in metric_buffer[0]
                }
                print(
                    f"epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"sup={supervised_loss.item():.4f} rl={policy_loss.item():.4f} awr={awr_loss.item():.4f} "
                    f"reward={rewards.mean().item():.4f} greedy={greedy_rewards.mean().item():.4f} "
                    f"adv={advantages.mean().item():.4f} "
                    f"sat={mean_metrics['sat_loss']:.4f} assign={mean_metrics['assignment_loss']:.4f} "
                    f"stop={mean_metrics['stop_loss']:.4f} "
                    f"halt={sum(halt_steps) / len(halt_steps):.2f} "
                    f"eps={self._current_epsilon():.4f} explore={explore_count}/{len(batch)}"
                )
                sample_num_vars = self._num_vars_from_clauses(batch[0].clauses)
                print(
                    f"sample_pred={samples_out[0]} "
                    f"verify={self.verify(batch[0].clauses, sample_num_vars, samples_out[0], batch[0].satisfiable)} "
                    f"target_sat={int(batch[0].satisfiable)}"
                )

    def save_checkpoint(self, name: str) -> None:
        save_path = self.save_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path / "model.pt")
        with (save_path / "config.json").open("w", encoding="utf-8") as f:
            json.dump(self.model.config.__dict__, f, ensure_ascii=False, indent=2)


def load_pretrained_model(
    checkpoint_dir: str | Path,
    device: str | None = None,
) -> LatentSATModel:
    checkpoint_path = Path(checkpoint_dir)
    config_path = checkpoint_path / "config.json"
    model_path = checkpoint_path / "model.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)
    model = build_model(LatentSATConfig(**config_dict))
    state_dict = torch.load(model_path, map_location=device or "cpu")
    model.load_state_dict(state_dict)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL fine-tuning for LatentSAT")
    parser.add_argument("--train-file", type=str, default="data/list_data_rl.jsonl")
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        default="checkpoints/list_struct/pretrain/epoch_4",
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints/list_struct/rl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--supervised-weight", type=float, default=0.7)
    parser.add_argument("--rl-weight", type=float, default=0.8)
    parser.add_argument("--entropy-coef", type=float, default=0.0005)
    parser.add_argument("--baseline-mix", type=float, default=0.3)
    parser.add_argument("--clause-reward-weight", type=float, default=0.6)
    parser.add_argument("--awr-weight", type=float, default=0.1)
    parser.add_argument("--awr-temp", type=float, default=0.7)
    parser.add_argument("--epsilon-start", type=float, default=0.15)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = ListDataset(args.train_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = load_pretrained_model(args.pretrained_dir, device=args.device)
    trainer = RLTrainer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        supervised_weight=args.supervised_weight,
        rl_weight=args.rl_weight,
        entropy_coef=args.entropy_coef,
        baseline_mix=args.baseline_mix,
        clause_reward_weight=args.clause_reward_weight,
        awr_weight=args.awr_weight,
        awr_temp=args.awr_temp,
        save_dir=args.save_dir,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        device=args.device,
    )

    for epoch in range(1, args.epochs + 1):
        trainer.train_epoch(dataloader, epoch=epoch, log_every=args.log_every)
        trainer.save_checkpoint(f"epoch_{epoch}")


if __name__ == "__main__":
    main()
