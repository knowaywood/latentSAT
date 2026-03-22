from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
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
        save_dir: str | Path,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.grad_clip = grad_clip
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

    def _current_epsilon(self) -> float:
        progress = min(1.0, self.global_step / self.epsilon_decay_steps)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def _sample_policy(
        self, sample: ListSample
    ) -> tuple[Tensor, Tensor, list[int], float, int, bool]:
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

        num_vars = self._num_vars_from_clauses(sample.clauses)
        assignment_log_prob = torch.zeros((), device=self.device)
        assignment_bits: list[int] = []
        if num_vars > 0:
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
            assignment_bits = [int(x) for x in assign_action.tolist()]

        sampled_output = [int(sat_action.item()), *assignment_bits]
        verify_ok = self.verify(
            sample.clauses, num_vars, sampled_output, sample.satisfiable
        )
        sat_ok = self.verify.verify_sat(sampled_output, sample.satisfiable)
        len_ok = self.verify.verify_len_err(
            num_vars, sampled_output, sample.satisfiable
        )
        reward = 1.0 if verify_ok else -0.5
        if (not verify_ok) and sat_ok:
            reward += 0.2
        if not len_ok:
            reward -= 0.2

        stop_prob = output.halt_probs.max(dim=1).values[0]
        stop_bonus = 0.1 * float(stop_prob.item())
        reward += stop_bonus
        log_prob = sat_log_prob + assignment_log_prob
        return (
            log_prob,
            stop_prob,
            sampled_output,
            reward,
            int(output.halt_steps[0].item()),
            explore,
        )

    def train_epoch(
        self, dataloader: DataLoader[list[ListSample]], epoch: int, log_every: int
    ) -> None:
        self.model.train()
        for step, batch in enumerate(dataloader, start=1):
            log_probs: list[Tensor] = []
            rewards_list: list[float] = []
            stop_probs: list[Tensor] = []
            samples_out: list[list[int]] = []
            halt_steps: list[int] = []
            explore_count = 0

            for sample in batch:
                self.global_step += 1
                log_prob, stop_prob, sampled_output, reward, halt_step, explored = (
                    self._sample_policy(sample)
                )
                log_probs.append(log_prob)
                rewards_list.append(reward)
                stop_probs.append(stop_prob)
                samples_out.append(sampled_output)
                halt_steps.append(halt_step)
                if explored:
                    explore_count += 1

            rewards = torch.tensor(
                rewards_list, dtype=torch.float32, device=self.device
            )
            baseline = self.baseline.update(rewards)
            advantages = rewards - baseline
            log_prob_tensor = torch.stack(log_probs)
            stop_prob_tensor = torch.stack(stop_probs)

            # Encourage high reward actions and mildly encourage eventually decoding.
            loss = -(advantages.detach() * log_prob_tensor).mean()
            loss = loss + 0.05 * (1.0 - stop_prob_tensor).mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if step % log_every == 0 or step == 1:
                print(
                    f"epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"reward={rewards.mean().item():.4f} halt={sum(halt_steps) / len(halt_steps):.2f} "
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
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
