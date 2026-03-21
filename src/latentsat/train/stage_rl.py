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
from latentsat.utils.verify import VerifyText

DEFAULT_PROMPT = "请判断该 CNF 是否可满足，并输出最终结论。若可满足，请输出 Assignment: {变量赋值字典}；否则输出 UNSAT。"

verify = VerifyText()


@dataclass
class RLSample:
    clauses: list[list[int]]
    satisfiable: bool
    prompt: str
    answer: str


@dataclass
class RolloutBatch:
    texts: list[str]
    token_ids: Tensor
    sequence_lengths: list[int]
    token_log_probs: Tensor
    token_mask: Tensor
    rewards: Tensor


class CNFRLDataset(Dataset[RLSample]):
    def __init__(self, path: str | Path, prompt: str = DEFAULT_PROMPT) -> None:
        self.path = Path(path)
        self.prompt = prompt
        self.samples: list[RLSample] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(
                    RLSample(
                        clauses=obj["clauses"],
                        satisfiable=bool(obj["satisfiable"]),
                        prompt=obj.get("prompt", self.prompt),
                        answer=obj.get("answer", "UNSAT"),
                    )
                )

        if not self.samples:
            raise ValueError(f"No samples found in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RLSample:
        return self.samples[index]


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


class PolicyGradientTrainer:
    def __init__(
        self,
        model: LatentSATModel,
        lr: float,
        weight_decay: float,
        temperature: float,
        max_new_tokens: int,
        grad_clip: float,
        save_dir: str | Path,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.grad_clip = grad_clip
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self._freeze_for_rl()
        self.optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        self.baseline = RunningMeanBaseline()

    def _freeze_for_rl(self) -> None:
        for param in self.model.llm.parameters():
            param.requires_grad = False
        for module in (self.model.cnf_adapter, self.model.dual_track):
            for param in module.parameters():
                param.requires_grad = True

    def _decode_texts(
        self, token_ids: Tensor, sequence_lengths: Sequence[int]
    ) -> list[str]:
        trimmed = [
            token_ids[i, : sequence_lengths[i]].detach().cpu().tolist()
            for i in range(token_ids.size(0))
        ]
        return self.model.tokenizer.batch_decode(trimmed, skip_special_tokens=True)

    def _compute_reward(
        self, clauses: list[list[int]], satisfiable: bool, text: str
    ) -> float:
        text = text.strip()
        if not text:
            return -0.5

        reward = 0.0
        predicted_sat = verify._is_sat(text)
        if predicted_sat == satisfiable:
            reward += 0.2
        else:
            reward -= 0.2

        assignment: dict[int, bool]
        try:
            assignment = verify.extract_conclusion(text)
        except Exception:
            assignment = {}

        if verify(clauses, assignment, satisfiable):
            reward += 1.0
        else:
            reward -= 0.5

        if satisfiable and "Assignment:" in text:
            reward += 0.1
        if not satisfiable and "UNSAT" in text:
            reward += 0.1

        return reward

    @torch.no_grad()
    def _sample_sequences(
        self,
        batch: Sequence[RLSample],
    ) -> tuple[Tensor, list[int], list[str], Tensor]:
        clauses_batch = [sample.clauses for sample in batch]
        prompts = [sample.prompt for sample in batch]
        (
            final_inputs,
            final_attention_mask,
            _,
            _,
            _,
        ) = self.model._compose_final_inputs(clauses_batch, prompts)

        batch_size = len(batch)
        generated_ids = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
        sequence_lengths = [self.max_new_tokens] * batch_size
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        current_embeds = final_inputs
        current_mask = final_attention_mask

        for _ in range(self.max_new_tokens):
            outputs = self.model.llm(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                return_dict=True,
                use_cache=False,
            )
            next_logits = outputs.logits[:, -1, :]
            next_logits = next_logits / max(self.temperature, 1e-5)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if self.model.tokenizer.eos_token_id is not None:
                eos_mask = next_token.squeeze(1) == self.model.tokenizer.eos_token_id
                newly_finished = eos_mask & (~finished)
                for idx in (
                    torch.nonzero(newly_finished, as_tuple=False).flatten().tolist()
                ):
                    sequence_lengths[idx] = generated_ids.size(1) + 1
                finished = finished | eos_mask

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            next_embeds = self.model._embed_tokens(next_token).to(
                dtype=self.model.model_dtype
            )
            current_embeds = torch.cat([current_embeds, next_embeds], dim=1)
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(
                        batch_size, 1, dtype=current_mask.dtype, device=self.device
                    ),
                ],
                dim=1,
            )

            if bool(finished.all()):
                break

        for i, is_finished in enumerate(finished.tolist()):
            if not is_finished:
                sequence_lengths[i] = generated_ids.size(1)

        texts = self._decode_texts(generated_ids, sequence_lengths)
        rewards = torch.tensor(
            [
                self._compute_reward(sample.clauses, sample.satisfiable, text)
                for sample, text in zip(batch, texts)
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return generated_ids, sequence_lengths, texts, rewards

    def _score_sequences(
        self,
        batch: Sequence[RLSample],
        generated_ids: Tensor,
        sequence_lengths: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        clauses_batch = [sample.clauses for sample in batch]
        prompts = [sample.prompt for sample in batch]
        (
            final_inputs,
            final_attention_mask,
            _,
            _,
            _,
        ) = self.model._compose_final_inputs(clauses_batch, prompts)

        if generated_ids.size(1) == 0:
            raise ValueError("generated_ids cannot be empty during RL scoring")

        generated_embeds = self.model._embed_tokens(generated_ids).to(
            dtype=self.model.model_dtype
        )
        full_inputs = torch.cat([final_inputs, generated_embeds], dim=1)
        full_attention_mask = torch.cat(
            [
                final_attention_mask,
                torch.ones(
                    generated_ids.size(0),
                    generated_ids.size(1),
                    dtype=final_attention_mask.dtype,
                    device=self.device,
                ),
            ],
            dim=1,
        )
        outputs = self.model.llm(
            inputs_embeds=full_inputs,
            attention_mask=full_attention_mask,
            return_dict=True,
            use_cache=False,
        )
        context_len = final_inputs.size(1)
        rollout_logits = outputs.logits[
            :, context_len - 1 : context_len - 1 + generated_ids.size(1), :
        ]
        rollout_log_probs = torch.log_softmax(rollout_logits, dim=-1)
        token_log_probs = rollout_log_probs.gather(
            2, generated_ids.unsqueeze(-1)
        ).squeeze(-1)

        token_mask = torch.zeros_like(token_log_probs, dtype=torch.float32)
        for i, length in enumerate(sequence_lengths):
            token_mask[i, :length] = 1.0

        return token_log_probs * token_mask, token_mask

    def rollout(self, batch: Sequence[RLSample]) -> RolloutBatch:
        generated_ids, sequence_lengths, texts, rewards = self._sample_sequences(batch)
        token_log_probs, token_mask = self._score_sequences(
            batch=batch,
            generated_ids=generated_ids,
            sequence_lengths=sequence_lengths,
        )

        return RolloutBatch(
            texts=texts,
            token_ids=generated_ids,
            sequence_lengths=sequence_lengths,
            token_log_probs=token_log_probs,
            token_mask=token_mask,
            rewards=rewards,
        )

    def compute_loss(self, rollout: RolloutBatch) -> tuple[Tensor, dict[str, float]]:
        token_counts = rollout.token_mask.sum(dim=1).clamp_min(1.0)
        sequence_log_probs = rollout.token_log_probs.sum(dim=1) / token_counts
        baseline = self.baseline.update(rollout.rewards)
        advantages = rollout.rewards - baseline
        loss = -(advantages.detach() * sequence_log_probs).mean()

        with torch.no_grad():
            metrics = {
                "loss": float(loss.item()),
                "reward_mean": float(rollout.rewards.mean().item()),
                "reward_std": float(rollout.rewards.std(unbiased=False).item()),
                "advantage_mean": float(advantages.mean().item()),
                "token_count_mean": float(token_counts.mean().item()),
            }
        return loss, metrics

    def train_epoch(
        self, dataloader: DataLoader[RLSample], epoch: int, log_every: int
    ) -> None:
        self.model.train()
        for step, batch in enumerate(dataloader, start=1):
            rollout = self.rollout(batch)
            loss, metrics = self.compute_loss(rollout)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if step % log_every == 0 or step == 1:
                preview = rollout.texts[0].replace("\n", " ")[:120]
                print(
                    f"epoch={epoch} step={step} loss={metrics['loss']:.4f} "
                    f"reward={metrics['reward_mean']:.4f} tokens={metrics['token_count_mean']:.2f}"
                )
                print(
                    f"sample_reward={rollout.rewards[0].item():.3f} sample_text={preview}"
                )

    def save_checkpoint(self, name: str) -> None:
        save_path = self.save_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path / "model.pt")
        self.model.tokenizer.save_pretrained(save_path)
        with (save_path / "trainer_state.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_path": self.model.config.model_path,
                    "prefix_tokens": self.model.config.prefix_tokens,
                    "num_reasoning_steps": self.model.config.num_reasoning_steps,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


def collate_fn(batch: Sequence[RLSample]) -> list[RLSample]:
    return list(batch)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Policy-gradient RL training for LatentSAT"
    )
    parser.add_argument("--train-file", type=str, default="data/clean_data.jsonl")
    parser.add_argument("--save-dir", type=str, default="checkpoints/rl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--freeze-llm", action="store_true")
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--num-reasoning-steps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = CNFRLDataset(args.train_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = build_model(
        LatentSATConfig(
            freeze_llm=args.freeze_llm,
            prefix_tokens=args.prefix_tokens,
            num_reasoning_steps=args.num_reasoning_steps,
        )
    )
    trainer = PolicyGradientTrainer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        grad_clip=args.grad_clip,
        save_dir=args.save_dir,
        device=args.device,
    )

    for epoch in range(1, args.epochs + 1):
        trainer.train_epoch(dataloader, epoch=epoch, log_every=args.log_every)
        trainer.save_checkpoint(f"epoch_{epoch}")


if __name__ == "__main__":
    main()
