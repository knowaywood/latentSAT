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


class SupervisedTrainer:
    def __init__(
        self,
        model: LatentSATModel,
        lr: float,
        weight_decay: float,
        grad_clip: float,
        save_dir: str | Path,
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
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def _sample_loss(self, sample: ListSample) -> tuple[Tensor, dict[str, float]]:
        output = self.model(sample.clauses)
        target = torch.tensor(
            sample.answer,
            dtype=output.sat_logits.dtype,
            device=self.device,
        )

        sat_target = target[:1].view_as(output.sat_logits[0])
        sat_loss = F.binary_cross_entropy_with_logits(
            output.sat_logits[0],
            sat_target,
        )

        assignment_loss = torch.zeros((), device=self.device, dtype=sat_loss.dtype)
        if len(sample.answer) > 1:
            assign_target = target[1:].view(-1)
            assign_logits = output.assignment_logits[0, : assign_target.numel()]
            assignment_loss = F.binary_cross_entropy_with_logits(
                assign_logits,
                assign_target,
            )

        # Encourage the model to eventually trigger decoding during pretraining.
        stop_loss = F.binary_cross_entropy(
            output.halt_probs.max(dim=1).values,
            torch.ones(1, device=self.device, dtype=output.halt_probs.dtype),
        )

        loss = sat_loss + assignment_loss + 0.1 * stop_loss

        with torch.no_grad():
            pred = self.model.decode_solution(
                output.sat_logits,
                output.assignment_logits,
                num_vars=max(len(sample.answer) - 1, 0),
            )[0]
            exact_match = float(pred == sample.answer)
            metrics = {
                "loss": float(loss.item()),
                "sat_loss": float(sat_loss.item()),
                "assignment_loss": float(assignment_loss.item()),
                "stop_loss": float(stop_loss.item()),
                "exact_match": exact_match,
            }
        return loss, metrics

    def train_epoch(
        self, dataloader: DataLoader[list[ListSample]], epoch: int, log_every: int
    ) -> None:
        self.model.train()
        for step, batch in enumerate(dataloader, start=1):
            self.optimizer.zero_grad(set_to_none=True)

            losses: list[Tensor] = []
            metric_buffer: list[dict[str, float]] = []
            for sample in batch:
                sample_loss, sample_metrics = self._sample_loss(sample)
                losses.append(sample_loss)
                metric_buffer.append(sample_metrics)

            loss = torch.stack(losses).mean()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if step % log_every == 0 or step == 1:
                mean_metrics = {
                    key: sum(m[key] for m in metric_buffer) / len(metric_buffer)
                    for key in metric_buffer[0]
                }
                print(
                    f"epoch={epoch} step={step} loss={mean_metrics['loss']:.4f} "
                    f"sat={mean_metrics['sat_loss']:.4f} assign={mean_metrics['assignment_loss']:.4f} "
                    f"stop={mean_metrics['stop_loss']:.4f} exact={mean_metrics['exact_match']:.4f}"
                )

    def save_checkpoint(self, name: str) -> None:
        save_path = self.save_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path / "model.pt")
        with (save_path / "config.json").open("w", encoding="utf-8") as f:
            json.dump(self.model.config.__dict__, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised pretraining for LatentSAT")
    parser.add_argument("--train-file", type=str, default="data/list_data.jsonl")
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints/list_struct/pretrain"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--num-reasoning-steps", type=int, default=8)
    parser.add_argument("--max-literal-index", type=int, default=256)
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

    model = build_model(
        LatentSATConfig(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            prefix_tokens=args.prefix_tokens,
            num_reasoning_steps=args.num_reasoning_steps,
            max_literal_index=args.max_literal_index,
        )
    )
    trainer = SupervisedTrainer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        save_dir=args.save_dir,
        device=args.device,
    )

    for epoch in range(1, args.epochs + 1):
        trainer.train_epoch(dataloader, epoch=epoch, log_every=args.log_every)
        trainer.save_checkpoint(f"epoch_{epoch}")


if __name__ == "__main__":
    main()
