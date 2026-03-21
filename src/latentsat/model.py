from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass
class LatentSATConfig:
    model_path: str = "latent-byte-transformer"
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 1024
    max_literal_index: int = 256
    prefix_tokens: int = 32
    num_reasoning_steps: int = 8
    stop_threshold: float = 0.5
    freeze_llm: bool = False
    layer_norm_eps: float = 1e-5
    dropout: float = 0.0

    def estimated_parameter_count(self) -> int:
        d = self.hidden_size
        ff = self.intermediate_size
        n_layers = self.num_layers
        core = self.max_seq_len * d
        core += n_layers * (4 * d * d + 2 * d * ff)
        core += 2 * self.max_literal_index * d
        core += self.prefix_tokens * d
        core += 8 * d * d
        return core


class FeedForward(nn.Module):
    def __init__(self, config: LatentSATConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: LatentSATConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        attn_out, _ = self.attn(
            self.norm_1(x),
            self.norm_1(x),
            self.norm_1(x),
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm_2(x))
        return x


class LatentReasoner(nn.Module):
    def __init__(self, config: LatentSATConfig) -> None:
        super().__init__()
        self.config = config
        self.position_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        return_dict: bool = True,
        use_cache: bool = False,
    ) -> SimpleNamespace | tuple[Tensor]:
        del use_cache
        seq_len = x.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embed(positions)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        hidden_states = self.final_norm(x)
        if return_dict:
            return SimpleNamespace(hidden_states=hidden_states)
        return (hidden_states,)


class CNFAdapter(nn.Module):
    def __init__(self, config: LatentSATConfig) -> None:
        super().__init__()
        self.config = config
        self.var_embed = nn.Embedding(config.max_literal_index + 1, config.hidden_size)
        self.sign_embed = nn.Embedding(2, config.hidden_size)
        self.literal_proj = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.clause_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.prefix_queries = nn.Parameter(
            torch.randn(config.prefix_tokens, config.hidden_size) * 0.02
        )
        self.prefix_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            batch_first=True,
        )
        self.prefix_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, clauses_batch: Sequence[Sequence[Sequence[int]]]) -> Tensor:
        device = self.prefix_queries.device
        clause_embeddings: list[Tensor] = []
        for clauses in clauses_batch:
            if not clauses:
                clause_embeddings.append(
                    torch.zeros(1, self.config.hidden_size, device=device)
                )
                continue
            encoded_clauses: list[Tensor] = []
            for clause in clauses:
                if not clause:
                    encoded_clauses.append(
                        torch.zeros(self.config.hidden_size, device=device)
                    )
                    continue
                literals = torch.tensor(
                    [min(abs(lit), self.config.max_literal_index) for lit in clause],
                    device=device,
                    dtype=torch.long,
                )
                signs = torch.tensor(
                    [1 if lit > 0 else 0 for lit in clause],
                    device=device,
                    dtype=torch.long,
                )
                lit_repr = torch.cat(
                    [self.var_embed(literals), self.sign_embed(signs)], dim=-1
                )
                clause_repr = self.literal_proj(lit_repr).mean(dim=0)
                encoded_clauses.append(self.clause_norm(clause_repr))
            clause_embeddings.append(torch.stack(encoded_clauses, dim=0))

        max_clauses = max(item.size(0) for item in clause_embeddings)
        padded = torch.zeros(
            len(clause_embeddings),
            max_clauses,
            self.config.hidden_size,
            device=device,
        )
        mask = torch.zeros(len(clause_embeddings), max_clauses, device=device)
        for idx, item in enumerate(clause_embeddings):
            padded[idx, : item.size(0)] = item
            mask[idx, : item.size(0)] = 1

        queries = self.prefix_queries.unsqueeze(0).expand(
            len(clause_embeddings), -1, -1
        )
        refined, _ = self.prefix_attn(
            queries,
            padded,
            padded,
            key_padding_mask=mask == 0,
            need_weights=False,
        )
        return self.prefix_norm(queries + refined)


class DualTrackReasoner(nn.Module):
    def __init__(self, config: LatentSATConfig) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            batch_first=True,
        )
        self.prefix_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.reason_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.reason_ffn = FeedForward(config)
        self.stop_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
        )
        nn.init.constant_(self.stop_head[-1].bias, -2.0)

    def forward(
        self, prefix: Tensor, z_t: Tensor, history: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        refined_prefix, _ = self.cross_attn(
            prefix,
            history,
            history,
            need_weights=False,
        )
        next_prefix = self.prefix_norm(prefix + refined_prefix)
        next_reason = z_t + self.reason_ffn(self.reason_norm(z_t))
        stop_logits = self.stop_head(z_t).squeeze(-1)
        return next_prefix, next_reason, stop_logits


@dataclass
class LatentSATOutput:
    sat_logits: Tensor
    assignment_logits: Tensor
    halt_logits: Tensor
    halt_probs: Tensor
    halt_steps: Tensor
    decode_ready: Tensor
    final_latent_states: Tensor
    final_attention_mask: Tensor
    reasoning_states: Tensor
    structured_output: list[int] | None = None


class LatentSATModel(nn.Module):
    def __init__(self, config: LatentSATConfig) -> None:
        super().__init__()
        self.config = config
        self.cnf_adapter = CNFAdapter(config)
        self.reasoner = LatentReasoner(config)
        self.dual_track = DualTrackReasoner(config)
        self.start_reason_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_size) * 0.02
        )
        self.sat_head = nn.Linear(config.hidden_size, 1)
        self.assignment_head = nn.Linear(config.hidden_size, config.max_literal_index)
        self.model_dtype = self.start_reason_token.dtype
        if config.freeze_llm:
            for param in self.reasoner.parameters():
                param.requires_grad = False

        if config.estimated_parameter_count() >= 50_000_000:
            raise ValueError("Model exceeds the requested 50M parameter budget")

    def _validate_clauses(self, clauses: Sequence[Sequence[int]]) -> list[list[int]]:
        if not isinstance(clauses, Sequence):
            raise TypeError("clauses must be a list[list[int]]")

        normalized: list[list[int]] = []
        for clause in clauses:
            if not isinstance(clause, Sequence) or isinstance(clause, (str, bytes)):
                raise TypeError("clauses must be a list[list[int]]")
            normalized_clause: list[int] = []
            for lit in clause:
                if not isinstance(lit, int):
                    raise TypeError("each literal in clauses must be int")
                normalized_clause.append(int(lit))
            normalized.append(normalized_clause)
        return normalized

    def _run_reasoning(
        self, clauses_batch: Sequence[Sequence[Sequence[int]]]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = next(self.parameters()).device
        prefix = self.cnf_adapter(clauses_batch).to(dtype=self.model_dtype)
        batch_size = prefix.size(0)
        start_reason = self.start_reason_token.expand(batch_size, -1, -1).to(
            dtype=self.model_dtype
        )

        history_states: list[Tensor] = []
        halt_logits: list[Tensor] = []
        current_reason_state = start_reason
        halt_steps = torch.full(
            (batch_size,),
            fill_value=self.config.num_reasoning_steps,
            dtype=torch.long,
            device=device,
        )
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(self.config.num_reasoning_steps):
            reasoning_context = (
                torch.cat(history_states, dim=1) if history_states else current_reason_state
            )
            llm_input = torch.cat([prefix, reasoning_context], dim=1)
            attention_mask = torch.ones(
                llm_input.size(0), llm_input.size(1), dtype=torch.long, device=device
            )
            llm_out = self.reasoner(
                x=llm_input,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )
            z_t = llm_out.hidden_states[:, -1:, :]
            next_prefix, next_reason, stop_logit = self.dual_track(
                prefix=prefix,
                z_t=z_t,
                history=torch.cat([reasoning_context, z_t], dim=1),
            )

            current_halt = (
                torch.sigmoid(stop_logit.squeeze(-1)) >= self.config.stop_threshold
            )
            new_halt = current_halt & (~halted)
            halt_steps = torch.where(
                new_halt,
                torch.full_like(halt_steps, step + 1),
                halt_steps,
            )
            active_mask = (~halted).view(batch_size, 1, 1)
            prefix = torch.where(active_mask, next_prefix, prefix)
            current_reason_state = torch.where(
                active_mask,
                next_reason,
                current_reason_state,
            )
            history_states.append(current_reason_state)
            halt_logits.append(stop_logit)
            halted = halted | current_halt

        reasoning_states = torch.cat(history_states, dim=1)
        halt_logits_tensor = torch.cat(halt_logits, dim=1)
        halt_probs = torch.sigmoid(halt_logits_tensor)
        return prefix, reasoning_states, halt_logits_tensor, halt_probs, halt_steps

    def _compose_final_inputs(
        self,
        clauses_batch: Sequence[Sequence[Sequence[int]]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        prefix, reasoning_states, halt_logits, halt_probs, halt_steps = (
            self._run_reasoning(clauses_batch=clauses_batch)
        )
        final_inputs = torch.cat([prefix, reasoning_states], dim=1)
        final_attention_mask = torch.ones(
            prefix.size(0),
            prefix.size(1) + reasoning_states.size(1),
            dtype=torch.long,
            device=prefix.device,
        )
        return (
            final_inputs.to(dtype=self.model_dtype),
            final_attention_mask,
            reasoning_states,
            halt_probs,
            halt_steps,
        )

    def forward(
        self,
        clauses_batch: Sequence[Sequence[int]],
    ) -> LatentSATOutput:
        normalized_clauses = self._validate_clauses(clauses_batch)
        normalized_clauses_batch = [normalized_clauses]
        (
            final_inputs,
            final_attention_mask,
            reasoning_states,
            halt_probs,
            halt_steps,
        ) = self._compose_final_inputs(normalized_clauses_batch)

        last_reason = reasoning_states[:, -1, :]
        sat_logits = self.sat_head(last_reason)
        assignment_logits = self.assignment_head(last_reason)
        halt_logits = torch.logit(halt_probs.clamp(1e-6, 1 - 1e-6))
        num_vars = max(
            (
                abs(lit)
                for clauses in normalized_clauses_batch
                for clause in clauses
                for lit in clause
            ),
            default=0,
        )
        raw_structured_output = self.decode_solution(
            sat_logits=sat_logits,
            assignment_logits=assignment_logits,
            num_vars=num_vars,
        )
        decode_ready = (halt_probs >= self.config.stop_threshold).any(dim=1)
        structured_output = (
            raw_structured_output[0] if bool(decode_ready[0].item()) else None
        )

        return LatentSATOutput(
            sat_logits=sat_logits,
            assignment_logits=assignment_logits,
            halt_logits=halt_logits,
            halt_probs=halt_probs,
            halt_steps=halt_steps,
            decode_ready=decode_ready,
            final_latent_states=final_inputs,
            final_attention_mask=final_attention_mask,
            reasoning_states=reasoning_states,
            structured_output=structured_output,
        )

    def decode_solution(
        self, sat_logits: Tensor, assignment_logits: Tensor, num_vars: int
    ) -> list[list[int]]:
        sat_values = (torch.sigmoid(sat_logits.squeeze(-1)) >= 0.5).to(torch.int64)
        assignment_values = (torch.sigmoid(assignment_logits[:, :num_vars]) >= 0.5).to(
            torch.int64
        )
        outputs: list[list[int]] = []
        for sat, assignment in zip(sat_values.tolist(), assignment_values.tolist()):
            outputs.append([int(sat), *[int(v) for v in assignment]])
        return outputs


def build_model(config: LatentSATConfig | None = None) -> LatentSATModel:
    if config is None:
        config = LatentSATConfig()
    return LatentSATModel(config)


if __name__ == "__main__":
    model = build_model()
    demo_clauses = [
        [-5, 3, 2],
        [-4, 1, -2],
        [-4, -3, -2],
        [-2, -1, -4],
        [3, 4, -1],
        [-5, 4, 2],
        [-2, 5, 3],
        [1, -5, -2],
        [-1, 4, -5],
        [-3, -1, -5],
    ]
    outputs = model(demo_clauses)

    print("sat_logits shape:", tuple(outputs.sat_logits.shape))
    print("assignment_logits shape:", tuple(outputs.assignment_logits.shape))
    print("halt_probs shape:", tuple(outputs.halt_probs.shape))
    print("halt_steps:", outputs.halt_steps.tolist())
    print("decode_ready:", outputs.decode_ready.tolist())
    print("structured_output:", outputs.structured_output)
