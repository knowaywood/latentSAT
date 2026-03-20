from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = "./model/Qwen3-0.6B"
DEFAULT_SYSTEM_PROMPT = (
    "你是一个逻辑推理助手。请结合给定的 CNF 前缀特征与文本提示进行分析，"
    "输出清晰、严格的 SAT 推理结论。"
)


@dataclass
class LatentSATOutput:
    logits: Tensor
    final_prefix: Tensor
    reasoning_states: list[Tensor]
    prompt_input_ids: Tensor
    attention_mask: Tensor


@dataclass
class LatentSATConfig:
    model_path: str = DEFAULT_MODEL_PATH
    prefix_tokens: int = 32
    num_reasoning_steps: int = 4
    clause_hidden_dim: int = 256
    adapter_heads: int = 8
    adapter_layers: int = 2
    dropout: float = 0.1
    freeze_llm: bool = False
    max_prompt_length: int = 512
    max_variables: int = 512
    max_clause_length: int = 16
    max_clause_count: int = 512


class ResidualMLP(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        return self.dropout(y)


class CNFAdapter(nn.Module):
    """Encode variable-length CNF clauses into a fixed number of prefix tokens."""

    def __init__(
        self,
        hidden_size: int,
        prefix_tokens: int,
        clause_hidden_dim: int,
        adapter_heads: int,
        adapter_layers: int,
        dropout: float,
        max_variables: int = 512,
        max_clause_length: int = 16,
        max_clause_count: int = 512,
    ) -> None:
        super().__init__()
        self.clause_hidden_dim = clause_hidden_dim
        self.max_variables = max_variables
        self.max_clause_length = max_clause_length
        self.max_clause_count = max_clause_count

        self.variable_embedding = nn.Embedding(max_variables + 1, clause_hidden_dim)
        self.sign_embedding = nn.Embedding(3, clause_hidden_dim)
        self.literal_position_embedding = nn.Embedding(
            max_clause_length, clause_hidden_dim
        )
        self.clause_position_embedding = nn.Embedding(
            max_clause_count, clause_hidden_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=clause_hidden_dim,
            nhead=adapter_heads,
            dim_feedforward=clause_hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.clause_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=adapter_layers
        )
        self.to_hidden = nn.Sequential(
            nn.LayerNorm(clause_hidden_dim),
            nn.Linear(clause_hidden_dim, hidden_size),
        )
        self.prefix_queries = nn.Parameter(torch.randn(prefix_tokens, hidden_size))
        self.prefix_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=adapter_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.prefix_norm = nn.LayerNorm(hidden_size)

    def _empty_clause_state(self, device: torch.device) -> Tensor:
        return torch.zeros(1, self.clause_hidden_dim, device=device)

    def _encode_single_cnf(self, clauses: Sequence[Sequence[int]]) -> Tensor:
        device = self.prefix_queries.device
        if not clauses:
            return self._empty_clause_state(device)

        clause_vectors = []
        for clause_idx, clause in enumerate(clauses[: self.max_clause_count]):
            if not clause:
                continue

            vars_idx = []
            sign_idx = []
            pos_idx = []
            for lit_pos, literal in enumerate(clause[: self.max_clause_length]):
                vars_idx.append(min(abs(int(literal)), self.max_variables))
                sign_idx.append(1 if literal > 0 else 2)
                pos_idx.append(lit_pos)

            var_tensor = torch.tensor(vars_idx, dtype=torch.long, device=device)
            sign_tensor = torch.tensor(sign_idx, dtype=torch.long, device=device)
            pos_tensor = torch.tensor(pos_idx, dtype=torch.long, device=device)
            clause_pos_tensor = torch.full_like(
                pos_tensor, fill_value=min(clause_idx, self.max_clause_count - 1)
            )

            literal_states = (
                self.variable_embedding(var_tensor)
                + self.sign_embedding(sign_tensor)
                + self.literal_position_embedding(pos_tensor)
                + self.clause_position_embedding(clause_pos_tensor)
            )
            clause_vectors.append(literal_states.mean(dim=0))

        if not clause_vectors:
            return self._empty_clause_state(device)

        clause_tensor = torch.stack(clause_vectors, dim=0).unsqueeze(0)
        encoded = self.clause_encoder(clause_tensor)
        return encoded.squeeze(0)

    def forward(self, batch_clauses: Sequence[Sequence[Sequence[int]]]) -> Tensor:
        if not batch_clauses:
            raise ValueError("batch_clauses cannot be empty")

        batch_clause_states = [
            self._encode_single_cnf(clauses) for clauses in batch_clauses
        ]
        max_clauses = max(state.size(0) for state in batch_clause_states)
        clause_dim = batch_clause_states[0].size(-1)
        device = batch_clause_states[0].device

        padded = torch.zeros(
            len(batch_clause_states), max_clauses, clause_dim, device=device
        )
        key_padding_mask = torch.ones(
            len(batch_clause_states), max_clauses, device=device, dtype=torch.bool
        )
        for i, state in enumerate(batch_clause_states):
            padded[i, : state.size(0)] = state
            key_padding_mask[i, : state.size(0)] = False

        clause_hidden = self.to_hidden(padded)
        queries = self.prefix_queries.unsqueeze(0).expand(
            len(batch_clause_states), -1, -1
        )
        prefix, _ = self.prefix_attention(
            query=queries,
            key=clause_hidden,
            value=clause_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.prefix_norm(prefix + queries)


class DualTrackBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.prefix_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.reasoning_ffn = ResidualMLP(hidden_size, dropout)

    def forward(self, prefix: Tensor, reasoning_trace: Tensor) -> tuple[Tensor, Tensor]:
        normalized_prefix = self.prefix_norm(prefix)
        refined_prefix, _ = self.cross_attn(
            query=normalized_prefix,
            key=reasoning_trace,
            value=reasoning_trace,
            need_weights=False,
        )
        next_prefix = self.output_norm(prefix + refined_prefix)
        latest_z = reasoning_trace[:, -1:, :]
        next_reasoning = latest_z + self.reasoning_ffn(latest_z)
        return next_prefix, next_reasoning


class LatentSATModel(nn.Module):
    def __init__(self, config: LatentSATConfig | None = None) -> None:
        super().__init__()
        self.config = config or LatentSATConfig()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        self.hidden_size = int(self.llm.config.hidden_size)
        self.model_dtype = next(self.llm.parameters()).dtype
        if not torch.cuda.is_available():
            self.llm = self.llm.float()
            self.model_dtype = torch.float32

        if self.config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        self.cnf_adapter = CNFAdapter(
            hidden_size=self.hidden_size,
            prefix_tokens=self.config.prefix_tokens,
            clause_hidden_dim=self.config.clause_hidden_dim,
            adapter_heads=self.config.adapter_heads,
            adapter_layers=self.config.adapter_layers,
            dropout=self.config.dropout,
            max_variables=self.config.max_variables,
            max_clause_length=self.config.max_clause_length,
            max_clause_count=self.config.max_clause_count,
        )
        self.dual_track = DualTrackBlock(
            hidden_size=self.hidden_size,
            num_heads=self.config.adapter_heads,
            dropout=self.config.dropout,
        )
        self.cnf_adapter = self.cnf_adapter.to(dtype=self.model_dtype)
        self.dual_track = self.dual_track.to(dtype=self.model_dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def cnf_to_text(self, clauses: Sequence[Sequence[int]]) -> str:
        clause_strs = []
        for clause in clauses:
            lits = [f"~x{abs(lit)}" if lit < 0 else f"x{lit}" for lit in clause]
            clause_strs.append("(" + " | ".join(lits) + ")")
        return " & ".join(clause_strs) if clause_strs else "EMPTY"

    def build_prompt(
        self, clauses: Sequence[Sequence[int]], prompt: str | None = None
    ) -> str:
        task_prompt = prompt or "请判断该 CNF 是否可满足，并给出最终结论。"
        formula_text = self.cnf_to_text(clauses)
        return (
            f"{DEFAULT_SYSTEM_PROMPT}\n\n"
            f"CNF 公式:\n{formula_text}\n\n"
            f"用户问题:\n{task_prompt}"
        )

    def _tokenize_prompts(self, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        return encoded.input_ids.to(self.device), encoded.attention_mask.to(self.device)

    def _embed_prompt(self, prompt_input_ids: Tensor) -> Tensor:
        return self.llm.get_input_embeddings()(prompt_input_ids)

    def _embed_tokens(self, token_ids: Tensor) -> Tensor:
        return self.llm.get_input_embeddings()(token_ids)

    def _build_attention_mask(
        self,
        batch_size: int,
        prefix_len: int,
        prompt_attention_mask: Tensor,
        reasoning_len: int,
        generated_len: int = 0,
    ) -> Tensor:
        prefix_mask = torch.ones(
            batch_size,
            prefix_len,
            dtype=prompt_attention_mask.dtype,
            device=self.device,
        )
        reasoning_mask = torch.ones(
            batch_size,
            reasoning_len,
            dtype=prompt_attention_mask.dtype,
            device=self.device,
        )
        generated_mask = torch.ones(
            batch_size,
            generated_len,
            dtype=prompt_attention_mask.dtype,
            device=self.device,
        )
        return torch.cat(
            [prefix_mask, prompt_attention_mask, reasoning_mask, generated_mask],
            dim=1,
        )

    def _run_reasoning_loop(
        self,
        prefix: Tensor,
        prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        reasoning_states: list[Tensor] = []
        latent_trace: list[Tensor] = []
        current_prefix = prefix

        for _ in range(self.config.num_reasoning_steps):
            if reasoning_states:
                reasoning_tensor = torch.cat(reasoning_states, dim=1)
            else:
                reasoning_tensor = torch.empty(
                    prefix.size(0),
                    0,
                    self.hidden_size,
                    device=self.device,
                    dtype=prefix.dtype,
                )

            llm_inputs = [current_prefix, prompt_embeds]
            if reasoning_tensor.size(1) > 0:
                llm_inputs.append(reasoning_tensor)
            inputs_embeds = torch.cat(llm_inputs, dim=1)
            attention_mask = self._build_attention_mask(
                batch_size=prefix.size(0),
                prefix_len=current_prefix.size(1),
                prompt_attention_mask=prompt_attention_mask,
                reasoning_len=reasoning_tensor.size(1),
            )
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            z_t = outputs.hidden_states[-1][:, -1:, :]
            latent_trace.append(z_t)
            z_history = torch.cat(latent_trace, dim=1)
            current_prefix, h_next = self.dual_track(current_prefix, z_history)
            reasoning_states.append(h_next)

        return current_prefix, reasoning_states

    def _compose_final_inputs(
        self,
        batch_clauses: Sequence[Sequence[Sequence[int]]],
        prompts: Sequence[str] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, list[Tensor]]:
        if not batch_clauses:
            raise ValueError("batch_clauses cannot be empty")

        if prompts is None:
            prompts = [self.build_prompt(clauses) for clauses in batch_clauses]
        if len(prompts) != len(batch_clauses):
            raise ValueError("prompts and batch_clauses must have the same batch size")

        prefix = self.cnf_adapter(batch_clauses).to(
            device=self.device, dtype=self.model_dtype
        )
        prompt_input_ids, prompt_attention_mask = self._tokenize_prompts(prompts)
        prompt_embeds = self._embed_prompt(prompt_input_ids).to(dtype=self.model_dtype)
        final_prefix, reasoning_states = self._run_reasoning_loop(
            prefix=prefix,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
        )

        reasoning_tensor = None
        reasoning_len = 0
        if reasoning_states:
            reasoning_tensor = torch.cat(reasoning_states, dim=1)
            reasoning_len = reasoning_tensor.size(1)

        pieces = [final_prefix, prompt_embeds]
        if reasoning_tensor is not None:
            pieces.append(reasoning_tensor)
        final_inputs = torch.cat(pieces, dim=1)
        final_attention_mask = self._build_attention_mask(
            batch_size=final_prefix.size(0),
            prefix_len=final_prefix.size(1),
            prompt_attention_mask=prompt_attention_mask,
            reasoning_len=reasoning_len,
        )
        return (
            final_inputs,
            final_attention_mask,
            final_prefix,
            prompt_input_ids,
            reasoning_states,
        )

    def forward(
        self,
        batch_clauses: Sequence[Sequence[Sequence[int]]],
        prompts: Sequence[str] | None = None,
    ) -> LatentSATOutput:
        (
            final_inputs,
            final_attention_mask,
            final_prefix,
            prompt_input_ids,
            reasoning_states,
        ) = self._compose_final_inputs(batch_clauses, prompts)
        final_outputs = self.llm(
            inputs_embeds=final_inputs,
            attention_mask=final_attention_mask,
            return_dict=True,
            use_cache=False,
        )
        return LatentSATOutput(
            logits=final_outputs.logits,
            final_prefix=final_prefix,
            reasoning_states=reasoning_states,
            prompt_input_ids=prompt_input_ids,
            attention_mask=final_attention_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        batch_clauses: Sequence[Sequence[Sequence[int]]],
        prompts: Sequence[str] | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        do_sample: bool = True,
        **_: Any,
    ) -> list[str]:
        (
            final_inputs,
            final_attention_mask,
            _,
            _,
            _,
        ) = self._compose_final_inputs(batch_clauses, prompts)

        batch_size = final_inputs.size(0)
        generated_ids = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
        current_embeds = final_inputs
        current_mask = final_attention_mask

        for _ in range(max_new_tokens):
            outputs = self.llm(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                return_dict=True,
                use_cache=False,
            )
            next_logits = outputs.logits[:, -1, :]
            if do_sample:
                next_logits = next_logits / max(temperature, 1e-5)
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            next_embeds = self._embed_tokens(next_token)
            current_embeds = torch.cat([current_embeds, next_embeds], dim=1)
            current_mask = self._build_attention_mask(
                batch_size=batch_size,
                prefix_len=0,
                prompt_attention_mask=current_mask,
                reasoning_len=0,
                generated_len=1,
            )

            if self.tokenizer.eos_token_id is not None:
                if bool(
                    torch.all(next_token.squeeze(-1) == self.tokenizer.eos_token_id)
                ):
                    break

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def build_model(config: LatentSATConfig | None = None) -> LatentSATModel:
    return LatentSATModel(config=config)


if __name__ == "__main__":
    demo_clauses = [[[1, -2], [2, 3], [-1, -3]]]
    demo_prompts = ["请判断这个 CNF 是否可满足，并简要说明结论。"]

    model = build_model()
    model.eval()

    with torch.no_grad():
        outputs = model(demo_clauses, prompts=demo_prompts)
        generated_texts = model.generate(
            demo_clauses,
            prompts=demo_prompts,
            max_new_tokens=64,
            do_sample=False,
        )

    print("logits shape:", tuple(outputs.logits.shape))
    print("final prefix shape:", tuple(outputs.final_prefix.shape))
    print("reasoning steps:", len(outputs.reasoning_states))
    print("generated text:")
    print(generated_texts[0])
