"""Raw Talker and Code Predictor modules for one-GPU Qwen3-TTS."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nari_qwen3_tts.model.config import Qwen3TTSConfig, Qwen3TTSTalkerConfig
from nari_qwen3_tts.model.layers import (
    Attention,
    GatedMLP,
    RMSNorm,
    qwen3_tts_add_rmsnorm,
    qwen3_tts_rmsnorm,
)


class ResizeMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.silu(self.linear_fc1(inputs)))


class Qwen3TTSTalkerLayer(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        attention_context: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run attention + MLP from an already input-normalized state.

        The enclosing backbone carries the MLP residual into the next layer's
        input norm so that boundary can use one fused add + RMSNorm launch.
        """

        hidden_states = self.self_attn(hidden_states, attention_context)
        hidden_states, residual = qwen3_tts_add_rmsnorm(
            hidden_states,
            residual,
            self.post_attention_layernorm,
            use_cuda_kernel=self.mlp.cuda_optimized_math_enabled,
        )
        return self.mlp(hidden_states), residual


class Qwen3TTSTalkerInnerModel(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Qwen3TTSTalkerLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_embeds: torch.Tensor, attention_context: object) -> torch.Tensor:
        residual = input_embeds
        hidden_states = self.layers[0].input_layernorm(input_embeds)
        for layer_index, layer in enumerate(self.layers):
            attention_context.select_layer(layer_index)
            hidden_states, residual = layer(hidden_states, residual, attention_context)
            next_norm = (
                self.norm if layer_index + 1 == len(self.layers) else self.layers[layer_index + 1].input_layernorm
            )
            hidden_states, residual = qwen3_tts_add_rmsnorm(
                hidden_states,
                residual,
                next_norm,
                use_cuda_kernel=layer.mlp.cuda_optimized_math_enabled,
            )
        attention_context.advance_sequence_lengths()
        return hidden_states


class Qwen3TTSTalkerModel(nn.Module):
    """Talker with parameter names matching the Hugging Face checkpoint."""

    def __init__(self, config: Qwen3TTSConfig) -> None:
        super().__init__()
        talker = config.talker
        self.config = talker
        self.model = Qwen3TTSTalkerInnerModel(talker)
        self.text_projection = ResizeMLP(talker.text_hidden_size, talker.text_hidden_size, talker.hidden_size)
        self.codec_head = nn.Linear(talker.hidden_size, talker.vocab_size, bias=False)
        self.register_buffer("_projected_text_embedding", None, persistent=False)

    @property
    def device(self) -> torch.device:
        return self.model.codec_embedding.weight.device

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.codec_embedding

    def get_text_embeddings(self) -> nn.Embedding:
        return self.model.text_embedding

    @torch.no_grad()
    def initialize_projected_text_embedding_cache(self) -> None:
        projected = self.text_projection(self.get_text_embeddings().weight)
        expected = (self.config.text_vocab_size, self.config.hidden_size)
        if projected.shape != expected:
            raise RuntimeError(f"projected text embedding shape {tuple(projected.shape)} != {expected}")
        self._projected_text_embedding = projected.contiguous()

    def get_projected_text_embedding_cache(self) -> torch.Tensor:
        if self._projected_text_embedding is None:
            raise RuntimeError("projected text embedding cache has not been initialized")
        return self._projected_text_embedding

    def forward(self, input_embeds: torch.Tensor, attention_context: object) -> torch.Tensor:
        return self.model(input_embeds, attention_context)


class Qwen3TTSCodePredictorLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor, attention_context: object) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, attention_context)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class Qwen3TTSCodePredictorInnerModel(nn.Module):
    def __init__(self, config: Qwen3TTSConfig) -> None:
        super().__init__()
        predictor = config.code_predictor
        self.layers = nn.ModuleList(Qwen3TTSCodePredictorLayer(predictor) for _ in range(predictor.num_hidden_layers))
        self.norm = RMSNorm(predictor.hidden_size, eps=predictor.rms_norm_eps)
        self.codec_embedding = nn.ModuleList(
            nn.Embedding(predictor.vocab_size, config.talker.hidden_size) for _ in range(config.num_code_groups - 1)
        )

    def forward(self, hidden_states: torch.Tensor, attention_context: object) -> torch.Tensor:
        for layer_index, layer in enumerate(self.layers):
            attention_context.select_layer(layer_index)
            hidden_states = layer(hidden_states, attention_context)
        attention_context.advance_sequence_lengths()
        return self.norm(hidden_states)


class Qwen3TTSCodePredictor(nn.Module):
    """Residual-code predictor with checkpoint-compatible parameters."""

    def __init__(self, config: Qwen3TTSConfig) -> None:
        super().__init__()
        predictor = config.code_predictor
        residual_group_count = config.num_code_groups - 1
        self.model = Qwen3TTSCodePredictorInnerModel(config)
        self.lm_head = nn.ModuleList(
            nn.Linear(predictor.hidden_size, predictor.vocab_size, bias=False) for _ in range(residual_group_count)
        )
        self.register_buffer(
            "lm_head_weight",
            torch.empty(residual_group_count, predictor.vocab_size, predictor.hidden_size),
            persistent=False,
        )
        self.small_to_mtp_projection = nn.Linear(config.talker.hidden_size, predictor.hidden_size, bias=True)

    @property
    def codec_embedding(self) -> nn.ModuleList:
        return self.model.codec_embedding

    def get_embedding(self, group_index: int) -> nn.Embedding:
        if not 0 < group_index <= len(self.model.codec_embedding):
            raise IndexError(group_index)
        return self.model.codec_embedding[group_index - 1]

    def get_lm_head(self, group_index: int) -> nn.Linear:
        if not 0 < group_index <= len(self.lm_head):
            raise IndexError(group_index)
        return self.lm_head[group_index - 1]

    @torch.no_grad()
    def consolidate_stacked_weights(self) -> None:
        self.lm_head_weight = torch.stack([head.weight for head in self.lm_head], dim=0).contiguous()

    def forward(self, hidden_states: torch.Tensor, attention_context: object) -> torch.Tensor:
        return self.model(hidden_states, attention_context)

    @staticmethod
    def _apply_rope(
        values: torch.Tensor,
        positions: torch.Tensor,
        *,
        rope_theta: float,
    ) -> torch.Tensor:
        """Apply the non-interleaved Qwen rotary convention."""

        head_dim = values.shape[-1]
        if head_dim % 2:
            raise ValueError("rotary head dimension must be even")
        frequencies = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, device=values.device, dtype=torch.float32) / head_dim)
        )
        angles = positions.to(torch.float32).unsqueeze(-1) * frequencies
        cos = torch.cat((angles.cos(), angles.cos()), dim=-1).unsqueeze(2)
        sin = torch.cat((angles.sin(), angles.sin()), dim=-1).unsqueeze(2)
        first, second = values.float().chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return (values.float() * cos + rotated * sin).to(values.dtype)

    @staticmethod
    def _depth_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic mathematical reference for the optimized kernel."""

        if query.shape[1] != key.shape[1]:
            repeat = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
        scores.mul_(query.shape[-1] ** -0.5)
        scores.masked_fill_(~causal_mask, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1).to(value.dtype)
        return torch.matmul(probabilities, value)

    def forward_depth_unrolled(  # noqa: PLR0915 - fixed 15-step model path stays explicit
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        cache_pos: int,
        *,
        inputs_are_projected: bool = False,
        apply_final_norm: bool = True,
    ) -> torch.Tensor:
        """Run one bounded Code Predictor depth step against dense frame-local KV.

        On CUDA the QK-norm/rotary/cache-write boundary and the short-cache
        attention run as fused Triton kernels; every other device uses the
        explicit PyTorch math below. Both paths share this model loop.
        """

        if inputs_embeds.ndim != 3:
            raise ValueError("Code Predictor inputs must have shape (rows, tokens, hidden)")
        if position_ids.shape != inputs_embeds.shape[:2]:
            raise ValueError("position_ids must match Code Predictor rows and tokens")
        if not inputs_are_projected:
            inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
        hidden_states = inputs_embeds
        rows, token_count = hidden_states.shape[:2]
        valid_length = cache_pos + token_count
        expected_prefix = (len(self.model.layers), rows, 2)
        if kv_cache.ndim != 6 or tuple(kv_cache.shape[:3]) != expected_prefix:
            raise ValueError("dense Code Predictor KV cache has an invalid shape")
        if valid_length > kv_cache.shape[3]:
            raise ValueError("dense Code Predictor KV cache is too short")

        optimized = all(layer.mlp.cuda_optimized_math_enabled for layer in self.model.layers)
        fused = (
            optimized
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
        )
        if fused:
            from nari_qwen3_tts.model.kernels import (
                decode_attn_nhd_short_cache,
                paired_qk_norm_rope_,
                paired_qk_norm_rope_dense_cache_,
            )

            flat_positions = position_ids.reshape(-1)
            for layer_index, layer in enumerate(self.model.layers):
                attention = layer.self_attn
                residual = hidden_states
                hidden_states = qwen3_tts_rmsnorm(hidden_states, layer.input_layernorm)
                total_tokens = rows * token_count
                query, key, value = attention.project_qkv(hidden_states.reshape(total_tokens, -1))
                query = query.view(total_tokens, attention.num_heads, attention.head_dim)
                key = key.view(total_tokens, attention.num_kv_heads, attention.head_dim)
                value = value.view(total_tokens, attention.num_kv_heads, attention.head_dim)
                if token_count == 1:
                    # The decode step folds the cache write into the same launch.
                    query, key = paired_qk_norm_rope_dense_cache_(
                        query,
                        key,
                        value,
                        attention.q_norm.weight,
                        attention.k_norm.weight,
                        flat_positions,
                        attention.q_norm.variance_epsilon,
                        float(attention.rope_theta),
                        kv_cache[layer_index, :, 0],
                        kv_cache[layer_index, :, 1],
                        cache_pos,
                    )
                else:
                    query, key = paired_qk_norm_rope_(
                        query,
                        key,
                        attention.q_norm.weight,
                        attention.k_norm.weight,
                        flat_positions,
                        attention.q_norm.variance_epsilon,
                        float(attention.rope_theta),
                    )
                query = query.view(rows, token_count, attention.num_heads, attention.head_dim)
                key = key.view(rows, token_count, attention.num_kv_heads, attention.head_dim)
                value = value.view(rows, token_count, attention.num_kv_heads, attention.head_dim)
                if token_count != 1:
                    kv_cache[layer_index, :, 0, cache_pos:valid_length].copy_(key)
                    kv_cache[layer_index, :, 1, cache_pos:valid_length].copy_(value)
                attended = decode_attn_nhd_short_cache(
                    query,
                    kv_cache[layer_index, :, 0],
                    kv_cache[layer_index, :, 1],
                    valid_length,
                )
                hidden_states = residual + attention.o_proj(
                    attended.reshape(rows, token_count, -1)
                )
                residual = hidden_states
                hidden_states = qwen3_tts_rmsnorm(hidden_states, layer.post_attention_layernorm)
                hidden_states = residual + layer.mlp(hidden_states)
            return (
                qwen3_tts_rmsnorm(hidden_states, self.model.norm)
                if apply_final_norm
                else hidden_states
            )

        query_positions = cache_pos + torch.arange(token_count, device=hidden_states.device)
        key_positions = torch.arange(valid_length, device=hidden_states.device)
        causal_mask = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)).view(
            1,
            1,
            token_count,
            valid_length,
        )

        for layer_index, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = qwen3_tts_rmsnorm(
                hidden_states,
                layer.input_layernorm,
                use_cuda_kernel=layer.mlp.cuda_optimized_math_enabled,
            )
            attention = layer.self_attn
            flat = hidden_states.reshape(rows * token_count, -1)
            query, key, value = attention.project_qkv(flat)
            query = attention.q_norm(query).view(rows, token_count, attention.num_heads, attention.head_dim)
            key = attention.k_norm(key).view(rows, token_count, attention.num_kv_heads, attention.head_dim)
            value = value.view(rows, token_count, attention.num_kv_heads, attention.head_dim)
            query = self._apply_rope(query, position_ids, rope_theta=attention.rope_theta)
            key = self._apply_rope(key, position_ids, rope_theta=attention.rope_theta)
            kv_cache[layer_index, :, 0, cache_pos:valid_length].copy_(key)
            kv_cache[layer_index, :, 1, cache_pos:valid_length].copy_(value)
            all_keys = kv_cache[layer_index, :, 0, :valid_length].transpose(1, 2)
            all_values = kv_cache[layer_index, :, 1, :valid_length].transpose(1, 2)
            queries = query.transpose(1, 2)
            attended = self._depth_attention(
                queries,
                all_keys,
                all_values,
                causal_mask,
            )
            attended = attended.transpose(1, 2).reshape(rows, token_count, -1)
            hidden_states = residual + attention.o_proj(attended)
            residual = hidden_states
            hidden_states = qwen3_tts_rmsnorm(
                hidden_states,
                layer.post_attention_layernorm,
                use_cuda_kernel=layer.mlp.cuda_optimized_math_enabled,
            )
            hidden_states = residual + layer.mlp(hidden_states)

        return (
            qwen3_tts_rmsnorm(
                hidden_states,
                self.model.norm,
                use_cuda_kernel=optimized,
            )
            if apply_final_norm
            else hidden_states
        )


__all__ = ["Qwen3TTSCodePredictor", "Qwen3TTSTalkerModel", "ResizeMLP"]
