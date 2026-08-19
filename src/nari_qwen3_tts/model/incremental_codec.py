"""Stateful incremental execution for the Qwen3-TTS 12 Hz decoder."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from nari_qwen3_tts.contract.codec_state import IncrementalCodecState


class Qwen3TTSIncrementalDecoder:
    """Run only fresh codec frames while preserving full-sequence causality.

    The upstream decoder exposes only a whole-sequence ``forward``. Its
    operations are causal, but two kinds of boundary state are implicit in
    that forward: left-padding histories for Conv1d and the trimmed overlap
    tail for ConvTranspose1d. This adapter materializes both alongside the
    pre-transformer's native KV cache.
    """

    def __init__(self, decoder: torch.nn.Module):
        self.decoder = decoder
        self.samples_per_frame = int(decoder.total_upsample)
        self._validate_layout()

    def _validate_layout(self) -> None:
        if not hasattr(self.decoder, "pre_conv"):
            raise TypeError("Qwen3-TTS decoder is missing pre_conv")
        if not hasattr(self.decoder, "pre_transformer"):
            raise TypeError("Qwen3-TTS decoder is missing pre_transformer")
        if not hasattr(self.decoder, "upsample"):
            raise TypeError("Qwen3-TTS decoder is missing upsample")
        if not hasattr(self.decoder, "decoder"):
            raise TypeError("Qwen3-TTS decoder is missing decoder blocks")

    @staticmethod
    def _stack_state(
        states: list[IncrementalCodecState],
        mapping_name: str,
        key: str | int,
        *,
        shape: tuple[int, int],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for state in states:
            mapping = getattr(state, mapping_name)
            value = mapping.get(key)
            if value is None:
                value = reference.new_zeros(shape)
            elif value.shape != shape:
                raise ValueError(f"invalid incremental state {key!r}: expected {shape}, got {tuple(value.shape)}")
            rows.append(value)
        if len(rows) == 1:
            return rows[0].unsqueeze(0)
        return torch.stack(rows, dim=0)

    @classmethod
    def _causal_conv(
        cls,
        module: torch.nn.Module,
        hidden: torch.Tensor,
        states: list[IncrementalCodecState],
        key: str,
    ) -> torch.Tensor:
        """Execute a stride-one causal Conv1d on fresh positions only."""
        conv = module.conv
        stride = int(conv.stride[0])
        if stride != 1:
            raise ValueError(f"incremental causal Conv1d requires stride 1, got {stride}")
        history_size = int(module.padding)
        history = cls._stack_state(
            states,
            "conv_histories",
            key,
            shape=(hidden.shape[1], history_size),
            reference=hidden,
        )
        combined = torch.cat((history, hidden), dim=-1)
        output = conv(combined)
        if output.shape[-1] != hidden.shape[-1]:
            raise RuntimeError(
                f"incremental causal Conv1d {key!r} produced {output.shape[-1]} positions for {hidden.shape[-1]} inputs"
            )
        if history_size:
            for row, state in enumerate(states):
                state.conv_histories[key] = combined[row, :, -history_size:].detach().clone()
        return output.contiguous()

    @classmethod
    def _causal_transconv(
        cls,
        module: torch.nn.Module,
        hidden: torch.Tensor,
        states: list[IncrementalCodecState],
        key: str,
    ) -> torch.Tensor:
        """Execute a causal ConvTranspose1d and carry its trimmed overlap."""
        conv = module.conv
        stride = int(conv.stride[0])
        overlap_size = int(module.right_pad)
        if overlap_size not in (0, stride):
            raise ValueError(
                f"incremental causal ConvTranspose1d expects zero or one "
                f"stride of overlap, got right_pad={overlap_size}, "
                f"stride={stride}"
            )

        # Bias belongs to each final output position. Keep the carried overlap
        # bias-free so joining it with the next chunk cannot add bias twice.
        expanded = F.conv_transpose1d(
            hidden,
            conv.weight,
            bias=None,
            stride=conv.stride,
            padding=conv.padding,
            output_padding=conv.output_padding,
            groups=conv.groups,
            dilation=conv.dilation,
        )
        if overlap_size:
            overlap = cls._stack_state(
                states,
                "transconv_overlaps",
                key,
                shape=(expanded.shape[1], overlap_size),
                reference=expanded,
            )
            expanded[:, :, :overlap_size] += overlap

        emitted_size = hidden.shape[-1] * stride
        emitted = expanded[:, :, :emitted_size]
        if conv.bias is not None:
            emitted = emitted + conv.bias.view(1, -1, 1)
        if overlap_size:
            for row, state in enumerate(states):
                state.transconv_overlaps[key] = expanded[row, :, emitted_size:].detach().clone()
        return emitted.contiguous()

    @classmethod
    def _convnext(
        cls,
        module: torch.nn.Module,
        hidden: torch.Tensor,
        states: list[IncrementalCodecState],
        key: str,
    ) -> torch.Tensor:
        residual = hidden
        hidden = cls._causal_conv(
            module.dwconv,
            hidden,
            states,
            f"{key}.dwconv",
        )
        hidden = hidden.permute(0, 2, 1)
        hidden = module.norm(hidden)
        hidden = module.pwconv1(hidden)
        hidden = module.act(hidden)
        hidden = module.pwconv2(hidden)
        hidden = module.gamma * hidden
        return residual + hidden.permute(0, 2, 1)

    @classmethod
    def _residual_unit(
        cls,
        module: torch.nn.Module,
        hidden: torch.Tensor,
        states: list[IncrementalCodecState],
        key: str,
    ) -> torch.Tensor:
        residual = hidden
        hidden = module.act1(hidden)
        hidden = cls._causal_conv(
            module.conv1,
            hidden,
            states,
            f"{key}.conv1",
        )
        hidden = module.act2(hidden)
        hidden = cls._causal_conv(
            module.conv2,
            hidden,
            states,
            f"{key}.conv2",
        )
        return hidden + residual

    @staticmethod
    def _rotate_half(hidden: torch.Tensor) -> torch.Tensor:
        first, second = hidden.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def _pre_transformer(  # noqa: PLR0915 - fixed Codec pipeline mirrors checkpoint blocks
        self,
        hidden: torch.Tensor,
        states: list[IncrementalCodecState],
        position_ids: torch.Tensor | None = None,
        context_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decoder transformer with fixed-shape request-local KV."""
        transformer = self.decoder.pre_transformer
        hidden = transformer.input_proj(hidden)
        batch_size = hidden.shape[0]
        fresh_frames = hidden.shape[1]
        if position_ids is None:
            if batch_size == 1:
                position_ids = torch.arange(
                    states[0].frame_position,
                    states[0].frame_position + fresh_frames,
                    dtype=torch.long,
                    device=hidden.device,
                ).unsqueeze(0)
            else:
                position_ids = torch.stack(
                    [
                        torch.arange(
                            state.frame_position,
                            state.frame_position + fresh_frames,
                            dtype=torch.long,
                            device=hidden.device,
                        )
                        for state in states
                    ],
                    dim=0,
                )
        elif position_ids.shape != (batch_size, fresh_frames):
            raise ValueError(
                "incremental position_ids must have shape "
                f"{(batch_size, fresh_frames)}, got "
                f"{tuple(position_ids.shape)}"
            )
        cos, sin = transformer.rotary_emb(hidden, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        sliding_window = int(transformer.config.sliding_window)
        retained_context = max(0, sliding_window - 1)
        if context_lengths is None:
            context_lengths = torch.tensor(
                [state.transformer_context_length for state in states],
                dtype=torch.long,
                device=hidden.device,
            )
        elif context_lengths.shape != (batch_size,):
            raise ValueError(
                f"incremental context_lengths must have shape {(batch_size,)}, got {tuple(context_lengths.shape)}"
            )
        context_lengths = context_lengths.to(
            device=hidden.device,
            dtype=torch.long,
        )
        context_slots = torch.arange(
            retained_context,
            dtype=torch.long,
            device=hidden.device,
        )
        valid_prior = context_slots.unsqueeze(0) >= retained_context - context_lengths.unsqueeze(1)
        prior_positions = position_ids[:, :1] + context_slots.unsqueeze(0) - retained_context
        key_positions = torch.cat(
            (prior_positions, position_ids),
            dim=1,
        )
        valid_keys = torch.cat(
            (
                valid_prior,
                torch.ones(
                    (batch_size, fresh_frames),
                    dtype=torch.bool,
                    device=hidden.device,
                ),
            ),
            dim=1,
        )
        query_positions = position_ids.unsqueeze(-1)
        expanded_key_positions = key_positions.unsqueeze(1)
        attention_mask = (
            valid_keys.unsqueeze(1)
            & (expanded_key_positions <= query_positions)
            & (expanded_key_positions > query_positions - sliding_window)
        ).unsqueeze(1)

        for layer_index, layer in enumerate(transformer.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attention = layer.self_attn
            input_shape = normalized.shape[:-1]
            projected_shape = (*input_shape, -1, attention.head_dim)
            query = attention.q_norm(attention.q_proj(normalized).view(projected_shape)).transpose(1, 2)
            key = attention.k_norm(attention.k_proj(normalized).view(projected_shape)).transpose(1, 2)
            value = (
                attention.v_proj(normalized)
                .view(projected_shape)
                .transpose(
                    1,
                    2,
                )
            )
            query = query * cos + self._rotate_half(query) * sin
            key = key * cos + self._rotate_half(key) * sin

            prior_keys = self._stack_state(
                states,
                "transformer_keys",
                layer_index,
                shape=(
                    key.shape[1],
                    retained_context,
                    key.shape[-1],
                ),
                reference=key,
            )
            prior_values = self._stack_state(
                states,
                "transformer_values",
                layer_index,
                shape=(
                    value.shape[1],
                    retained_context,
                    value.shape[-1],
                ),
                reference=value,
            )
            all_keys = torch.cat((prior_keys, key), dim=2)
            all_values = torch.cat((prior_values, value), dim=2)

            cache_keys = all_keys
            cache_values = all_values

            if attention.num_key_value_groups > 1:
                all_keys = all_keys.repeat_interleave(
                    attention.num_key_value_groups,
                    dim=1,
                )
                all_values = all_values.repeat_interleave(
                    attention.num_key_value_groups,
                    dim=1,
                )
            attended = F.scaled_dot_product_attention(
                query,
                all_keys,
                all_values,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=attention.scaling,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).contiguous()
            attended = attention.o_proj(attended.reshape(*input_shape, -1))
            hidden = residual + layer.self_attn_layer_scale(attended)

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + layer.mlp_layer_scale(hidden)

            for row, state in enumerate(states):
                if retained_context:
                    row_key = cache_keys[
                        row,
                        :,
                        -retained_context:,
                    ]
                    row_value = cache_values[
                        row,
                        :,
                        -retained_context:,
                    ]
                else:
                    row_key = cache_keys[row, :, :0]
                    row_value = cache_values[row, :, :0]
                state.transformer_keys[layer_index] = row_key.detach().clone()
                state.transformer_values[layer_index] = row_value.detach().clone()

        hidden = transformer.norm(hidden)
        hidden = transformer.output_proj(hidden)
        for state in states:
            state.frame_position += fresh_frames
            state.transformer_context_length = min(
                retained_context,
                state.transformer_context_length + fresh_frames,
            )
        return hidden

    def __call__(
        self,
        codes: torch.Tensor,
        states: list[IncrementalCodecState],
        *,
        position_ids: torch.Tensor | None = None,
        context_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode ``(B, codebooks, fresh_frames)`` into fresh waveform samples."""
        if codes.ndim != 3:
            raise ValueError(f"expected codec codes with rank 3, got shape {tuple(codes.shape)}")
        if codes.shape[0] != len(states):
            raise ValueError(f"state count {len(states)} does not match batch {codes.shape[0]}")
        if codes.shape[1] != self.decoder.config.num_quantizers:
            raise ValueError(f"expected {self.decoder.config.num_quantizers} codebooks, got {codes.shape[1]}")
        if codes.shape[2] == 0:
            return next(self.decoder.parameters()).new_empty((codes.shape[0], 1, 0))

        hidden = self.decoder.quantizer.decode(codes)
        hidden = self._causal_conv(
            self.decoder.pre_conv,
            hidden,
            states,
            "pre_conv",
        ).transpose(1, 2)
        hidden = self._pre_transformer(
            hidden,
            states,
            position_ids=position_ids,
            context_lengths=context_lengths,
        ).permute(0, 2, 1)

        for upsample_index, blocks in enumerate(self.decoder.upsample):
            hidden = self._causal_transconv(
                blocks[0],
                hidden,
                states,
                f"upsample.{upsample_index}.transconv",
            )
            hidden = self._convnext(
                blocks[1],
                hidden,
                states,
                f"upsample.{upsample_index}.convnext",
            )

        wav = self._causal_conv(
            self.decoder.decoder[0],
            hidden,
            states,
            "decoder.0",
        )
        final_index = len(self.decoder.decoder) - 1
        for block_index, block in enumerate(
            self.decoder.decoder[1:final_index],
            start=1,
        ):
            if hasattr(block, "block"):
                wav = block.block[0](wav)
                wav = self._causal_transconv(
                    block.block[1],
                    wav,
                    states,
                    f"decoder.{block_index}.transconv",
                )
                for unit_index, unit in enumerate(block.block[2:]):
                    wav = self._residual_unit(
                        unit,
                        wav,
                        states,
                        f"decoder.{block_index}.residual.{unit_index}",
                    )
            else:
                wav = block(wav)
        wav = self._causal_conv(
            self.decoder.decoder[final_index],
            wav,
            states,
            f"decoder.{final_index}",
        )
        expected_samples = codes.shape[2] * self.samples_per_frame
        if wav.shape[-1] != expected_samples:
            raise RuntimeError(f"incremental decoder produced {wav.shape[-1]} samples, expected {expected_samples}")
        return wav.clamp(min=-1, max=1)
