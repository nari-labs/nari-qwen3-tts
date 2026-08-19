"""Convert batched stage inputs into the production per-request row inputs.

The runtime always drives the executors through direct static staging, so the row
inputs are the only ingress production uses. Tests build their fixtures as
batched tensors because that reads better, then convert here, so what runs
under test is the code that runs in production.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from nari_qwen3_tts.executor import (
    CodecExecutionRow,
    CodePredictorExecutionRow,
    TalkerDecodeExecutionRow,
    TalkerPrefillExecutionRow,
    TalkerSamplingExecutionRow,
)


def cp_rows(values) -> tuple[CodePredictorExecutionRow, ...]:
    return tuple(
        CodePredictorExecutionRow(
            layer0_token=values.layer0_token[row],
            past_hidden=values.past_hidden[row],
            temperature=float(values.temperature[row]),
            top_k=int(values.top_k[row]),
            top_p=float(values.top_p[row]),
            seed=int(values.seed[row]),
            offsets=tuple(int(offset) for offset in values.offsets[row]),
            position_ids=None if values.position_ids is None else values.position_ids[row],
        )
        for row in range(values.layer0_token.shape[0])
    )


def codec_rows(frames: torch.Tensor, states) -> tuple[CodecExecutionRow, ...]:
    return tuple(
        CodecExecutionRow(
            frames=tuple(frames[row, frame] for frame in range(frames.shape[1])),
            state=None if states is None else states[row],
        )
        for row in range(frames.shape[0])
    )


def _sampling_row(sampling, row: int) -> TalkerSamplingExecutionRow:
    return TalkerSamplingExecutionRow(
        temperature=float(sampling.temperature[row]),
        top_k=int(sampling.top_k[row]),
        top_p=float(sampling.top_p[row]),
        repetition_penalty=float(sampling.repetition_penalty[row]),
        seed=int(sampling.seed[row]),
        offset=int(sampling.offsets[row]),
        seen_token_mask=(
            None if sampling.seen_token_mask is None else sampling.seen_token_mask[row]
        ),
    )


def talker_decode_rows(values) -> tuple[TalkerDecodeExecutionRow, ...]:
    return tuple(
        TalkerDecodeExecutionRow(
            talker_step_embed=values.talker_step_embed[row],
            text_token_id=values.text_token_ids[row],
            suppress_eos=bool(values.suppress_eos[row]),
            sampling=_sampling_row(values.sampling, row),
        )
        for row in range(values.talker_step_embed.shape[0])
    )


def talker_prefill_rows(
    values,
    sequence_lengths: tuple[int, ...],
) -> tuple[TalkerPrefillExecutionRow, ...]:
    rows: list[TalkerPrefillExecutionRow] = []
    start = 0
    for row, length in enumerate(sequence_lengths):
        stop = start + length
        rows.append(
            TalkerPrefillExecutionRow(
                text_token_ids=values.text_token_ids[start:stop],
                codec_token_ids=values.codec_token_ids[start:stop],
                codec_token_mask=values.codec_token_mask[start:stop],
                suppress_eos=bool(values.suppress_eos[row]),
                sampling=_sampling_row(values.sampling, row),
            )
        )
        start = stop
    return tuple(rows)




# Static-buffer introspection. These reach into executor internals on purpose:
# inspecting staged buffers is a test concern, and exposing accessors for it
# would put a production API in the package that production never calls.


def cp_static(executor, key):
    return executor._slots[key].values


def codec_static(executor, key):
    return executor._slots[key].frames


def talker_decode_static(executor, key, slot: int = 0):
    staged = executor._decode[key][slot]
    return SimpleNamespace(
        talker_step_embed=staged.step_embed,
        text_token_ids=staged.text,
        suppress_eos=staged.suppress,
        sampling=staged.sampling,
    )


def talker_prefill_static(executor, key, slot: int = 0):
    staged = executor._prefill[key][slot]
    return SimpleNamespace(
        text_token_ids=staged.text,
        codec_token_ids=staged.codec,
        codec_token_mask=staged.mask,
        last_token_indices=staged.last,
        suppress_eos=staged.suppress,
        sampling=staged.sampling,
    )



__all__ = [
    "codec_rows",
    "codec_static",
    "cp_rows",
    "cp_static",
    "talker_decode_rows",
    "talker_decode_static",
    "talker_prefill_rows",
    "talker_prefill_static",
]
