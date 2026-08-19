"""Input construction helpers for the two Qwen3-TTS Talker modes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedTalkerTokenLayout:
    """Integer Talker prefill layout plus request-local continuation data."""

    text_token_ids: torch.Tensor
    codec_token_ids: torch.Tensor
    codec_token_mask: torch.Tensor
    seq_lens: list[int]
    trailing_text_ids: list[torch.Tensor]
    tts_pad_id: torch.Tensor


def _custom_voice_codec_prefix(
    config,
    *,
    language: str,
    speaker: str,
) -> tuple[int, list[int]]:
    """Resolve request metadata to the official CustomVoice codec prefix."""
    talker_config = config.talker_config
    speaker_key = speaker.lower()
    if speaker_key not in talker_config.spk_id:
        raise ValueError(f"Unsupported Qwen3-TTS speaker: {speaker!r}")
    speaker_id = talker_config.spk_id[speaker_key]

    language_key = language.lower()
    if language_key == "auto":
        language_id = None
    elif language_key in talker_config.codec_language_id:
        language_id = talker_config.codec_language_id[language_key]
    else:
        raise ValueError(f"Unsupported Qwen3-TTS language: {language!r}")

    if language_key in {"chinese", "auto"} and talker_config.spk_is_dialect[speaker_key]:
        dialect = talker_config.spk_is_dialect[speaker_key]
        language_id = talker_config.codec_language_id[dialect]

    if language_id is None:
        codec_prefix = [
            talker_config.codec_nothink_id,
            talker_config.codec_think_bos_id,
            talker_config.codec_think_eos_id,
        ]
    else:
        codec_prefix = [
            talker_config.codec_think_id,
            talker_config.codec_think_bos_id,
            language_id,
            talker_config.codec_think_eos_id,
        ]
    return speaker_id, codec_prefix


def custom_voice_talker_input_seq_len(
    config,
    *,
    text_token_count: int,
    instruct_token_count: int,
    language: str,
    speaker: str,
    non_streaming_mode: bool,
) -> int:
    """Return the prefill length without launching tensor operations."""
    _, codec_prefix = _custom_voice_codec_prefix(
        config,
        language=language,
        speaker=speaker,
    )
    role_token_count = min(text_token_count, 3)
    prefix_token_count = len(codec_prefix) + 2
    if non_streaming_mode:
        target_token_count = max(0, text_token_count - 8) + 1
        continuation_token_count = target_token_count + 1
    else:
        continuation_token_count = int(text_token_count > 3)
    return instruct_token_count + role_token_count + prefix_token_count + continuation_token_count


def build_batched_custom_voice_talker_token_layout(  # noqa: PLR0915 - one explicit fixed layout
    config,
    text_inputs: list[torch.Tensor],
    instruct_inputs: list[torch.Tensor],
    *,
    languages: list[str],
    speakers: list[str],
    non_streaming_modes: list[bool],
) -> PackedTalkerTokenLayout:
    """Build the integer CustomVoice layout consumed by captured prefill.

    Keeping embedding lookup out of this function lets the Talker CUDA Graph
    capture text/codec lookup and addition together with transformer prefill.
    Trailing text is returned separately so callers can materialize it in one
    batched lookup without copying the full prefill embedding tensor.
    """
    batch_size = len(text_inputs)
    if batch_size == 0:
        raise ValueError("Qwen3-TTS Talker prefill batch cannot be empty")
    if not (len(instruct_inputs) == len(languages) == len(speakers) == len(non_streaming_modes) == batch_size):
        raise ValueError("Qwen3-TTS Talker prefill batch fields must have equal lengths")
    talker_config = config.talker_config
    text_dtype = text_inputs[0].dtype
    device = text_inputs[0].device
    special_text_ids = torch.tensor(
        [
            config.tts_bos_token_id,
            config.tts_eos_token_id,
            config.tts_pad_token_id,
        ],
        dtype=text_dtype,
        device=device,
    )
    tts_bos_id = special_text_ids[0:1]
    tts_eos_id = special_text_ids[1:2]
    tts_pad_id = special_text_ids[2:3]

    prefill_text_pieces: list[torch.Tensor] = []
    prefill_codec_ids: list[int] = []
    prefill_codec_mask: list[bool] = []
    trailing_text_ids: list[torch.Tensor] = []
    seq_lens: list[int] = []
    packed_length = 0

    for raw_text, raw_instruct, language, speaker, non_streaming_mode in zip(
        text_inputs,
        instruct_inputs,
        languages,
        speakers,
        non_streaming_modes,
        strict=True,
    ):
        text = raw_text.reshape(-1)
        instruct = raw_instruct.reshape(-1)
        speaker_id, codec_prefix = _custom_voice_codec_prefix(
            config,
            language=language,
            speaker=speaker,
        )

        if instruct.numel():
            prefill_text_pieces.append(instruct)
            prefill_codec_ids.extend([talker_config.codec_pad_id] * instruct.numel())
            prefill_codec_mask.extend([False] * instruct.numel())
        role_text = text[:3]
        if role_text.numel():
            prefill_text_pieces.append(role_text)
            prefill_codec_ids.extend([talker_config.codec_pad_id] * role_text.numel())
            prefill_codec_mask.extend([False] * role_text.numel())

        # Prefix text is PAD for every codec conditioning token except the
        # final slot, where the official layout uses text BOS.
        prefill_text_pieces.append(tts_pad_id.expand(len(codec_prefix) + 1))
        prefill_text_pieces.append(tts_bos_id)
        request_codec_ids = [
            *codec_prefix,
            speaker_id,
            talker_config.codec_pad_id,
        ]
        prefill_codec_ids.extend(request_codec_ids)
        prefill_codec_mask.extend([True] * len(request_codec_ids))

        if non_streaming_mode:
            target_text = text[3:-5]
            if target_text.numel():
                prefill_text_pieces.append(target_text)
            prefill_text_pieces.extend([tts_eos_id, tts_pad_id])
            continuation_codec_ids = [talker_config.codec_pad_id] * (target_text.numel() + 1)
            continuation_codec_ids.append(talker_config.codec_bos_id)
            prefill_codec_ids.extend(continuation_codec_ids)
            prefill_codec_mask.extend([True] * len(continuation_codec_ids))
            trailing_text_ids.append(tts_pad_id)
        else:
            first_text = text[3:4]
            if first_text.numel():
                prefill_text_pieces.append(first_text)
                prefill_codec_ids.append(talker_config.codec_bos_id)
                prefill_codec_mask.append(True)

            trailing_text = text[4:-5]
            trailing_text_ids.append(torch.cat([trailing_text, tts_eos_id], dim=0))

        seq_len = custom_voice_talker_input_seq_len(
            config,
            text_token_count=text.numel(),
            instruct_token_count=instruct.numel(),
            language=language,
            speaker=speaker,
            non_streaming_mode=non_streaming_mode,
        )
        if len(prefill_codec_ids) != packed_length + seq_len or len(prefill_codec_mask) != packed_length + seq_len:
            raise RuntimeError("Qwen3-TTS Talker prefill layout length mismatch")
        seq_lens.append(seq_len)
        packed_length += seq_len

    packed_text_ids = torch.cat(prefill_text_pieces, dim=0)
    packed_codec_ids = torch.tensor(
        prefill_codec_ids,
        dtype=text_dtype,
        device=device,
    )
    packed_codec_mask = torch.tensor(
        prefill_codec_mask,
        dtype=torch.bool,
        device=device,
    )
    if not (packed_text_ids.shape == packed_codec_ids.shape == packed_codec_mask.shape):
        raise RuntimeError("Qwen3-TTS packed prefill fields have mismatched lengths")
    return PackedTalkerTokenLayout(
        text_token_ids=packed_text_ids,
        codec_token_ids=packed_codec_ids,
        codec_token_mask=packed_codec_mask,
        seq_lens=seq_lens,
        trailing_text_ids=trailing_text_ids,
        tts_pad_id=tts_pad_id,
    )
