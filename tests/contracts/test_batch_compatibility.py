from __future__ import annotations


def test_decode_compatibility_singleton_has_the_canonical_type() -> None:
    from nari_qwen3_tts.contract import (
        TALKER_DECODE_COMPATIBILITY,
        TalkerDecodeBatchCompatibility,
    )
    assert isinstance(TALKER_DECODE_COMPATIBILITY, TalkerDecodeBatchCompatibility)
    assert TalkerDecodeBatchCompatibility() is not TALKER_DECODE_COMPATIBILITY
    assert TalkerDecodeBatchCompatibility() == TALKER_DECODE_COMPATIBILITY


def test_codec_batch_compatibility_preserves_exact_and_padded_terminal_matrix() -> None:
    from nari_qwen3_tts.contract import (
        CodecBatchCompatibility,
        CodecExecutionMode,
        codec_batch_compatible,
    )

    def value(
        mode: CodecExecutionMode,
        *,
        model: int,
        input_frames: int,
        visible: int,
        start: int = 0,
        terminal: bool = False,
    ) -> CodecBatchCompatibility:
        return CodecBatchCompatibility(
            mode=mode,
            model_frames=model,
            input_frames=input_frames,
            visible_frames=visible,
            pcm_start_frame=start,
            producer_frames=input_frames,
            terminal=terminal,
        )

    whole_a = value(CodecExecutionMode.WHOLE_SEQUENCE, model=2, input_frames=2, visible=1, start=1)
    whole_b = value(CodecExecutionMode.WHOLE_SEQUENCE, model=2, input_frames=2, visible=2)
    assert codec_batch_compatible(whole_a, whole_b)

    cold_a = value(CodecExecutionMode.COLD, model=6, input_frames=6, visible=4, start=2)
    cold_b = value(CodecExecutionMode.COLD, model=6, input_frames=6, visible=6)
    assert codec_batch_compatible(cold_a, cold_b)

    warm_short = value(
        CodecExecutionMode.WARM,
        model=12,
        input_frames=5,
        visible=5,
        terminal=True,
    )
    warm_long = value(
        CodecExecutionMode.WARM,
        model=12,
        input_frames=9,
        visible=9,
        terminal=True,
    )
    assert codec_batch_compatible(warm_short, warm_long)
    assert not codec_batch_compatible(warm_short, cold_a)
    assert not codec_batch_compatible(
        warm_short,
        value(CodecExecutionMode.WARM, model=12, input_frames=12, visible=12),
    )
