from __future__ import annotations

import pytest
import torch

from nari_qwen3_tts.executor.codec import CodecExecutor as CodecCudaExecutor
from nari_qwen3_tts.model.incremental_codec import IncrementalCodecState


class _WholeDecoder(torch.nn.Module):
    total_upsample = 3

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        values = codes.float().sum(dim=1, keepdim=True)
        return values.repeat_interleave(self.total_upsample, dim=2)


class _Incremental:
    samples_per_frame = 3
    retained_context = 8

    def __call__(self, codes: torch.Tensor, states: list[IncrementalCodecState]) -> torch.Tensor:
        frames = codes.shape[2]
        for state in states:
            state.frame_position += frames
            state.transformer_context_length += frames
            state.transformer_keys[0] = torch.ones((1, 1, 1))
            state.transformer_values[0] = torch.ones((1, 1, 1))
            state.conv_histories["conv"] = torch.ones((1, 1))
            state.transconv_overlaps["transconv"] = torch.ones((1, 1))
        return codes.float().sum(dim=1, keepdim=True).repeat_interleave(3, dim=2)


def _codec() -> CodecCudaExecutor:
    model = type("Codec", (), {"decoder": _WholeDecoder()})()
    return CodecCudaExecutor(
        model=model,
        incremental_decoder=_Incremental(),
        num_code_groups=2,
        cold_frame_sizes=(7,),
        device=torch.device("cpu"),
        driver=object(),
    )


def test_whole_sequence_and_incremental_lifecycles_are_explicit() -> None:
    job = _codec()
    whole_sequence = job.whole_sequence_decode(torch.tensor([[[1, 2], [3, 4]]]))
    assert whole_sequence.pcm.dtype == torch.int16
    assert whole_sequence.pcm.shape == (1, 6)
    assert whole_sequence.states is None

    original = [job.new_state(), job.new_state()]
    cold = job.state_bootstrap(torch.ones((2, 2, 2), dtype=torch.long), original)
    assert [state.frame_position for state in original] == [0, 0]
    assert [state.frame_position for state in cold.states or ()] == [2, 2]
    warm = job.warm_incremental(torch.ones((2, 1, 2), dtype=torch.long), cold.states or ())
    terminal = job.terminal(torch.empty((2, 0, 2), dtype=torch.long), warm.states or ())
    assert terminal.terminal
    assert terminal.pcm.shape == (2, 0)
    assert [state.frame_position for state in terminal.states or ()] == [3, 3]


def test_cold_and_warm_guards_fail_closed() -> None:
    job = _codec()
    warm = job.new_state()
    warm.frame_position = 1
    try:
        job.state_bootstrap(torch.ones((1, 1, 2), dtype=torch.long), [warm])
    except ValueError as error:
        assert "cold" in str(error)
    else:
        raise AssertionError("warm state was accepted as cold")


@pytest.mark.parametrize(
    "mapping_name,key",
    (
        ("transformer_keys", 0),
        ("transformer_values", 0),
        ("conv_histories", "conv"),
        ("transconv_overlaps", "transconv"),
    ),
)
def test_state_bootstrap_rejects_cold_counters_with_stale_tensors(mapping_name: str, key: str | int) -> None:
    job = _codec()
    stale = job.new_state()
    getattr(stale, mapping_name)[key] = torch.ones(1)

    with pytest.raises(ValueError, match="cold Codec states"):
        job.state_bootstrap(torch.ones((1, 1, 2), dtype=torch.long), [stale])


def test_warm_incremental_rejects_incoherent_counters_and_mappings() -> None:
    job = _codec()
    frames = torch.ones((1, 1, 2), dtype=torch.long)
    coherent = (job.state_bootstrap(frames, [job.new_state()]).states or ())[0]

    zero_context = job.clone_state(coherent)
    zero_context.transformer_context_length = 0
    with pytest.raises(ValueError, match="coherent warm Codec states"):
        job.warm_incremental(frames, [zero_context])

    missing_mapping = job.clone_state(coherent)
    missing_mapping.conv_histories.clear()
    with pytest.raises(ValueError, match="coherent warm Codec states"):
        job.warm_incremental(frames, [missing_mapping])
    with pytest.raises(ValueError, match="coherent warm Codec states"):
        job.terminal(frames[:, :0], [missing_mapping])

    mismatched_kv = job.clone_state(coherent)
    mismatched_kv.transformer_values.clear()
    with pytest.raises(ValueError, match="coherent warm Codec states"):
        job.warm_incremental(frames, [mismatched_kv])

    stale_context = job.clone_state(coherent)
    stale_context.frame_position = 2
    stale_context.transformer_context_length = 1
    with pytest.raises(ValueError, match="coherent warm Codec states"):
        job.warm_incremental(frames, [stale_context])


def test_codec_jobs_reject_empty_batches_but_allow_request_local_empty_terminal() -> None:
    job = _codec()
    with pytest.raises(ValueError, match="at least one row"):
        job.whole_sequence_decode(torch.empty((0, 1, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="at least one row"):
        job.state_bootstrap(torch.empty((0, 1, 2), dtype=torch.long), [])
    with pytest.raises(ValueError, match="at least one row"):
        job.terminal(torch.empty((0, 0, 2), dtype=torch.long), [])

    warm = job.state_bootstrap(torch.ones((1, 1, 2), dtype=torch.long), [job.new_state()])
    terminal = job.terminal(torch.empty((1, 0, 2), dtype=torch.long), warm.states or ())
    assert terminal.pcm.shape == (1, 0)
    assert terminal.terminal is True


def test_codec_clones_every_request_state_tensor_before_decoder_mutation() -> None:
    job = _codec()
    first = job.state_bootstrap(torch.ones((1, 1, 2), dtype=torch.long), [job.new_state()])
    source = (first.states or ())[0]
    source_snapshot = job.clone_state(source)

    successor = (job.warm_incremental(torch.ones((1, 1, 2), dtype=torch.long), [source]).states or ())[0]

    assert source.frame_position == source_snapshot.frame_position
    assert source.transformer_context_length == source_snapshot.transformer_context_length
    for mapping_name in ("transformer_keys", "transformer_values", "conv_histories", "transconv_overlaps"):
        source_mapping = getattr(source, mapping_name)
        snapshot_mapping = getattr(source_snapshot, mapping_name)
        successor_mapping = getattr(successor, mapping_name)
        assert source_mapping.keys() == snapshot_mapping.keys() == successor_mapping.keys()
        for key in source_mapping:
            torch.testing.assert_close(source_mapping[key], snapshot_mapping[key])
            assert successor_mapping[key].data_ptr() != source_mapping[key].data_ptr()


def test_codec_partial_terminal_advances_owned_state_and_emits_exact_pcm() -> None:
    job = _codec()
    bootstrap = job.state_bootstrap(torch.ones((1, 2, 2), dtype=torch.long), [job.new_state()])
    source = (bootstrap.states or ())[0]

    terminal = job.terminal(torch.ones((1, 2, 2), dtype=torch.long), [source])

    assert terminal.terminal is True
    assert terminal.pcm.shape == (1, 6)
    assert (terminal.states or ())[0].frame_position == 4
    assert source.frame_position == 2


def test_whole_sequence_codec_fails_closed_when_decoder_returns_too_few_samples() -> None:
    class _ShortWholeDecoder(torch.nn.Module):
        total_upsample = 3

        def forward(self, codes: torch.Tensor) -> torch.Tensor:
            return codes.new_zeros((codes.shape[0], 1, 1), dtype=torch.float32)

    model = type("Codec", (), {"decoder": _ShortWholeDecoder()})()
    job = CodecCudaExecutor(
        model=model,
        incremental_decoder=_Incremental(),
        num_code_groups=2,
        cold_frame_sizes=(7,),
        device=torch.device("cpu"),
        driver=object(),
    )

    with pytest.raises(RuntimeError, match="produced 1 samples, expected 6"):
        job.whole_sequence_decode(torch.ones((1, 2, 2), dtype=torch.long))


def test_pcm16_conversion_clamps_and_rounds_the_fixed_audio_contract() -> None:
    wav = torch.tensor([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    assert CodecCudaExecutor._pcm16(wav).tolist() == [[-32767, -16384, 0, 16384, 32767]]


def test_codec_rejects_incompatible_warm_rows_before_decoder_execution() -> None:
    job = _codec()
    bootstrap = job.state_bootstrap(torch.ones((2, 1, 2), dtype=torch.long), [job.new_state(), job.new_state()])
    first, second = (job.clone_state(state) for state in (bootstrap.states or ()))
    second.conv_histories["conv"] = torch.ones((2, 1))

    with pytest.raises(ValueError, match="coherent warm Codec states"):
        job.warm_incremental(torch.ones((2, 1, 2), dtype=torch.long), [first, second])

    second = job.clone_state(first)
    second.transformer_values[0] = torch.ones((2, 1, 1))
    with pytest.raises(ValueError, match="coherent warm Codec states"):
        job.warm_incremental(torch.ones((2, 1, 2), dtype=torch.long), [first, second])


def test_codec_decoder_failure_cannot_mutate_input_state() -> None:
    job = _codec()
    source = (job.state_bootstrap(torch.ones((1, 1, 2), dtype=torch.long), [job.new_state()]).states or ())[0]
    snapshot = job.clone_state(source)

    class _FailingIncremental(_Incremental):
        def __call__(self, codes: torch.Tensor, states: list[IncrementalCodecState]) -> torch.Tensor:
            states[0].frame_position += 100
            states[0].conv_histories["corrupt"] = torch.ones(1)
            raise RuntimeError("decoder failure")

    job.incremental_decoder = _FailingIncremental()
    with pytest.raises(RuntimeError, match="decoder failure"):
        job.warm_incremental(torch.ones((1, 1, 2), dtype=torch.long), [source])

    assert source.frame_position == snapshot.frame_position
    assert source.conv_histories.keys() == snapshot.conv_histories.keys()
    for key in source.conv_histories:
        torch.testing.assert_close(source.conv_histories[key], snapshot.conv_histories[key])


@pytest.mark.parametrize("method", ["whole_sequence_decode", "state_bootstrap"])
def test_codec_rejects_wrong_frame_dtype_or_codebook_width(method: str) -> None:
    job = _codec()
    states = [job.new_state()]

    def invoke(frames: torch.Tensor) -> object:
        if method == "whole_sequence_decode":
            return job.whole_sequence_decode(frames)
        return job.state_bootstrap(frames, states)

    with pytest.raises(TypeError, match="long"):
        invoke(torch.ones((1, 1, 2)))
    with pytest.raises(ValueError, match="codebook"):
        invoke(torch.ones((1, 1, 3), dtype=torch.long))


def test_codec_rejects_state_device_mismatch_before_decoder_execution() -> None:
    job = _codec()
    warm = (job.state_bootstrap(torch.ones((1, 1, 2), dtype=torch.long), [job.new_state()]).states or ())[0]
    mismatched = job.clone_state(warm)
    mismatched.conv_histories["conv"] = torch.ones((1, 1), device="meta")

    with pytest.raises(ValueError, match="device"):
        job.warm_incremental(torch.ones((1, 1, 2), dtype=torch.long), [mismatched])


@pytest.mark.parametrize("sample_delta", [-1, 1])
def test_incremental_codec_rejects_short_or_long_pcm_without_mutating_input(sample_delta: int) -> None:
    job = _codec()
    source = job.new_state()

    class _WrongLengthIncremental(_Incremental):
        def __call__(self, codes: torch.Tensor, states: list[IncrementalCodecState]) -> torch.Tensor:
            exact = super().__call__(codes, states)
            if sample_delta < 0:
                return exact[:, :, :sample_delta]
            return torch.cat([exact, exact[:, :, :sample_delta]], dim=2)

    job.incremental_decoder = _WrongLengthIncremental()
    with pytest.raises(RuntimeError, match="samples, expected"):
        job.state_bootstrap(torch.ones((1, 2, 2), dtype=torch.long), [source])

    assert source.frame_position == 0
    assert source.transformer_context_length == 0
    assert all(
        not getattr(source, name)
        for name in ("transformer_keys", "transformer_values", "conv_histories", "transconv_overlaps")
    )
