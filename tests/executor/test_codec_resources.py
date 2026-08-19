from __future__ import annotations

import torch

from nari_qwen3_tts.executor.codec import CodecExecutor
from nari_qwen3_tts.model.incremental_codec import IncrementalCodecState


class _WholeDecoder(torch.nn.Module):
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return codes.float().sum(dim=1, keepdim=True).repeat_interleave(3, dim=2)


class _IncrementalDecoder:
    samples_per_frame = 3
    retained_context = 8

    def __call__(
        self,
        codes: torch.Tensor,
        states: list[IncrementalCodecState],
        **_kwargs,
    ) -> torch.Tensor:
        frames = codes.shape[2]
        for state in states:
            state.frame_position += frames
            state.transformer_context_length = min(
                self.retained_context,
                state.transformer_context_length + frames,
            )
            state.transformer_keys[0] = torch.ones((1, 1, 1))
            state.transformer_values[0] = torch.ones((1, 1, 1))
            state.conv_histories["conv"] = torch.ones((1, 1))
            state.transconv_overlaps["transconv"] = torch.ones((1, 1))
        return codes.float().sum(dim=1, keepdim=True).repeat_interleave(3, dim=2)


def test_codec_executor_owns_model_and_incremental_state() -> None:
    model = type("Codec", (), {"decoder": _WholeDecoder()})()
    incremental = _IncrementalDecoder()
    executor = CodecExecutor(
        model=model,
        incremental_decoder=incremental,
        num_code_groups=2,
        cold_frame_sizes=(7,),
        device=torch.device("cpu"),
        driver=object(),
    )

    assert executor.model is model
    assert executor.incremental_decoder is incremental
    assert executor.samples_per_frame == 3
    assert executor.retained_context == 8
    assert callable(executor.whole_sequence_decode)
    assert callable(executor.state_bootstrap)
    assert callable(executor.warm_incremental)
    assert callable(executor.terminal)
def test_pcm_transfer_pool_owns_cuda_stream_event_and_pinned_storage() -> None:
    from nari_qwen3_tts.executor.pcm import PcmTransfer, PcmTransferPool

    assert callable(PcmTransferPool.begin)
    assert callable(PcmTransferPool.prepare)
    assert callable(PcmTransfer.poll)
    assert callable(PcmTransfer.discard_when_ready)
