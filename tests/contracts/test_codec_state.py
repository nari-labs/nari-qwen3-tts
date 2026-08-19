from __future__ import annotations

import copy
import pickle

import torch


def test_incremental_codec_state_preserves_mutable_request_local_storage() -> None:
    from nari_qwen3_tts.contract.codec_state import IncrementalCodecState

    state = IncrementalCodecState()
    key = torch.tensor([[1.0]])
    state.frame_position = 3
    state.transformer_context_length = 2
    state.transformer_keys[0] = key
    state.conv_histories["layer"] = key.clone()

    cloned = copy.deepcopy(state)
    restored = pickle.loads(pickle.dumps(state))
    for value in (cloned, restored):
        assert value.frame_position == 3
        assert value.transformer_context_length == 2
        assert torch.equal(value.transformer_keys[0], key)
        assert torch.equal(value.conv_histories["layer"], key)
        assert value.transformer_keys is not state.transformer_keys
