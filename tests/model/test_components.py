from __future__ import annotations

import torch

from nari_qwen3_tts.model.components import Qwen3TTSCodePredictor, Qwen3TTSTalkerModel
from nari_qwen3_tts.model.config import (
    Qwen3TTSCodePredictorConfig,
    Qwen3TTSConfig,
    Qwen3TTSTalkerConfig,
)


def _config() -> Qwen3TTSConfig:
    cp = Qwen3TTSCodePredictorConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=64,
        num_code_groups=4,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )
    talker = Qwen3TTSTalkerConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        text_vocab_size=80,
        text_hidden_size=8,
        num_code_groups=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        codec_bos_id=10,
        codec_eos_token_id=11,
        codec_think_id=12,
        codec_nothink_id=13,
        codec_pad_id=14,
        codec_think_bos_id=15,
        codec_think_eos_id=16,
        codec_language_id={"english": 20},
        spk_id={"aiden": 30},
        spk_is_dialect={"aiden": False},
        code_predictor=cp,
    )
    return Qwen3TTSConfig(talker, 1, 2, 3, "qwen2")


def test_raw_components_have_exact_packed_checkpoint_layout() -> None:
    config = _config()
    talker = Qwen3TTSTalkerModel(config)
    predictor = Qwen3TTSCodePredictor(config)
    talker_params = dict(talker.named_parameters())
    cp_params = dict(predictor.named_parameters())

    assert talker_params["model.layers.0.self_attn.qkv_proj.weight"].shape == (32, 16)
    assert talker_params["model.layers.0.mlp.gate_up_proj.weight"].shape == (64, 16)
    assert talker_params["text_projection.linear_fc1.weight"].shape == (8, 8)
    assert cp_params["model.layers.0.self_attn.qkv_proj.weight"].shape == (16, 8)
    assert cp_params["small_to_mtp_projection.weight"].shape == (8, 16)
    assert cp_params["model.codec_embedding.0.weight"].shape == (64, 16)
    assert len(predictor.lm_head) == 3


def test_rmsnorm_and_packed_modules_are_real_torch_modules() -> None:
    model = Qwen3TTSTalkerModel(_config())
    assert isinstance(model, torch.nn.Module)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
