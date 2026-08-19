from __future__ import annotations

import copy
import json

import pytest

from nari_qwen3_tts.model.config import Qwen3TTSConfig


def _raw_config() -> dict[str, object]:
    return {
        "model_type": "qwen3_tts",
        "tts_model_type": "custom_voice",
        "tts_model_size": "1b7",
        "tts_bos_token_id": 1,
        "tts_eos_token_id": 2,
        "tts_pad_token_id": 3,
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "talker_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": 64,
            "text_vocab_size": 80,
            "text_hidden_size": 8,
            "num_code_groups": 16,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "codec_bos_id": 10,
            "codec_eos_token_id": 11,
            "codec_think_id": 12,
            "codec_nothink_id": 13,
            "codec_pad_id": 14,
            "codec_think_bos_id": 15,
            "codec_think_eos_id": 16,
            "codec_language_id": {
                "chinese": 20,
                "english": 21,
                "french": 22,
                "german": 23,
                "italian": 24,
                "japanese": 25,
                "korean": 26,
                "portuguese": 27,
                "russian": 28,
                "spanish": 29,
                "beijing_dialect": 30,
                "sichuan_dialect": 31,
            },
            "spk_id": {
                "aiden": 40,
                "dylan": 41,
                "eric": 42,
                "ono_anna": 43,
                "ryan": 44,
                "serena": 45,
                "sohee": 46,
                "uncle_fu": 47,
                "vivian": 48,
            },
            "spk_is_dialect": {
                "aiden": False,
                "dylan": "beijing_dialect",
                "eric": "sichuan_dialect",
                "ono_anna": False,
                "ryan": False,
                "serena": False,
                "sohee": False,
                "uncle_fu": False,
                "vivian": False,
            },
            "code_predictor_config": {
                # The official Code Predictor has a wider Q projection than
                # its hidden size (1024 hidden, 16 heads x 128 head dim).
                "hidden_size": 4,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 64,
                "num_code_groups": 16,
                "max_position_embeddings": 32,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000.0,
            },
        },
    }


def test_model_config_is_loaded_from_snapshot(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_raw_config()))
    config = Qwen3TTSConfig.from_pretrained_dir(tmp_path)
    assert config.talker.hidden_size == 16
    assert config.code_predictor.num_hidden_layers == 2
    assert config.num_code_groups == 16
    assert config.talker_config is config.talker
    assert config.tokenizer_type == "qwen3_tts_tokenizer_12hz"
    assert config.talker.spk_is_dialect["dylan"] == "beijing_dialect"


def _replace(raw: dict[str, object], path: tuple[str, ...], value: object) -> dict[str, object]:
    result = copy.deepcopy(raw)
    target = result
    for name in path[:-1]:
        target = target[name]  # type: ignore[assignment,index]
    target[path[-1]] = value
    return result


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model_type",), "qwen2", "model_type"),
        (("tts_model_type",), "base", "tts_model_type"),
        (("tts_model_size",), "0b6", "tts_model_size"),
        (("tokenizer_type",), "qwen2", "tokenizer_type"),
        (("talker_config", "num_code_groups"), 15, "code groups"),
        (("talker_config", "code_predictor_config", "num_code_groups"), 15, "code groups"),
        (("talker_config", "num_attention_heads"), 3, "attention heads"),
        (("talker_config", "num_key_value_heads"), 3, "KV heads"),
        (("talker_config", "head_dim"), 3, "head_dim"),
        (("talker_config", "rms_norm_eps"), 0, "rms_norm_eps"),
        (("talker_config", "rope_theta"), -1, "rope_theta"),
        (("talker_config", "spk_id"), {"aiden": 40}, "speaker"),
        (("talker_config", "codec_language_id"), {"english": 21}, "language"),
        (("talker_config", "spk_is_dialect", "dylan"), "unknown_dialect", "dialect"),
    ],
)
def test_snapshot_config_rejects_incompatible_fixed_model_family(
    tmp_path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_replace(_raw_config(), path, value)))
    with pytest.raises((TypeError, ValueError), match=message):
        Qwen3TTSConfig.from_pretrained_dir(tmp_path)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("tts_bos_token_id",), True, "tts_bos_token_id"),
        (("talker_config", "hidden_size"), "16", "hidden_size"),
        (("talker_config", "rms_norm_eps"), True, "rms_norm_eps"),
        (("talker_config", "codec_language_id", "english"), True, "language"),
        (("talker_config", "spk_id", "aiden"), True, "speaker"),
        (
            ("talker_config", "code_predictor_config", "num_hidden_layers"),
            2.0,
            "num_hidden_layers",
        ),
    ],
)
def test_snapshot_config_does_not_coerce_incompatible_json_scalar_types(
    tmp_path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_replace(_raw_config(), path, value)))
    with pytest.raises((TypeError, ValueError), match=message):
        Qwen3TTSConfig.from_pretrained_dir(tmp_path)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("tts_eos_token_id",), 1, "TTS control"),
        (("tts_pad_token_id",), 80, "text vocabulary"),
        (("talker_config", "codec_eos_token_id"), 10, "Codec control"),
        (("talker_config", "codec_pad_id"), 64, "Talker vocabulary"),
        (("talker_config", "codec_language_id", "english"), 64, "language"),
        (("talker_config", "spk_id", "aiden"), 64, "speaker"),
        (("talker_config", "codec_language_id", "french"), 21, "language IDs"),
        (("talker_config", "spk_id", "dylan"), 40, "speaker IDs"),
    ],
)
def test_snapshot_config_rejects_colliding_or_out_of_vocabulary_model_ids(
    tmp_path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_replace(_raw_config(), path, value)))
    with pytest.raises(ValueError, match=message):
        Qwen3TTSConfig.from_pretrained_dir(tmp_path)
