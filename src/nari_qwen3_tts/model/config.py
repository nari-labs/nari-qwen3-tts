"""Checkpoint model configuration, independent of execution policy."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MODEL_TYPE = "qwen3_tts"
_TTS_MODEL_TYPE = "custom_voice"
_TTS_MODEL_SIZE = "1b7"
_TOKENIZER_TYPE = "qwen3_tts_tokenizer_12hz"
_NUM_CODE_GROUPS = 16
_SUPPORTED_SPEAKERS = frozenset(
    {"aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"}
)
_SUPPORTED_LANGUAGES = frozenset(
    {
        "beijing_dialect",
        "chinese",
        "english",
        "french",
        "german",
        "italian",
        "japanese",
        "korean",
        "portuguese",
        "russian",
        "sichuan_dialect",
        "spanish",
    }
)


def _required_int(values: dict[str, Any], name: str, path: str) -> int:
    value = values[name]
    if type(value) is not int:
        raise TypeError(f"{path} must be an integer")
    return value


def _required_float(values: dict[str, Any], name: str, path: str) -> float:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _required_int_mapping(values: dict[str, Any], name: str, path: str) -> dict[str, int]:
    mapping = values[name]
    if not isinstance(mapping, dict):
        raise TypeError(f"{path} must be a mapping")
    result: dict[str, int] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or type(value) is not int:
            raise TypeError(f"{path} entries must map strings to integers")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class Qwen3TTSCodePredictorConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    num_code_groups: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float


@dataclass(frozen=True, slots=True)
class Qwen3TTSTalkerConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    text_vocab_size: int
    text_hidden_size: int
    num_code_groups: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    codec_bos_id: int
    codec_eos_token_id: int
    codec_think_id: int
    codec_nothink_id: int
    codec_pad_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int
    codec_language_id: dict[str, int]
    spk_id: dict[str, int]
    spk_is_dialect: dict[str, str | bool]
    code_predictor: Qwen3TTSCodePredictorConfig


@dataclass(frozen=True, slots=True)
class Qwen3TTSConfig:
    talker: Qwen3TTSTalkerConfig
    tts_bos_token_id: int
    tts_eos_token_id: int
    tts_pad_token_id: int
    tokenizer_type: str

    @property
    def talker_config(self) -> Qwen3TTSTalkerConfig:
        return self.talker

    @property
    def code_predictor(self) -> Qwen3TTSCodePredictorConfig:
        return self.talker.code_predictor

    @property
    def num_code_groups(self) -> int:
        return self.talker.num_code_groups

    @classmethod
    def from_pretrained_dir(cls, model_dir: str | Path) -> Qwen3TTSConfig:
        config_path = Path(model_dir) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Qwen3-TTS config not found: {config_path}")
        raw = json.loads(config_path.read_text())
        if not isinstance(raw, dict):
            raise TypeError("Qwen3-TTS config root must be a mapping")
        _validate_model_identity(raw)
        talker_raw = raw["talker_config"]
        cp_raw = talker_raw["code_predictor_config"]
        cp = Qwen3TTSCodePredictorConfig(
            hidden_size=_required_int(cp_raw, "hidden_size", "Code Predictor hidden_size"),
            intermediate_size=_required_int(
                cp_raw, "intermediate_size", "Code Predictor intermediate_size"
            ),
            num_hidden_layers=_required_int(
                cp_raw, "num_hidden_layers", "Code Predictor num_hidden_layers"
            ),
            num_attention_heads=_required_int(
                cp_raw, "num_attention_heads", "Code Predictor num_attention_heads"
            ),
            num_key_value_heads=_required_int(
                cp_raw, "num_key_value_heads", "Code Predictor num_key_value_heads"
            ),
            head_dim=_required_int(cp_raw, "head_dim", "Code Predictor head_dim"),
            vocab_size=_required_int(cp_raw, "vocab_size", "Code Predictor vocab_size"),
            num_code_groups=_required_int(
                cp_raw, "num_code_groups", "Code Predictor num_code_groups"
            ),
            max_position_embeddings=_required_int(
                cp_raw,
                "max_position_embeddings",
                "Code Predictor max_position_embeddings",
            ),
            rms_norm_eps=_required_float(cp_raw, "rms_norm_eps", "Code Predictor rms_norm_eps"),
            rope_theta=_required_float(cp_raw, "rope_theta", "Code Predictor rope_theta"),
        )
        talker = Qwen3TTSTalkerConfig(
            hidden_size=_required_int(talker_raw, "hidden_size", "Talker hidden_size"),
            intermediate_size=_required_int(talker_raw, "intermediate_size", "Talker intermediate_size"),
            num_hidden_layers=_required_int(talker_raw, "num_hidden_layers", "Talker num_hidden_layers"),
            num_attention_heads=_required_int(
                talker_raw, "num_attention_heads", "Talker num_attention_heads"
            ),
            num_key_value_heads=_required_int(
                talker_raw, "num_key_value_heads", "Talker num_key_value_heads"
            ),
            head_dim=_required_int(talker_raw, "head_dim", "Talker head_dim"),
            vocab_size=_required_int(talker_raw, "vocab_size", "Talker vocab_size"),
            text_vocab_size=_required_int(talker_raw, "text_vocab_size", "Talker text_vocab_size"),
            text_hidden_size=_required_int(talker_raw, "text_hidden_size", "Talker text_hidden_size"),
            num_code_groups=_required_int(talker_raw, "num_code_groups", "Talker num_code_groups"),
            max_position_embeddings=_required_int(
                talker_raw, "max_position_embeddings", "Talker max_position_embeddings"
            ),
            rms_norm_eps=_required_float(talker_raw, "rms_norm_eps", "Talker rms_norm_eps"),
            rope_theta=_required_float(talker_raw, "rope_theta", "Talker rope_theta"),
            codec_bos_id=_required_int(talker_raw, "codec_bos_id", "Talker codec_bos_id"),
            codec_eos_token_id=_required_int(
                talker_raw, "codec_eos_token_id", "Talker codec_eos_token_id"
            ),
            codec_think_id=_required_int(talker_raw, "codec_think_id", "Talker codec_think_id"),
            codec_nothink_id=_required_int(
                talker_raw, "codec_nothink_id", "Talker codec_nothink_id"
            ),
            codec_pad_id=_required_int(talker_raw, "codec_pad_id", "Talker codec_pad_id"),
            codec_think_bos_id=_required_int(
                talker_raw, "codec_think_bos_id", "Talker codec_think_bos_id"
            ),
            codec_think_eos_id=_required_int(
                talker_raw, "codec_think_eos_id", "Talker codec_think_eos_id"
            ),
            codec_language_id=_required_int_mapping(
                talker_raw, "codec_language_id", "Talker language ID map"
            ),
            spk_id=_required_int_mapping(talker_raw, "spk_id", "Talker speaker ID map"),
            spk_is_dialect=dict(talker_raw["spk_is_dialect"]),
            code_predictor=cp,
        )
        config = cls(
            talker=talker,
            tts_bos_token_id=_required_int(raw, "tts_bos_token_id", "tts_bos_token_id"),
            tts_eos_token_id=_required_int(raw, "tts_eos_token_id", "tts_eos_token_id"),
            tts_pad_token_id=_required_int(raw, "tts_pad_token_id", "tts_pad_token_id"),
            tokenizer_type=str(raw["tokenizer_type"]),
        )
        _validate_fixed_contract(config)
        return config


def _validate_model_identity(raw: dict[str, Any]) -> None:
    expected = {
        "model_type": _MODEL_TYPE,
        "tts_model_type": _TTS_MODEL_TYPE,
        "tts_model_size": _TTS_MODEL_SIZE,
        "tokenizer_type": _TOKENIZER_TYPE,
    }
    for name, value in expected.items():
        if raw.get(name) != value:
            raise ValueError(f"unsupported Qwen3-TTS {name}: {raw.get(name)!r}; expected {value!r}")


def _validate_attention_contract(
    config: Qwen3TTSTalkerConfig | Qwen3TTSCodePredictorConfig,
    *,
    label: str,
) -> None:
    integer_fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
    )
    for name in integer_fields:
        if getattr(config, name) <= 0:
            raise ValueError(f"{label} {name} must be positive")
    if config.head_dim % 2:
        raise ValueError(f"{label} head_dim must be even for rotary embedding")
    if config.num_attention_heads % config.num_key_value_heads:
        raise ValueError(f"{label} attention heads must be divisible by KV heads")
    if config.rms_norm_eps <= 0:
        raise ValueError(f"{label} rms_norm_eps must be positive")
    if config.rope_theta <= 0:
        raise ValueError(f"{label} rope_theta must be positive")


def _validate_model_domain(talker: Qwen3TTSTalkerConfig) -> None:
    if set(talker.spk_id) != _SUPPORTED_SPEAKERS:
        raise ValueError("Qwen3-TTS speaker map does not match the supported fixed model domain")
    if set(talker.spk_is_dialect) != _SUPPORTED_SPEAKERS:
        raise ValueError("Qwen3-TTS speaker dialect map does not match the supported speaker domain")
    if set(talker.codec_language_id) != _SUPPORTED_LANGUAGES:
        raise ValueError("Qwen3-TTS language map does not match the supported fixed model domain")
    for speaker, dialect in talker.spk_is_dialect.items():
        if dialect is not False and (not isinstance(dialect, str) or dialect not in talker.codec_language_id):
            raise ValueError(f"speaker {speaker!r} references unknown dialect {dialect!r}")


def _validate_token_ids(config: Qwen3TTSConfig) -> None:
    talker = config.talker
    tts_control_ids = (config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id)
    if len(set(tts_control_ids)) != len(tts_control_ids):
        raise ValueError("TTS control token IDs must be distinct")
    if any(token_id < 0 or token_id >= talker.text_vocab_size for token_id in tts_control_ids):
        raise ValueError("TTS control token ID is outside the Talker text vocabulary")
    codec_control_ids = (
        talker.codec_bos_id,
        talker.codec_eos_token_id,
        talker.codec_think_id,
        talker.codec_nothink_id,
        talker.codec_pad_id,
        talker.codec_think_bos_id,
        talker.codec_think_eos_id,
    )
    if len(set(codec_control_ids)) != len(codec_control_ids):
        raise ValueError("Codec control token IDs must be distinct")
    if any(token_id < 0 or token_id >= talker.vocab_size for token_id in codec_control_ids):
        raise ValueError("Codec control token ID is outside the Talker vocabulary")
    language_ids = tuple(talker.codec_language_id.values())
    if len(set(language_ids)) != len(language_ids):
        raise ValueError("Codec language IDs must be distinct")
    if any(token_id < 0 or token_id >= talker.vocab_size for token_id in language_ids):
        raise ValueError("Codec language ID is outside the Talker vocabulary")
    speaker_ids = tuple(talker.spk_id.values())
    if len(set(speaker_ids)) != len(speaker_ids):
        raise ValueError("speaker IDs must be distinct")
    if any(token_id < 0 or token_id >= talker.vocab_size for token_id in speaker_ids):
        raise ValueError("speaker ID is outside the Talker vocabulary")


def _validate_fixed_contract(config: Qwen3TTSConfig) -> None:
    talker = config.talker
    predictor = config.code_predictor
    if talker.num_code_groups != _NUM_CODE_GROUPS:
        raise ValueError(f"Talker must use exactly {_NUM_CODE_GROUPS} code groups")
    if predictor.num_code_groups != _NUM_CODE_GROUPS:
        raise ValueError(f"Code Predictor must use exactly {_NUM_CODE_GROUPS} code groups")
    _validate_attention_contract(talker, label="Talker")
    _validate_attention_contract(predictor, label="Code Predictor")
    _validate_model_domain(talker)
    _validate_token_ids(config)


__all__ = ["Qwen3TTSCodePredictorConfig", "Qwen3TTSConfig", "Qwen3TTSTalkerConfig"]
