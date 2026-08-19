from __future__ import annotations

import json
from dataclasses import replace

import pytest

from nari_qwen3_tts.contract import (
    CodecCaptureKey,
    CodecExecutionMode,
    CodePredictorCaptureKey,
    TalkerDecodeCaptureKey,
    TalkerPrefillCaptureKey,
)
from nari_qwen3_tts.planner import CaptureCatalog, CaptureCoverageError
from nari_qwen3_tts.profile import (
    BatchCaptureConfig,
    CodecBatchCaptureConfig,
    CodecCaptureConfig,
    CodecFrameCaptureConfig,
    ExecutionProfile,
    PrefillCaptureConfig,
    ProfileLoader,
    RequiredSchedulingPolicy,
    SchedulingPolicyConfig,
)


@pytest.mark.parametrize(
    ("profile", "policy", "lead_s", "chunks", "max_batch"),
    [
        (
            ExecutionProfile.TTFA,
            RequiredSchedulingPolicy.DEADLINE_AWARE,
            1.0,
            (2, 4, 8, 12),
            32,
        ),
        (
            ExecutionProfile.BALANCED,
            RequiredSchedulingPolicy.ROUND_ROBIN,
            None,
            (4, 4, 8, 16, 25),
            64,
        ),
        (
            ExecutionProfile.THROUGHPUT,
            RequiredSchedulingPolicy.ROUND_ROBIN,
            None,
            (25,),
            64,
        ),
    ],
)
def test_required_profiles_define_dynamic_capacity_and_policy_metadata(
    profile: ExecutionProfile,
    policy: RequiredSchedulingPolicy,
    lead_s: float | None,
    chunks: tuple[int, ...],
    max_batch: int,
) -> None:
    config = ProfileLoader().load_profile(profile)
    catalog = CaptureCatalog.from_config(config.stages)

    assert config.policy.kind is policy
    assert config.policy.pressing_lead_s == lead_s
    assert config.codec_chunks(silent_bootstrap_suppressed=True) == chunks
    assert config.stages.talker_prefill.max_batch_size == 8
    assert config.stages.talker_decode.max_batch_size == max_batch
    assert config.stages.code_predictor.max_batch_size == max_batch
    assert config.stages.codec.max_batch_size == max_batch
    assert max(key.capture_batch_size for key in catalog.talker_decode) == max_batch
    assert max(key.capture_batch_size for key in catalog.code_predictor) == max_batch
    assert max(key.capture_batch_size for key in catalog.codec) == max_batch


def test_required_profiles_fix_every_stage_capture_surface_to_the_slo_configs() -> None:
    dense_b32 = (*range(1, 13), 16, 24, 32)
    dense_b64 = (*range(1, 13), 16, 24, 32, 40, 48, 56, 64)

    ttfa = ProfileLoader().load_profile(ExecutionProfile.TTFA)
    assert ttfa.stages.talker_decode.batch_sizes == dense_b32
    assert ttfa.stages.code_predictor.batch_sizes == dense_b32
    assert ttfa.stages.codec.batches == CodecBatchCaptureConfig(
        whole_sequence_first_frame=(1, 2, 4, 8, 16, 32),
        whole_sequence_followup=(1, 2, 4, 8),
        cold=(1, 2, 4, 8),
        warm_partial=tuple(range(1, 9)),
        warm_full=dense_b32,
    )
    assert ttfa.stages.codec.frames == CodecFrameCaptureConfig(
        cold=tuple(range(4, 8)),
        warm=tuple(range(1, 13)),
        warm_full=(12,),
        terminal_pad=(12,),
    )

    balanced = ProfileLoader().load_profile(ExecutionProfile.BALANCED)
    assert balanced.stages.talker_decode.batch_sizes == dense_b64
    assert balanced.stages.code_predictor.batch_sizes == dense_b64
    assert balanced.stages.codec.batches.warm_full == dense_b64
    assert balanced.stages.codec.frames == CodecFrameCaptureConfig(
        cold=tuple(range(4, 8)),
        warm=(*range(1, 13), 16, 25),
        warm_full=(4, 8, 12, 16, 25),
        terminal_pad=(12, 25),
    )

    throughput = ProfileLoader().load_profile(ExecutionProfile.THROUGHPUT)
    assert throughput.stages.talker_decode.batch_sizes == dense_b64
    assert throughput.stages.code_predictor.batch_sizes == dense_b64
    assert throughput.stages.codec.batches.warm_full == dense_b64
    assert throughput.stages.codec.frames.warm == (*range(1, 13), 25)


def test_ttfa_dense_b32_lowers_exercised_generation_cohorts_to_b24() -> None:
    catalog = CaptureCatalog.from_config(
        ProfileLoader().load_profile(ExecutionProfile.TTFA).stages
    )

    for logical_rows in range(17, 25):
        assert catalog.lower_talker_decode(logical_rows)[0].capture_batch_size == 24
        assert catalog.lower_code_predictor(logical_rows)[0].capture_batch_size == 24
        assert catalog.lower_codec(
            CodecExecutionMode.WARM,
            model_frames=12,
            logical_rows=logical_rows,
        )[0].capture_batch_size == 24


def test_config_mapping_is_a_strict_recursive_overlay_on_a_named_profile() -> None:
    config = ProfileLoader().load_mapping(
        {
            "extends": "ttfa",
            "talker_decode": {"max_batch_size": 6, "batch_sizes": [1, 3, 6]},
            "code_predictor": {"max_batch_size": 6, "batch_sizes": [1, 3, 6]},
            "codec": {
                "max_batch_size": 8,
                "chunk_schedule": [3, 6],
                "suppressed_bootstrap_chunk_schedule": [3, 6],
                "batches": {
                    "whole_sequence_first_frame": [1, 3, 8],
                    "warm_full": [1, 3, 8],
                },
            },
            "kv_pages": 128,
        }
    )

    assert config.name == ExecutionProfile.TTFA.value
    assert config.stages.talker_decode == BatchCaptureConfig(6, (1, 3, 6))
    assert config.stages.code_predictor == BatchCaptureConfig(6, (1, 3, 6))
    assert config.stages.codec.max_batch_size == 8
    assert config.stages.codec.batches.whole_sequence_first_frame == (1, 3, 8)
    assert config.stages.codec.batches.whole_sequence_followup == (1, 2, 4, 8)
    assert config.stages.codec.batches.warm_full == (1, 3, 8)
    assert config.resources.kv_pages == 128

    with pytest.raises(ValueError, match="Unknown profile setting"):
        ProfileLoader().load_mapping({"extends": "ttfa", "talkre_decode": {}})
    with pytest.raises(ValueError, match="Unknown profile.codec setting"):
        ProfileLoader().load_mapping({"extends": "ttfa", "codec": {"batchs": {}}})


def test_yaml_overlay_is_loaded_fail_closed_and_has_stable_resolved_provenance(tmp_path) -> None:
    path = tmp_path / "engine.yaml"
    path.write_text(
        """
extends: ttfa
talker_decode:
  max_batch_size: 24
  batch_sizes: [1, 2, 4, 8, 12, 16, 24]
code_predictor:
  max_batch_size: 24
  batch_sizes: [1, 2, 4, 8, 12, 16, 24]
""".lstrip(),
        encoding="utf-8",
    )

    first = ProfileLoader().load_yaml(path)
    second = ProfileLoader().load_yaml(path)

    assert first == second
    assert first.stages.talker_decode.max_batch_size == 24
    assert first.stages.talker_decode.batch_sizes[-1] == 24
    assert first.to_dict()["profile"] == "ttfa"
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    assert json.loads(first.canonical_json())["talker_decode"]["batch_sizes"][-1] == 24

    path.write_text("extends: ttfa\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown profile setting"):
        ProfileLoader().load_yaml(path)


def test_throughput_catalog_contains_sparse_25_frame_captures_only_when_declared() -> None:
    catalog = CaptureCatalog.from_config(
        ProfileLoader().load_profile(ExecutionProfile.THROUGHPUT).stages
    )
    assert CodecCaptureKey(CodecExecutionMode.COLD, 25, 64) in catalog.codec
    assert CodecCaptureKey(CodecExecutionMode.COLD, 8, 64) in catalog.codec
    assert CodecCaptureKey(CodecExecutionMode.COLD, 8, 1) not in catalog.codec
    assert CodecCaptureKey(CodecExecutionMode.WARM, 25, 64) in catalog.codec
    assert CodecCaptureKey(CodecExecutionMode.WARM, 13, 1) not in catalog.codec
    assert catalog.terminal_pad_frames == (12, 25)


def test_decode_and_cp_split_above_capacity_and_pad_only_the_tail() -> None:
    catalog = CaptureCatalog.from_config(
        ProfileLoader().load_profile(ExecutionProfile.TTFA).stages
    )
    decode = catalog.lower_talker_decode(45)
    cp = catalog.lower_code_predictor(45)
    expected = [(32, 32, 0), (13, 16, 3)]
    assert [(part.logical_rows, part.capture_batch_size, part.padding) for part in decode] == expected
    assert [(part.logical_rows, part.capture_batch_size, part.padding) for part in cp] == expected


def test_prefill_selects_tightest_token_capacity_and_splits_at_cap() -> None:
    catalog = CaptureCatalog.from_config(
        ProfileLoader().load_profile(ExecutionProfile.TTFA).stages
    )
    exact = catalog.lower_talker_prefill((10, 10, 10))
    mixed = catalog.lower_talker_prefill((11, 17, 9))
    split = catalog.lower_talker_prefill((10,) * 9)

    assert exact[0].key.token_capacity == 30
    assert exact[0].capture_batch_size == 3
    assert mixed[0].key.token_capacity == 40
    assert mixed[0].capture_batch_size == 4
    assert mixed[0].padding == 1
    assert [(part.logical_rows, part.capture_batch_size) for part in split] == [(8, 8), (1, 1)]


def test_codec_requires_exact_mode_and_frame_capture_without_eager_fallback() -> None:
    catalog = CaptureCatalog.from_config(
        ProfileLoader().load_profile(ExecutionProfile.TTFA).stages
    )
    parts = catalog.lower_codec(CodecExecutionMode.WARM, model_frames=12, logical_rows=37)
    assert [(part.logical_rows, part.capture_batch_size) for part in parts] == [(32, 32), (5, 5)]
    with pytest.raises(CaptureCoverageError, match="13 frames"):
        catalog.lower_codec(CodecExecutionMode.WARM, model_frames=13, logical_rows=1)


def test_ttfa_deadline_aware_whole_sequence_phase_cap_is_distinct_from_b32_acceleration_capture() -> None:
    catalog = CaptureCatalog.from_config(
        ProfileLoader().load_profile(ExecutionProfile.TTFA).stages
    )

    assert CodecCaptureKey(CodecExecutionMode.WHOLE_SEQUENCE, 1, 32) in catalog.codec
    assert catalog.codec_batch_capacity(
        CodecExecutionMode.WHOLE_SEQUENCE,
        model_frames=1,
    ) == 8


def test_empty_terminal_is_the_only_metadata_only_action() -> None:
    catalog = CaptureCatalog.from_config(
        ProfileLoader().load_profile(ExecutionProfile.TTFA).stages
    )
    action = catalog.lower_empty_terminal(3)
    assert action.metadata_only
    assert action.key is None
    with pytest.raises(ValueError):
        catalog.lower_empty_terminal(0)


@pytest.mark.parametrize(
    "config",
    [
        BatchCaptureConfig,
        PrefillCaptureConfig,
    ],
)
def test_capture_configs_reject_boolean_and_nonpositive_capacity(config) -> None:
    kwargs = {"max_batch_size": True, "batch_sizes": (1,)}
    if config is PrefillCaptureConfig:
        kwargs.update(
            token_buckets=(8,),
            exact_sequence_lengths=(2,),
            exact_batch_sizes=(1,),
        )
    with pytest.raises(ValueError, match="positive integer"):
        config(**kwargs)


@pytest.mark.parametrize(
    "batch_sizes",
    [(), (1, 1), (2, 1), (1, 5)],
)
def test_batch_capture_surface_is_nonempty_ordered_unique_and_bounded(
    batch_sizes: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        BatchCaptureConfig(max_batch_size=4, batch_sizes=batch_sizes)


def test_prefill_exact_capture_surface_cannot_exceed_its_dynamic_cap() -> None:
    with pytest.raises(ValueError, match="exact prefill"):
        PrefillCaptureConfig(
            max_batch_size=4,
            batch_sizes=(1, 4),
            token_buckets=(8, 16),
            exact_sequence_lengths=(2,),
            exact_batch_sizes=(1, 5),
        )


def test_codec_capture_subsets_are_closed_over_declared_frames_and_capacity() -> None:
    with pytest.raises(ValueError, match="uncaptured cold"):
        CodecFrameCaptureConfig(
            cold=(4,),
            cold_terminal_partial=(5,),
            warm=(1,),
            warm_full=(1,),
            terminal_pad=(1,),
        )
    with pytest.raises(ValueError, match="full-cohort"):
        CodecFrameCaptureConfig(
            cold=(4,),
            warm=(1, 2),
            warm_full=(1,),
            terminal_pad=(2,),
        )
    with pytest.raises(ValueError, match="exceeds max_batch_size"):
        CodecCaptureConfig(
            max_batch_size=4,
            chunk_schedule=(1,),
            suppressed_bootstrap_chunk_schedule=(1,),
            batches=CodecBatchCaptureConfig(cold=(1, 5)),
        )


@pytest.mark.parametrize(
    ("kind", "lead_s"),
    [
        ("deadline_aware", 1.0),
        (RequiredSchedulingPolicy.DEADLINE_AWARE, True),
        (RequiredSchedulingPolicy.ROUND_ROBIN, 1.0),
    ],
)
def test_scheduling_policy_config_requires_typed_policy_and_numeric_pressing_lead(
    kind: object,
    lead_s: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SchedulingPolicyConfig(kind, lead_s)  # type: ignore[arg-type]


def test_capture_catalog_is_derived_from_custom_validated_capacity_not_profile_constants() -> None:
    base = ProfileLoader().load_profile(ExecutionProfile.TTFA)
    config = replace(
        base.stages,
        talker_decode=BatchCaptureConfig(6, (1, 3, 6)),
        code_predictor=BatchCaptureConfig(5, (1, 2, 5)),
    )
    catalog = CaptureCatalog.from_config(config)

    assert sorted(key.capture_batch_size for key in catalog.talker_decode) == [1, 3, 6]
    assert sorted(key.capture_batch_size for key in catalog.code_predictor) == [1, 2, 5]
    assert [part.capture_batch_size for part in catalog.lower_talker_decode(7)] == [6, 1]
    assert [part.capture_batch_size for part in catalog.lower_code_predictor(7)] == [5, 2]


@pytest.mark.parametrize(
    ("constructor", "args"),
    [
        (TalkerDecodeCaptureKey, (0,)),
        (CodePredictorCaptureKey, (True,)),
        (TalkerPrefillCaptureKey, (1, 0, None)),
        (CodecCaptureKey, ("warm", 2, 1)),
        (CodecCaptureKey, (CodecExecutionMode.WARM, 0, 1)),
    ],
)
def test_capture_keys_are_validated_even_when_constructed_outside_the_catalog(
    constructor,
    args: tuple[object, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        constructor(*args)


@pytest.mark.parametrize("logical_rows", [True, False, 0, -1, 1.0])
def test_catalog_row_lowering_rejects_nonpositive_or_noninteger_counts(
    logical_rows: object,
) -> None:
    catalog = CaptureCatalog.from_config(
        ProfileLoader().load_profile(ExecutionProfile.TTFA).stages
    )
    with pytest.raises((TypeError, ValueError), match="logical batch"):
        catalog.lower_talker_decode(logical_rows)  # type: ignore[arg-type]


@pytest.mark.parametrize("lengths", [(True,), (1.0,), (0,), (-1,)])
def test_prefill_lowering_requires_positive_integer_sequence_lengths(
    lengths: tuple[object, ...],
) -> None:
    catalog = CaptureCatalog.from_config(
        ProfileLoader().load_profile(ExecutionProfile.TTFA).stages
    )
    with pytest.raises((TypeError, ValueError), match="sequence lengths"):
        catalog.lower_talker_prefill(lengths)  # type: ignore[arg-type]
