from __future__ import annotations

import pytest

from nari_qwen3_tts.planner.catalog import CaptureCatalog
from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader


def _assert_partition(catalog: CaptureCatalog, slices, *, logical_rows: int) -> None:
    assert sum(value.logical_rows for value in slices) == logical_rows
    assert all(value.logical_start < value.logical_stop for value in slices)
    assert all(value.capture_batch_size >= value.logical_rows for value in slices)
    assert all(value.padding == value.capture_batch_size - value.logical_rows for value in slices)
    assert all(value.key is None or value.key in catalog.required_keys for value in slices)
    assert tuple(value.logical_start for value in slices) == tuple(
        [0, *[value.logical_stop for value in slices[:-1]]]
    )


@pytest.mark.parametrize("profile_name", ("ttfa", "balanced", "throughput"))
def test_planner_catalog_covers_every_declared_stage_surface(profile_name: str) -> None:
    config = ProfileLoader().load_profile(ExecutionProfile(profile_name))
    catalog = CaptureCatalog.from_config(config.stages)

    assert catalog.required_keys == frozenset(
        (
            *catalog.talker_prefill,
            *catalog.talker_decode,
            *catalog.code_predictor,
            *catalog.codec,
        )
    )
    row_limit = max(
        config.stages.talker_decode.max_batch_size,
        config.stages.code_predictor.max_batch_size,
        config.stages.codec.max_batch_size,
    )
    for rows in range(1, row_limit * 2 + 2):
        _assert_partition(catalog, catalog.lower_talker_decode(rows), logical_rows=rows)
        _assert_partition(catalog, catalog.lower_code_predictor(rows), logical_rows=rows)

    for key in catalog.codec:
        assert catalog.codec_batch_capacity(key.mode, model_frames=key.model_frames) >= 1
        for rows in range(1, config.stages.codec.max_batch_size * 2 + 2):
            _assert_partition(
                catalog,
                catalog.lower_codec(
                    key.mode,
                    model_frames=key.model_frames,
                    logical_rows=rows,
                ),
                logical_rows=rows,
            )

    empty = catalog.lower_empty_terminal(row_limit)
    assert empty.metadata_only and empty.key is None
    assert empty.logical_rows == row_limit


@pytest.mark.parametrize("profile_name", ("ttfa", "balanced", "throughput"))
def test_planner_catalog_prefill_lowering_covers_homogeneous_and_mixed_rows(
    profile_name: str,
) -> None:
    config = ProfileLoader().load_profile(ExecutionProfile(profile_name))
    catalog = CaptureCatalog.from_config(config.stages)
    supported_lengths = tuple(
        sorted(
            {
                *config.stages.talker_prefill.exact_sequence_lengths,
                1,
                max(1, config.stages.talker_prefill.token_buckets[0] // 2),
                config.stages.talker_prefill.token_buckets[0],
            }
        )
    )
    row_limit = config.stages.talker_prefill.max_batch_size
    surfaces = [
        (length,) * rows
        for length in supported_lengths
        for rows in range(1, row_limit + 1)
        if length * rows <= config.stages.talker_prefill.token_buckets[-1]
    ]
    surfaces.extend(
        tuple(supported_lengths[index % len(supported_lengths)] for index in range(rows))
        for rows in range(2, row_limit + 1)
    )
    for lengths in surfaces:
        try:
            slices = catalog.lower_talker_prefill(lengths)
        except Exception as error:
            assert "capture" in str(error).lower() or "prefill" in str(error).lower()
        else:
            _assert_partition(catalog, slices, logical_rows=len(lengths))
