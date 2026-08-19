"""Composition root for the Qwen3-TTS Engine."""

from __future__ import annotations

import torch

from nari_qwen3_tts.config import EngineConfig
from nari_qwen3_tts.contract.model import SynthesisModelSpec
from nari_qwen3_tts.engine.engine import Engine
from nari_qwen3_tts.engine.pipeline import SynthesisPipeline
from nari_qwen3_tts.executor.build import build_cuda_execution
from nari_qwen3_tts.executor.executor import Executor
from nari_qwen3_tts.model.checkpoint import LoadedModelAssets
from nari_qwen3_tts.planner.catalog import CaptureCatalog
from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader, ResolvedProfile


@torch.inference_mode()
def _build_engine_components(
    assets: LoadedModelAssets,
    *,
    config: ResolvedProfile | None = None,
    capture: bool = True,
    trace_enabled: bool = False,
) -> tuple[Executor, SynthesisPipeline, CaptureCatalog]:
    resolved = config or ProfileLoader().load_profile(ExecutionProfile.TTFA)
    catalog = CaptureCatalog.from_config(resolved.stages)
    executor = build_cuda_execution(
        assets,
        config=resolved,
        required_keys=catalog.required_keys,
    )
    if capture:
        executor.capture_all()
        if not executor.health().ready:
            raise RuntimeError("Qwen3-TTS CUDA execution is not ready after capture startup")
    model = assets.model_config
    pipeline = SynthesisPipeline(
        executor=executor,
        capture_catalog=catalog,
        policy_config=resolved.policy,
        model_config=SynthesisModelSpec(
            codec_eos_token_id=model.talker.codec_eos_token_id,
            talker_vocab_size=model.talker.vocab_size,
            text_vocab_size=model.talker.text_vocab_size,
            num_codebooks=model.num_code_groups,
            samples_per_frame=executor.codec.samples_per_frame,
        ),
        max_in_flight_rows=(
            resolved.stages.talker_decode.max_batch_size
            + resolved.stages.codec.max_batch_size
        ),
        trace_enabled=trace_enabled,
    )
    return executor, pipeline, catalog


@torch.inference_mode()
def build_qwen3_tts_engine(
    model,
    assets: LoadedModelAssets,
    *,
    config=None,
    engine_config: EngineConfig | None = None,
    capture: bool = True,
    trace_enabled: bool = False,
) -> Engine:
    """Compose the production Engine from concrete model, planner, and executor."""

    executor, pipeline, capture_catalog = _build_engine_components(
        assets,
        config=config,
        capture=capture,
        trace_enabled=trace_enabled,
    )
    return Engine(
        model,
        executor=executor,
        pipeline=pipeline,
        capture_catalog=capture_catalog,
        config=engine_config,
    )


__all__ = ["build_qwen3_tts_engine"]
