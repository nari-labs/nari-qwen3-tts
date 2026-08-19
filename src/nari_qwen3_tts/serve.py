"""Explicit single-process production server driver."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict
from pathlib import Path

# `nari_qwen3_tts.config` is dataclass/pathlib only, so naming the default model
# here does not undo the deliberately lazy heavy imports inside `main`.
from nari_qwen3_tts.config import DEFAULT_MODEL_ID


def _run_uvicorn(app, *, host: str, port: int, uvicorn_module=None) -> None:
    if uvicorn_module is None:
        import uvicorn as uvicorn_module
    uvicorn_module.run(app, host=host, port=port, access_log=False)


def _freeze_serving_heap() -> None:
    """Exclude the initialized model/capture heap from later cyclic-GC scans."""

    automatic_gc_enabled = gc.isenabled()
    for generation in range(3):
        gc.collect(generation)
    gc.freeze()
    if automatic_gc_enabled:
        gc.enable()


def _diagnostic_trace(*, max_events: int):
    """Install the low-allocation trace only after the serving heap is frozen."""

    from nari_qwen3_tts.engine.trace import TraceRecorder

    gc.disable()
    return TraceRecorder(
        enabled=True,
        max_events=max_events,
        direct_object_ring=True,
        timestamps=True,
    )


def _write_diagnostic(path: Path, *, engine, execution_config) -> None:
    trace = engine.pipeline.trace
    health = engine.executor.health()
    metrics = engine.metrics()
    payload = {
        "schema": "nari_qwen3_tts_server_diagnostic_v1",
        "trace_enabled": trace.enabled,
        "trace_total_events": trace.total_events,
        "trace_dropped_events": trace.dropped_events,
        "trace": trace.normalized(),
        "execution_config_sha256": execution_config.sha256,
        "execution_health": asdict(health),
        "service_metrics": asdict(metrics),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nari-qwen3-tts-server")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", choices=("ttfa", "balanced", "throughput"))
    parser.add_argument(
        "--engine-config",
        type=Path,
        help="strict YAML overlay for stage capacities and CUDA Graph capture surfaces",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-active-requests", type=int, default=128)
    parser.add_argument("--max-buffered-pcm-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--startup-timeout-seconds", type=float, default=1_800.0)
    parser.add_argument(
        "--readiness-text",
        default="The Nari Qwen three T T S engine is ready.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="write an opt-in causal diagnostic after graceful shutdown",
    )
    parser.add_argument("--trace-max-events", type=int, default=250_000)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    """Reject every out-of-range numeric argument before any import or load.

    Argparse only proves these values parse as numbers. Leaving the range checks
    to the objects that consume them would report a typo in a service capacity
    after the model and the complete CUDA Graph capture are already resident.
    """

    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be in [1, 65535]")
    if args.trace_max_events < 1:
        raise ValueError("trace max events must be positive")
    if args.max_active_requests < 1:
        raise ValueError("max active requests must be positive")
    if args.max_buffered_pcm_bytes < 1:
        raise ValueError("max buffered PCM bytes must be positive")
    if not math.isfinite(args.startup_timeout_seconds) or args.startup_timeout_seconds <= 0:
        raise ValueError("startup timeout seconds must be finite and positive")


def _load_execution_config(*, profile: str | None, config_path: Path | None):
    from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader

    loader = ProfileLoader()

    if config_path is None:
        # TTFA is this engine's primary metric, so an unqualified server is the
        # one tuned for it. Benchmark runs name the profile explicitly so the
        # selected latency/throughput tradeoff is part of the result.
        return loader.load_profile(ExecutionProfile(profile or "ttfa"))
    base = ExecutionProfile(profile) if profile is not None else None
    return loader.load_yaml(config_path, base=base)


def main() -> None:
    args = _parser().parse_args()
    _validate_arguments(args)
    execution_config = _load_execution_config(
        profile=args.profile,
        config_path=args.engine_config,
    )

    from nari_qwen3_tts.api.app import create_app
    from nari_qwen3_tts.config import ApiConfig, EngineConfig, ModelAssetConfig
    from nari_qwen3_tts.contract.request import SynthesisRequest
    from nari_qwen3_tts.factory import build_qwen3_tts_engine
    from nari_qwen3_tts.model.public import Qwen3TTSModel

    model = Qwen3TTSModel(
        ModelAssetConfig(
            model_id=args.model,
            revision=args.revision,
            cache_dir=args.model_cache_dir,
            device=args.device,
            require_h100=True,
            local_files_only=args.local_files_only,
        )
    )
    print("Loading model...", flush=True)
    assets = model.load_assets()
    print(
        json.dumps(
            {
                "event": "qwen3_tts_execution_config",
                "sha256": execution_config.sha256,
                "source": str(args.engine_config.resolve()) if args.engine_config else "packaged-profile",
                "config": execution_config.to_dict(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("Building engine...", flush=True)
    engine = build_qwen3_tts_engine(
        model,
        assets,
        config=execution_config,
        engine_config=EngineConfig(
            max_active_requests=args.max_active_requests,
            max_buffered_pcm_bytes=args.max_buffered_pcm_bytes,
        ),
        capture=True,
        trace_enabled=False,
    )
    engine.start(
        SynthesisRequest(
            text=args.readiness_text,
            voice="aiden",
            language="english",
            non_streaming_mode=True,
            random_seed=1_234,
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=4,
            ignore_eos=True,
            skip_fixed_bootstrap_audio=False,
            stream_chunk_schedule=(1,),
        ),
        timeout_s=args.startup_timeout_seconds,
    )
    print("Startup complete.", flush=True)
    api_config = ApiConfig(startup_timeout_s=args.startup_timeout_seconds)
    app = create_app(
        engine,
        text_frontend=model.text,
        config=api_config,
        model_id=args.model,
    )
    _freeze_serving_heap()
    if args.trace_output is not None:
        engine.pipeline.trace = _diagnostic_trace(max_events=args.trace_max_events)
    try:
        _run_uvicorn(app, host=args.host, port=args.port)
    finally:
        engine.stop(timeout_s=30.0)
        if args.trace_output is not None:
            _write_diagnostic(
                args.trace_output,
                engine=engine,
                execution_config=execution_config,
            )


if __name__ == "__main__":
    main()
