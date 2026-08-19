from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from nari_qwen3_tts import serve
from nari_qwen3_tts.model import public
from nari_qwen3_tts.profile import ExecutionProfile


def test_production_server_freezes_the_initialized_serving_heap(monkeypatch) -> None:
    calls = []
    enabled = []
    disabled = []

    monkeypatch.setattr(serve.gc, "isenabled", lambda: True)
    monkeypatch.setattr(serve.gc, "collect", lambda generation: calls.append(generation) or 0)
    monkeypatch.setattr(serve.gc, "freeze", lambda: calls.append("freeze"))
    monkeypatch.setattr(serve.gc, "enable", lambda: enabled.append(True))
    monkeypatch.setattr(serve.gc, "disable", lambda: disabled.append(True))

    serve._freeze_serving_heap()

    assert calls == [0, 1, 2, "freeze"]
    assert enabled == [True]
    assert disabled == []


def test_production_server_preserves_disabled_automatic_gc(monkeypatch) -> None:
    enabled = []
    disabled = []

    monkeypatch.setattr(serve.gc, "isenabled", lambda: False)
    monkeypatch.setattr(serve.gc, "collect", lambda _generation: 0)
    monkeypatch.setattr(serve.gc, "freeze", lambda: None)
    monkeypatch.setattr(serve.gc, "enable", lambda: enabled.append(True))
    monkeypatch.setattr(serve.gc, "disable", lambda: disabled.append(True))

    serve._freeze_serving_heap()

    assert enabled == []
    assert disabled == []


def test_production_server_disables_per_request_access_logging() -> None:
    calls = []

    class Uvicorn:
        @staticmethod
        def run(app, **kwargs):
            calls.append((app, kwargs))

    app = object()
    serve._run_uvicorn(app, host="127.0.0.1", port=8000, uvicorn_module=Uvicorn)

    assert calls == [
        (app, {"host": "127.0.0.1", "port": 8000, "access_log": False})
    ]


def test_server_loads_a_yaml_execution_overlay_instead_of_only_hardcoded_profiles(tmp_path) -> None:
    path = tmp_path / "engine.yaml"
    path.write_text(
        """
extends: ttfa
talker_decode:
  max_batch_size: 24
  batch_sizes: [1, 2, 4, 8, 12, 16, 24]
""".lstrip(),
        encoding="utf-8",
    )

    config = serve._load_execution_config(profile=None, config_path=path)

    assert config.name == ExecutionProfile.TTFA.value
    assert config.stages.talker_decode.max_batch_size == 24
    assert config.stages.talker_decode.batch_sizes[-1] == 24


def test_explicit_server_profile_is_the_base_for_a_profileless_yaml_overlay(tmp_path) -> None:
    path = tmp_path / "engine.yaml"
    path.write_text("kv_pages: 128\n", encoding="utf-8")

    config = serve._load_execution_config(profile="throughput", config_path=path)

    assert config.name == ExecutionProfile.THROUGHPUT.value
    assert config.resources.kv_pages == 128
    assert config.stages.talker_decode.max_batch_size == 64


def test_server_rejects_invalid_execution_config_before_model_construction(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("extends: ttfa\nunknown: true\n", encoding="utf-8")
    constructed = []

    def reject_construction(*args, **kwargs) -> None:
        constructed.append((args, kwargs))
        raise AssertionError("model construction happened before config validation")

    monkeypatch.setattr(public.Qwen3TTSModel, "__init__", reject_construction)
    monkeypatch.setattr(
        sys,
        "argv",
        ["nari-qwen3-tts-server", "--engine-config", str(path)],
    )

    with pytest.raises(ValueError, match="Unknown profile setting"):
        serve.main()

    assert constructed == []


def test_diagnostic_trace_disables_gc_for_the_direct_object_ring(
    monkeypatch,
) -> None:
    disabled = []
    monkeypatch.setattr(serve.gc, "disable", lambda: disabled.append(True))
    monkeypatch.setattr(serve.gc, "isenabled", lambda: False)

    trace = serve._diagnostic_trace(max_events=123)

    assert disabled == [True]
    assert trace.enabled is True
    assert trace.max_events == 123
    assert trace._direct_object_ring is True
    assert trace._timestamps is True


def test_diagnostic_writer_uses_the_resolved_profile_hash_property(tmp_path) -> None:
    @dataclass(frozen=True)
    class Snapshot:
        value: int

    trace = SimpleNamespace(
        enabled=True,
        total_events=1,
        dropped_events=0,
        normalized=lambda: ({"kind": "probe"},),
    )
    engine = SimpleNamespace(
        pipeline=SimpleNamespace(trace=trace),
        executor=SimpleNamespace(health=lambda: Snapshot(1)),
        metrics=lambda: Snapshot(2),
    )
    config = serve._load_execution_config(profile="ttfa", config_path=None)
    output = tmp_path / "diagnostic.json"

    serve._write_diagnostic(output, engine=engine, execution_config=config)

    assert f'"execution_config_sha256": "{config.sha256}"' in output.read_text(
        encoding="utf-8"
    )
