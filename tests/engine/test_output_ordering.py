from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from server.fakes import (
    FakeEngine,
    FakeEngineBackend,
    FakeRequestSnapshot,
    FakeRequestState,
    request,
)

import nari_qwen3_tts.engine.output as output_module
import nari_qwen3_tts.executor.pcm as pcm_module
from nari_qwen3_tts.contract.errors import BackpressureExceeded
from nari_qwen3_tts.engine.engine import _ClientSession
from nari_qwen3_tts.engine.output import OutputQueue
from nari_qwen3_tts.executor.pcm import PcmTransferPool


class _RecordingStream:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events
        self._terminal = False
        self.values: list[bytes] = []

    @property
    def terminal(self) -> bool:
        return self._terminal

    def publish(self, value: bytes) -> None:
        self.events.append(("publish", value))
        self.values.append(value)

    def close(self) -> None:
        self.events.append(("terminal", None))
        self._terminal = True

    def fail(self, error: Exception) -> None:
        self.events.append(("fail", error))
        self._terminal = True


def _record(request_id: str, stream: object) -> _ClientSession:
    return _ClientSession(
        request_id=request_id,
        request=request(request_id),
        stream=stream,
        admitted=True,
    )


def test_pcm_publish_precedes_playback_credit_terminal_and_release() -> None:
    events: list[tuple[str, object]] = []

    class _Port(FakeEngineBackend):
        def record_pcm_routed(self, request_id, *, pcm_bytes, routed_at_s):
            events.append(("playback_credit", (request_id, pcm_bytes)))
            return super().record_pcm_routed(
                request_id,
                pcm_bytes=pcm_bytes,
                routed_at_s=routed_at_s,
            )

        def remove(self, request_id):
            events.append(("release", request_id))
            return super().remove(request_id)

    port = _Port()
    port.requests["one"] = FakeRequestState((b"\x01\x00",), cursor=1)
    service = FakeEngine(port)
    service._records["one"] = _record("one", _RecordingStream(events))
    service._set_metrics(active=1)

    service._poll_records()

    assert [name for name, _value in events] == [
        "publish",
        "playback_credit",
        "terminal",
        "release",
    ]
    assert port.routed[0][:2] == ("one", 2)
    assert service.metrics().completed == 1


def test_failed_publish_never_advances_playback_credit() -> None:
    class _FullStream(_RecordingStream):
        def publish(self, value: bytes) -> None:
            del value
            raise BackpressureExceeded("full")

    events: list[tuple[str, object]] = []
    port = FakeEngineBackend()
    port.requests["slow"] = FakeRequestState((b"\x01\x00",), cursor=1)
    service = FakeEngine(port)
    service._records["slow"] = _record("slow", _FullStream(events))
    service._set_metrics(active=1)

    service._poll_records()

    assert port.routed == []
    assert "slow" not in port.requests
    assert "slow" in port.removed
    assert [name for name, _value in events] == ["fail"]
    assert service.metrics().backpressured == 1


def test_pcm_poll_traverses_admission_order_without_blocking_ready_requests() -> None:
    events: list[tuple[str, object]] = []

    class _IndependentPort(FakeEngineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.snapshots: list[str] = []

        def snapshot(self, request_id, *, pcm_start_index=0):
            self.snapshots.append(request_id)
            if request_id == "waiting":
                return FakeRequestSnapshot((), False, False, False)
            return super().snapshot(request_id, pcm_start_index=pcm_start_index)

    port = _IndependentPort()
    port.requests["waiting"] = FakeRequestState(())
    port.requests["ready"] = FakeRequestState((b"\x02\x00",), cursor=1)
    service = FakeEngine(port)
    waiting = _RecordingStream(events)
    ready = _RecordingStream(events)
    service._records["waiting"] = _record("waiting", waiting)
    service._records["ready"] = _record("ready", ready)
    service._set_metrics(active=2)

    service._poll_records()

    assert port.snapshots == ["waiting", "ready"]
    assert waiting.values == []
    assert ready.values == [b"\x02\x00"]
    assert "waiting" in service._records
    assert "ready" not in service._records


class _FakeDeviceTensor:
    def __init__(self, events: list[str], dtype: object, *, samples: int = 2) -> None:
        self.events = events
        self.dtype = dtype
        self.device = SimpleNamespace(index=0)
        self.is_cuda = True
        self.samples = samples

    def detach(self):
        self.events.append("source_detach")
        return self

    def contiguous(self):
        self.events.append("source_contiguous")
        return self

    def reshape(self, *shape):
        del shape
        self.events.append("source_reshape")
        return self

    def numel(self) -> int:
        return self.samples


class _FakeHost:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __getitem__(self, key):
        del key
        return self

    def copy_(self, source, *, non_blocking=False):
        del source
        self.events.append(f"d2h_copy:{non_blocking}")
        return self

    def numpy(self):
        self.events.append("host_bytes")
        return SimpleNamespace(tobytes=lambda: b"\x01\x00\x02\x00")


class _FakeEvent:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.ready = False

    def record(self, stream) -> None:
        del stream
        self.events.append("d2h_event_record")

    def query(self) -> bool:
        self.events.append("d2h_event_query")
        return self.ready

    def synchronize(self) -> None:
        self.events.append("d2h_event_synchronize")
        self.ready = True


class _FakeStream:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def wait_stream(self, stream) -> None:
        del stream
        self.events.append("d2h_wait_compute")


def _fake_torch(events: list[str], event: _FakeEvent):
    dtype = object()
    stream = _FakeStream(events)

    class _Cuda:
        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def current_stream(device):
            del device
            events.append("current_compute_stream")
            return object()

        @staticmethod
        def stream(value):
            assert value is stream
            events.append("d2h_stream_context")
            return nullcontext()

        @staticmethod
        def Event(*, blocking):
            assert blocking is False
            return event

    return SimpleNamespace(Tensor=_FakeDeviceTensor, int16=dtype, cuda=_Cuda()), dtype, stream


def test_codec_submit_defers_d2h_until_poll_and_waits_for_event(monkeypatch) -> None:
    events: list[str] = []
    event = _FakeEvent(events)
    fake_torch, dtype, stream = _fake_torch(events, event)
    source = _FakeDeviceTensor(events, dtype)
    pool = PcmTransferPool(maximum_samples=8, device=None)
    pool._hosts = [_FakeHost(events)]
    pool._streams = {0: stream}
    output = OutputQueue(pool)
    monkeypatch.setattr(output_module, "torch", fake_torch)
    monkeypatch.setattr(pcm_module, "torch", fake_torch)

    output.enqueue("one", source, terminal_after=True)

    # Codec completion only registers a device tensor. The first post-decision
    # output poll, not submission, starts the dedicated-stream D2H.
    assert events == []
    blocked = output.poll_ready(request_order=("one",))

    assert blocked == ()
    assert "d2h_wait_compute" in events
    assert "d2h_copy:True" in events
    assert "d2h_event_record" in events
    assert "host_bytes" not in events
    assert events.index("d2h_wait_compute") < events.index("d2h_copy:True")
    assert events.index("d2h_copy:True") < events.index("d2h_event_record")

    event.ready = True
    ready = output.poll_ready(request_order=("one",))

    assert len(ready) == 1
    assert ready[0].value == b"\x01\x00\x02\x00"
    assert ready[0].terminal_after
    assert events.index("d2h_event_record") < events.index("host_bytes")
    assert pool._hosts


def test_cancelled_late_d2h_is_discarded_and_releases_pinned_slot(monkeypatch) -> None:
    events: list[str] = []
    event = _FakeEvent(events)
    fake_torch, dtype, stream = _fake_torch(events, event)
    source = _FakeDeviceTensor(events, dtype)
    pool = PcmTransferPool(maximum_samples=8, device=None)
    pool._hosts = [_FakeHost(events)]
    pool._streams = {0: stream}
    output = OutputQueue(pool)
    monkeypatch.setattr(output_module, "torch", fake_torch)
    monkeypatch.setattr(pcm_module, "torch", fake_torch)

    output.enqueue("one", source, terminal_after=True)
    assert output.poll_ready(request_order=("one",)) == ()
    assert pool._hosts == []

    output.cancel_request("one")
    assert output.poll_ready(request_order=("one",)) == ()
    assert "d2h_event_synchronize" not in events

    event.ready = True
    discarded = output.poll_ready(request_order=("one",))

    assert len(discarded) == 1 and discarded[0].discarded
    assert len(pool._hosts) == 1
    assert "d2h_event_synchronize" not in events
    assert "host_bytes" not in events
