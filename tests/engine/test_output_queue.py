from __future__ import annotations

import torch
from scheduler.test_pipeline_loop import _request, _runtime

from nari_qwen3_tts.executor.pcm import PcmTransferPool


class _Pool:
    def __init__(self) -> None:
        self.begun = []

    def begin(self, source):
        self.begun.append(source)
        raise AssertionError("metadata registration must not begin D2H")


class _Event:
    def __init__(self, ready: bool) -> None:
        self.ready = ready
        self.waits = 0

    def synchronize(self) -> None:
        self.waits += 1
        self.ready = True


class _Transfer:
    def __init__(self, value: bytes, *, ready: bool) -> None:
        self.value = value
        self.event = _Event(ready)
        self.polls = 0
        self.discard_polls = 0

    def poll(self):
        self.polls += 1
        return self.value if self.event.ready else None

    def discard_when_ready(self) -> None:
        self.event.synchronize()

    def discard_if_ready(self) -> bool:
        self.discard_polls += 1
        return self.event.ready


class _TransferPool:
    def __init__(self, transfers) -> None:
        self.transfers = transfers
        self.begun = []

    def begin(self, source):
        self.begun.append(source)
        return self.transfers[id(source)]


def test_output_queue_registration_is_metadata_only() -> None:
    from nari_qwen3_tts.engine.output import OutputQueue

    pool = _Pool()
    output = OutputQueue(pool)
    first = torch.tensor([1, 2], dtype=torch.int16)
    second = torch.tensor([3], dtype=torch.int16)

    output.enqueue("later", second, terminal_after=True)
    output.enqueue("first", first, terminal_after=False)

    assert pool.begun == []
    assert output.pending_metadata(("first", "later")) == (
        ("first", first, False),
        ("later", second, True),
    )


def test_codec_commit_registers_output_only_after_successful_state_commit() -> None:
    from nari_qwen3_tts.engine.output import OutputQueue

    runtime, _execution = _runtime()
    output = OutputQueue(_Pool())
    runtime.attach_output_queue(output)
    runtime.admit(_request(0))

    runtime.step(now_s=0.0)
    runtime.step(now_s=0.1)
    assert not output.has_pending("request-0")

    runtime.step(now_s=0.2)

    state = runtime.request("request-0")
    pending = output.pending_metadata(("request-0",))
    assert len(pending) == 1
    assert pending[0][0] == "request-0"
    assert pending[0][1].dtype is torch.int16
    assert pending[0][2] is False
    assert state.codec.pending_outputs == 1
    assert output.begin_count == 0


def test_output_traversal_follows_admission_order() -> None:
    from nari_qwen3_tts.engine.output import OutputQueue

    output = OutputQueue(_Pool())
    values = {
        "b": torch.tensor([2], dtype=torch.int16),
        "a": torch.tensor([1], dtype=torch.int16),
    }
    output.enqueue("b", values["b"], terminal_after=False)
    output.enqueue("a", values["a"], terminal_after=False)

    assert tuple(item[0] for item in output.pending_metadata(("a", "b"))) == (
        "a",
        "b",
    )


def test_cleanup_releases_all_unstarted_metadata_without_d2h() -> None:
    from nari_qwen3_tts.engine.output import OutputQueue

    pool = _Pool()
    output = OutputQueue(pool)
    output.enqueue("one", torch.tensor([1], dtype=torch.int16), terminal_after=False)
    output.enqueue("one", torch.tensor([], dtype=torch.int16), terminal_after=True)

    assert output.discard_request("one") == 2
    assert not output.has_pending("one")
    assert pool.begun == []


def test_first_poll_begins_d2h_once_and_waits_for_visibility() -> None:
    from nari_qwen3_tts.engine.output import OutputQueue

    source = torch.tensor([7], dtype=torch.int16)
    transfer = _Transfer(b"\x07\x00", ready=False)
    pool = _TransferPool({id(source): transfer})
    output = OutputQueue(pool)
    output.enqueue("one", source, terminal_after=True)

    assert output.poll_ready(request_order=("one",)) == ()
    assert pool.begun == [source]
    assert output.poll_ready(request_order=("one",)) == ()
    assert pool.begun == [source]

    transfer.event.ready = True
    deliveries = output.poll_ready(request_order=("one",))

    assert [(item.request_id, item.value, item.terminal_after) for item in deliveries] == [
        ("one", b"\x07\x00", True)
    ]
    assert not output.has_pending("one")


def test_pending_request_head_does_not_block_another_request() -> None:
    from nari_qwen3_tts.engine.output import OutputQueue

    slow = torch.tensor([1], dtype=torch.int16)
    fast = torch.tensor([2], dtype=torch.int16)
    pool = _TransferPool(
        {
            id(slow): _Transfer(b"\x01\x00", ready=False),
            id(fast): _Transfer(b"\x02\x00", ready=True),
        }
    )
    output = OutputQueue(pool)
    output.enqueue("slow", slow, terminal_after=False)
    output.enqueue("fast", fast, terminal_after=False)

    deliveries = output.poll_ready(request_order=("slow", "fast"))

    assert [(item.request_id, item.value) for item in deliveries] == [
        ("fast", b"\x02\x00")
    ]
    assert output.has_pending("slow")


def test_request_fifo_prestarts_successor_d2h_but_publishes_only_ready_prefix() -> None:
    from nari_qwen3_tts.engine.output import OutputQueue

    first = torch.tensor([1], dtype=torch.int16)
    second = torch.tensor([2], dtype=torch.int16)
    pool = _TransferPool(
        {
            id(first): _Transfer(b"\x01\x00", ready=False),
            id(second): _Transfer(b"\x02\x00", ready=True),
        }
    )
    output = OutputQueue(pool)
    output.enqueue("one", first, terminal_after=False)
    output.enqueue("one", second, terminal_after=True)

    assert output.poll_ready(request_order=("one",)) == ()
    assert pool.begun == [first, second]

    pool.transfers[id(first)].event.ready = True
    deliveries = output.poll_ready(request_order=("one",))

    assert [(item.source, item.value, item.terminal_after) for item in deliveries] == [
        (first, b"\x01\x00", False),
        (second, b"\x02\x00", True),
    ]


def test_blocking_poll_waits_only_the_oldest_request_head() -> None:
    from nari_qwen3_tts.engine.output import OutputQueue

    first = torch.tensor([1], dtype=torch.int16)
    second = torch.tensor([2], dtype=torch.int16)
    first_transfer = _Transfer(b"\x01\x00", ready=False)
    second_transfer = _Transfer(b"\x02\x00", ready=False)
    pool = _TransferPool(
        {id(first): first_transfer, id(second): second_transfer}
    )
    output = OutputQueue(pool)
    output.enqueue("first", first, terminal_after=False)
    output.enqueue("second", second, terminal_after=False)

    deliveries = output.poll_ready(
        request_order=("first", "second"),
        block_oldest=True,
    )

    assert [(item.request_id, item.value) for item in deliveries] == [
        ("first", b"\x01\x00")
    ]
    assert first_transfer.event.waits == 1
    assert second_transfer.event.waits == 0


def test_cancelled_started_d2h_is_reclaimed_without_blocking_or_materializing() -> None:
    from nari_qwen3_tts.engine.output import OutputQueue

    source = torch.tensor([1], dtype=torch.int16)
    transfer = _Transfer(b"\x01\x00", ready=False)
    output = OutputQueue(_TransferPool({id(source): transfer}))
    output.enqueue("one", source, terminal_after=True)

    assert output.poll_ready(request_order=("one",)) == ()
    output.cancel_request("one")

    assert output.poll_ready(request_order=("one",)) == ()
    assert transfer.event.waits == 0
    assert transfer.polls == 1

    transfer.event.ready = True
    deliveries = output.poll_ready(request_order=("one",))

    assert len(deliveries) == 1
    assert deliveries[0].discarded
    assert deliveries[0].value == b""
    assert transfer.polls == 1
    assert transfer.event.waits == 0
    assert not output.has_pending("one")


def test_pcm_delivery_retains_the_exact_committed_source() -> None:
    from nari_qwen3_tts.engine.output import OutputQueue

    source = torch.tensor([9], dtype=torch.int16)
    output = OutputQueue(PcmTransferPool(maximum_samples=0, device=None))
    output.enqueue("one", source, terminal_after=True)

    delivery = output.poll_ready(request_order=("one",))[0]

    assert delivery.source is source


def test_publish_precedes_playback_credit_and_terminal_close() -> None:
    from nari_qwen3_tts.contract.request import SynthesisRequest
    from nari_qwen3_tts.engine.engine import Engine, _ClientSession
    from nari_qwen3_tts.engine.output import OutputQueue

    runtime, _execution = _runtime()
    output = OutputQueue(PcmTransferPool(maximum_samples=0, device=None))
    runtime.attach_output_queue(output)
    runtime.admit(_request(0))
    runtime.step(now_s=0.0)
    runtime.step(now_s=0.1)
    runtime.step(now_s=0.2)
    state = runtime.request("request-0")
    state.codec.compute_terminal = True
    pending = output._requests["request-0"][0]
    pending.terminal_after = True
    delivery = output.poll_ready(request_order=("request-0",))[0]
    events = []

    class _Stream:
        terminal = False

        def publish(self, value: bytes) -> None:
            events.append(("publish", value))

        def close(self) -> None:
            events.append(("close", None))
            self.terminal = True

    stream = _Stream()
    record = _ClientSession(
        "request-0",
        SynthesisRequest(text="output"),
        stream,
        admitted=True,
    )
    engine = Engine.__new__(Engine)
    engine.pipeline = runtime
    engine._records = {"request-0": record}
    complete = runtime.complete_pcm_output

    def audited_complete(*args, **kwargs) -> None:
        complete(*args, **kwargs)
        events.append(("playback_credit", state.codec.emitted_duration_s))

    runtime.complete_pcm_output = audited_complete

    engine._deliver_pcm(delivery)

    assert [event[0] for event in events] == [
        "publish",
        "playback_credit",
        "close",
    ]
    assert state.codec.pending_outputs == 0
    assert state.codec.output_terminal
    assert state.codec.playback_started_at_s is not None
