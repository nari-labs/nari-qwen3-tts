from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _Fence:
    is_ready: bool = False
    waits: int = 0

    def ready(self) -> bool:
        return self.is_ready

    def wait(self) -> None:
        self.waits += 1
        self.is_ready = True


@dataclass
class _Submission:
    completion_fence: _Fence
    decision_fence: _Fence | None = None
    requires_host_finalize: bool = False


def test_submission_window_bounds_decisions_and_reaps_in_fifo_order() -> None:
    from nari_qwen3_tts.executor.submission import SubmissionWindow

    first = _Fence()
    second = _Fence()
    window = SubmissionWindow(max_decisions=2)
    window.record(decision_id=11, submissions=(_Submission(_Fence(), first),))
    window.record(decision_id=12, submissions=(_Submission(_Fence(), second),))

    assert window.decision_ids == (11, 12)
    assert window.can_submit is False
    with pytest.raises(RuntimeError, match="full"):
        window.record(decision_id=13, submissions=(_Submission(_Fence(), _Fence()),))

    second.is_ready = True
    window.reap_fences()
    assert window.decision_ids == (11, 12)
    first.is_ready = True
    window.reap_fences()
    assert window.decision_ids == ()
    assert window.can_submit is True


def test_submission_window_waits_only_the_oldest_decision_fence() -> None:
    from nari_qwen3_tts.executor.submission import SubmissionWindow

    first = _Fence()
    second = _Fence()
    window = SubmissionWindow(max_decisions=2)
    window.record(decision_id=1, submissions=(_Submission(_Fence(), first),))
    window.record(decision_id=2, submissions=(_Submission(_Fence(), second),))

    window.wait_oldest()

    assert first.waits == 1
    assert second.waits == 0
    assert window.decision_ids == (2,)


def test_host_finalize_poll_bypasses_slow_work_and_blocks_at_most_once() -> None:
    from nari_qwen3_tts.executor.submission import SubmissionWindow

    first_fence = _Fence()
    second_fence = _Fence(is_ready=True)
    first = _Submission(first_fence, _Fence(), requires_host_finalize=True)
    second = _Submission(second_fence, _Fence(), requires_host_finalize=True)
    window = SubmissionWindow(max_decisions=2)
    window.record(decision_id=1, submissions=(first,))
    window.record(decision_id=2, submissions=(second,))

    assert window.poll_host_ready(block_oldest=False) == (second,)
    assert window.poll_host_ready(block_oldest=True) == (first,)
    assert first_fence.waits == 1


def test_submission_window_rejects_duplicate_or_mixed_decision_records() -> None:
    from nari_qwen3_tts.executor.submission import SubmissionWindow

    window = SubmissionWindow(max_decisions=2)
    with pytest.raises(ValueError, match="positive"):
        window.record(decision_id=0, submissions=(_Submission(_Fence(), _Fence()),))
    with pytest.raises(ValueError, match="submissions"):
        window.record(decision_id=1, submissions=())
    window.record(decision_id=1, submissions=(_Submission(_Fence(), _Fence()),))
    with pytest.raises(ValueError, match="increasing"):
        window.record(decision_id=1, submissions=(_Submission(_Fence(), _Fence()),))


def test_submission_window_uses_decision_fence_not_host_completion_fence() -> None:
    from nari_qwen3_tts.executor.submission import SubmissionWindow

    host = _Fence(is_ready=True)
    decision = _Fence()
    window = SubmissionWindow(max_decisions=2)
    window.record(
        decision_id=1,
        submissions=(_Submission(host, decision, requires_host_finalize=True),),
    )

    assert window.poll_host_ready(block_oldest=False)
    window.reap_fences()
    assert window.decision_ids == (1,)
    decision.is_ready = True
    window.reap_fences()
    assert window.decision_ids == ()


def test_decision_fence_retires_while_host_completion_remains_pending() -> None:
    from nari_qwen3_tts.executor.submission import SubmissionWindow

    host = _Fence()
    window = SubmissionWindow(max_decisions=1)
    window.record(
        decision_id=1,
        submissions=(_Submission(host, _Fence(is_ready=True), True),),
    )

    window.reap_fences()

    assert window.can_submit is True
    assert window.decision_ids == ()
    assert window.poll_host_ready(block_oldest=False) == ()
    host.is_ready = True
    assert len(window.poll_host_ready(block_oldest=False)) == 1
