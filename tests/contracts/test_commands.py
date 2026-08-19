from __future__ import annotations

from concurrent.futures import Future

import pytest


def test_engine_commands_fail_closed_before_queueing() -> None:
    from nari_qwen3_tts.contract.command import (
        AppendText,
        CancelRequest,
        StopEngine,
        SubmitRequest,
    )
    from nari_qwen3_tts.contract.request import SynthesisRequest

    submit_reply: Future = Future()
    submit = SubmitRequest(
        "request-1",
        SynthesisRequest(text="hello"),
        False,
        (),
        (),
        True,
        submit_reply,
    )
    assert submit.reply is submit_reply

    append_reply: Future = Future()
    append = AppendText("request-1", 0, (1,), False, append_reply)
    assert append.sequence == 0
    assert CancelRequest("request-1", Future()).request_id == "request-1"
    assert isinstance(StopEngine(Future()).reply, Future)

    with pytest.raises(ValueError, match="request ID"):
        SubmitRequest("", SynthesisRequest(text="hello"), False, (), (), True, Future())
    with pytest.raises(ValueError, match="sequence"):
        AppendText("request-1", -1, (1,), False, Future())
    with pytest.raises(TypeError, match="Future"):
        StopEngine(object())  # type: ignore[arg-type]
