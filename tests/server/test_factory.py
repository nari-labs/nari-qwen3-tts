from __future__ import annotations

from scheduler.test_pipeline_loop import _input_plan

from nari_qwen3_tts.contract.request import SynthesisRequest
from nari_qwen3_tts.engine.admission import make_admitted_request
from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader


def test_live_runtime_request_extracts_terminal_text_eos_without_redefining_layout() -> None:
    request = SynthesisRequest(
        text="live",
        voice="aiden",
        language="english",
        non_streaming_mode=False,
        do_sample=False,
    )
    plan = _input_plan(1, streaming=True)
    runtime_request = make_admitted_request(
        request_id="live",
        request=request,
        talker_plan=plan,
        execution_config=ProfileLoader().load_profile(ExecutionProfile.BALANCED),
        admitted_at_s=1.0,
        input_finished=False,
    )
    continuation = runtime_request.talker_input.continuation
    assert continuation.token_ids.tolist() == [41]
    assert continuation.terminal_token_id.tolist() == [90]
    assert not continuation.input_finished
    assert plan.continuations[0].token_ids.tolist() == [41, 90]
