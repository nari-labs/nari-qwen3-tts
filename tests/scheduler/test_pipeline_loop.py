from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch

from nari_qwen3_tts.contract import KVPublication, SynthesisStage
from nari_qwen3_tts.contract.model import SynthesisModelSpec
from nari_qwen3_tts.contract.request import AdmittedRequest, SynthesisRequest, TalkerPrompt, TextContinuation
from nari_qwen3_tts.contract.rng import CodePredictorSamplerRoute
from nari_qwen3_tts.engine.pipeline import SynthesisPipeline
from nari_qwen3_tts.executor.executor import Executor
from nari_qwen3_tts.executor.input_layout import (
    TalkerInputPlan,
)
from nari_qwen3_tts.executor.rows import (
    CodecRowsExecutionInput,
    CodePredictorRowsExecutionInput,
    TalkerDecodeRowsExecutionInput,
    TalkerExecutionResult,
    TalkerPrefillRowsExecutionInput,
)
from nari_qwen3_tts.executor.types import CodecResult, CodePredictorResult, TalkerResult
from nari_qwen3_tts.model.incremental_codec import IncrementalCodecState
from nari_qwen3_tts.planner import CaptureCatalog
from nari_qwen3_tts.planner.planner import PlanningWaitReasonCode as WaitReasonCode
from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader


class _Publication:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.published = False
        self.discarded = False

    def validate(self) -> None:
        if self.published or self.discarded:
            raise RuntimeError("stale publication")

    def proposal(self) -> KVPublication:
        return KVPublication(self.request_id, (), 1)

    def publish(self) -> None:
        self.validate()
        self.published = True

    def discard(self) -> None:
        self.validate()
        self.discarded = True


@dataclass
class _FakeExecution:
    """The only execution surface the runtime drives: direct static staging."""

    calls: list[tuple[str, tuple[str, ...], object]]
    completion_event_factory: object | None = None

    def add_request(self, request_id: str) -> None:
        del request_id  # this fake owns no per-request execution resources

    def remove_request(self, request_id: str) -> None:
        del request_id

    def talker_prefill_rows(self, key, call):
        del key
        self.calls.append(("prefill_rows", call.request_ids, call))
        tokens = torch.tensor([10 + int(request_id[-1]) for request_id in call.request_ids])
        hidden = torch.stack([torch.full((4,), float(token)) for token in tokens])
        return TalkerExecutionResult(
            result=TalkerResult(
                tokens=tokens,
                last_hidden=hidden,
                logits=torch.zeros((len(call.rows), 32)),
            ),
            next_seen_token_masks=torch.zeros((len(call.rows), 32), dtype=torch.bool),
            next_sampling_offsets=torch.tensor(
                [row.sampling.offset + 512 for row in call.rows]
            ),
            kv_publications=tuple(_Publication(request_id) for request_id in call.request_ids),
        )

    def talker_decode_rows(self, key, call):
        del key
        self.calls.append(("decode_rows", call.request_ids, call))
        tokens = torch.tensor([20 + int(request_id[-1]) for request_id in call.request_ids])
        hidden = torch.stack(tuple(row.talker_step_embed for row in call.rows)) + 1
        return TalkerExecutionResult(
            result=TalkerResult(
                tokens=tokens,
                last_hidden=hidden,
                logits=torch.zeros((len(call.rows), 32)),
            ),
            next_seen_token_masks=torch.ones((len(call.rows), 32), dtype=torch.bool),
            next_sampling_offsets=torch.tensor(
                [row.sampling.offset + 512 for row in call.rows]
            ),
            kv_publications=tuple(_Publication(request_id) for request_id in call.request_ids),
        )

    def code_predictor_rows(self, key, call):
        del key
        self.calls.append(("code_predictor_rows", tuple(range(len(call.rows))), call))
        frames = torch.stack(tuple(row.layer0_token for row in call.rows))[:, None].expand(-1, 16).clone()
        hidden = torch.stack(tuple(row.past_hidden for row in call.rows))
        return CodePredictorResult(frames=frames, codec_sum=hidden + 1)

    def codec_rows(self, key, call):
        del key
        self.calls.append(("codec_rows", tuple(range(len(call.rows))), call))
        frames = torch.stack(
            tuple(torch.stack(tuple(row.frames)) for row in call.rows)
        )
        pcm_rows = []
        for index, row in enumerate(call.rows):
            start = call.pcm_start_frame if row.pcm_start_frame is None else row.pcm_start_frame
            visible = call.visible_frames if row.visible_frames is None else row.visible_frames
            pcm_rows.append(
                frames[index, start : start + visible]
                .sum(1)
                .repeat_interleave(2)
                .to(torch.int16)
            )
        pcm_lengths = tuple(int(row.numel()) for row in pcm_rows)
        pcm = torch.zeros((len(pcm_rows), max(pcm_lengths, default=0)), dtype=torch.int16)
        for index, row_pcm in enumerate(pcm_rows):
            pcm[index, : row_pcm.numel()] = row_pcm
        states = None
        if any(row.state is not None for row in call.rows):
            states = tuple(
                IncrementalCodecState(
                    frame_position=frames.shape[1],
                    transformer_context_length=frames.shape[1],
                )
                for _ in call.rows
            )
        return CodecResult(
            pcm=pcm,
            states=states,
            terminal=call.terminal,
            pcm_lengths=pcm_lengths,
        )

    def empty_terminal(self, *, rows: int) -> None:
        self.calls.append(("empty_terminal", tuple(str(row) for row in range(rows)), rows))

    def _executor(self) -> Executor:
        from nari_qwen3_tts.contract import TalkerPrefillCaptureKey

        def replay_talker(key, values):
            method = (
                self.talker_prefill_rows
                if isinstance(key, TalkerPrefillCaptureKey)
                else self.talker_decode_rows
            )
            return method(key, values)

        return Executor(
            None,
            None,
            SimpleNamespace(replay=replay_talker),
            SimpleNamespace(replay=self.code_predictor_rows),
            SimpleNamespace(replay=self.codec_rows),
            None,
            self.completion_event_factory,
        )

    def preflight(self, decision, inputs) -> None:
        self._executor().preflight(decision, inputs)

    def submit(self, decision, inputs):
        return self._executor().submit(decision, inputs)

    def submit_preflighted(self, decision, inputs):
        return self._executor().submit_preflighted(decision, inputs)


def _input_plan(request_index: int, *, streaming: bool) -> TalkerInputPlan:
    pad = torch.tensor([99])
    continuation = TextContinuation(
        non_streaming_mode=not streaming,
        token_ids=(torch.tensor([40 + request_index, 90]) if streaming else pad),
        pad_token_id=pad,
    )
    length = 10 + request_index
    return TalkerInputPlan(
        text_token_ids=torch.arange(length),
        codec_token_ids=torch.zeros(length, dtype=torch.long),
        codec_token_mask=torch.zeros(length, dtype=torch.bool),
        sequence_lengths=(length,),
        continuations=(continuation,),
    )


def _request(index: int) -> AdmittedRequest:
    synthesis = SynthesisRequest(
        text=f"request {index}",
        voice="aiden",
        language="english",
        non_streaming_mode=index % 2 == 0,
        random_seed=100 + index,
        do_sample=False,
        max_new_tokens=2,
        skip_fixed_bootstrap_audio=False,
        stream_chunk_schedule=(1,),
    )
    plan = _input_plan(index, streaming=not synthesis.non_streaming_mode)
    continuation = plan.continuations[0]
    return AdmittedRequest(
        request_id=f"request-{index}",
        request=synthesis,
        talker_input=TalkerPrompt(
            text_token_ids=plan.text_token_ids,
            codec_token_ids=plan.codec_token_ids,
            codec_token_mask=plan.codec_token_mask,
            sequence_length=plan.sequence_lengths[0],
                continuation=continuation,
        ),
        chunk_schedule=(1,),
        suppress_first_silent_frame=False,
        admitted_at_s=float(index),
    )


def _runtime() -> tuple[SynthesisPipeline, _FakeExecution]:
    execution = _FakeExecution([])
    config = ProfileLoader().load_profile(ExecutionProfile.BALANCED)
    runtime = SynthesisPipeline(
        executor=execution,
        capture_catalog=CaptureCatalog.from_config(config.stages),
        policy_config=config.policy,
        model_config=SynthesisModelSpec(
            codec_eos_token_id=31,
            talker_vocab_size=32,
            num_codebooks=16,
            samples_per_frame=2,
            sample_rate=2,
        ),
        max_in_flight_rows=(
            config.stages.talker_decode.max_batch_size
            + config.stages.codec.max_batch_size
        ),
    )
    return runtime, execution


def _direct_runtime() -> tuple[SynthesisPipeline, _FakeExecution]:
    execution = _FakeExecution([])
    config = ProfileLoader().load_profile(ExecutionProfile.BALANCED)
    runtime = SynthesisPipeline(
        executor=execution,
        capture_catalog=CaptureCatalog.from_config(config.stages),
        policy_config=config.policy,
        model_config=SynthesisModelSpec(
            codec_eos_token_id=31,
            talker_vocab_size=32,
            num_codebooks=16,
            samples_per_frame=2,
            sample_rate=2,
        ),
        max_in_flight_rows=(
            config.stages.talker_decode.max_batch_size
            + config.stages.codec.max_batch_size
        ),
    )
    return runtime, execution


def test_production_execution_port_consumes_request_rows_without_aggregate_gpu_inputs() -> None:
    runtime, execution = _direct_runtime()
    runtime.admit(_request(0))
    runtime.admit(_request(1))
    source_text = runtime.request("request-0").input.talker_input.text_token_ids

    runtime.step(now_s=0.0)
    prefill = execution.calls[-1][2]
    assert isinstance(prefill, TalkerPrefillRowsExecutionInput)
    assert prefill.rows[0].text_token_ids is source_text
    assert [row.sampling.seed for row in prefill.rows] == [100, 101]

    runtime.step(now_s=0.1)
    code_predictor = execution.calls[-1][2]
    assert isinstance(code_predictor, CodePredictorRowsExecutionInput)
    assert code_predictor.rows[0].offsets == tuple(range(32, 512, 32))
    assert code_predictor.sampler_route is CodePredictorSamplerRoute.FUSED

    runtime.step(now_s=0.2)
    codec = execution.calls[-1][2]
    assert isinstance(codec, CodecRowsExecutionInput)
    assert all(len(row.frames) == 1 for row in codec.rows)

    runtime.step(now_s=0.3)
    decode = execution.calls[-1][2]
    assert isinstance(decode, TalkerDecodeRowsExecutionInput)
    assert [int(row.text_token_id) for row in decode.rows] == [99, 41]
    assert decode.reuse_attention_plan is False


def test_runtime_reuses_talker_decode_plan_only_for_all_streaming_rows() -> None:
    runtime, execution = _direct_runtime()
    runtime.admit(_request(1))
    runtime.admit(_request(3))

    runtime.step(now_s=0.0)
    runtime.step(now_s=0.1)
    runtime.step(now_s=0.2)
    runtime.step(now_s=0.3)

    decode = execution.calls[-1][2]
    assert isinstance(decode, TalkerDecodeRowsExecutionInput)
    assert decode.reuse_attention_plan is True


def test_runtime_bounds_cuda_submission_lookahead_to_one_speculative_step() -> None:
    from types import SimpleNamespace

    from nari_qwen3_tts.executor.cuda_graph import CudaSubmissionFence

    runtime, _execution = _direct_runtime()

    class Event:
        def __init__(self, *, ready: bool) -> None:
            self.ready = ready
            self.synchronizations = 0

        def query(self) -> bool:
            return self.ready

        def synchronize(self) -> None:
            self.synchronizations += 1
            self.ready = True

    oldest = Event(ready=False)
    newest = Event(ready=False)
    runtime._submissions.record(
        decision_id=1,
        submissions=(
            SimpleNamespace(
                completion_fence=CudaSubmissionFence.completed(),
                decision_fence=CudaSubmissionFence(oldest),
                requires_host_finalize=False,
            ),
        ),
    )
    runtime._submissions.record(
        decision_id=2,
        submissions=(
            SimpleNamespace(
                completion_fence=CudaSubmissionFence.completed(),
                decision_fence=CudaSubmissionFence(newest),
                requires_host_finalize=False,
            ),
        ),
    )

    runtime._throttle_cuda_submissions()

    assert oldest.synchronizations == 1
    assert runtime._submissions.decision_ids == (2,)


def test_runtime_splits_code_predictor_sampler_routes_before_execution() -> None:
    runtime, execution = _direct_runtime()
    general = _request(0)
    general = replace(
        general,
        request=replace(general.request, subtalker_top_k=0),
    )
    runtime.admit(general)
    runtime.admit(_request(1))
    runtime.step(now_s=0.0)

    snapshot = runtime.planner.candidates(runtime.state_store.requests, now_s=0.1)
    compatibilities = {
        work.request_id: work.compatibility
        for work in snapshot
        if work.stage is SynthesisStage.CODE_PREDICTOR
    }
    assert compatibilities["request-0"].sampler_route is CodePredictorSamplerRoute.GENERAL
    assert compatibilities["request-1"].sampler_route is CodePredictorSamplerRoute.FUSED

    observed_decision = runtime.planner.plan(
        snapshot,
        now_s=0.1,
        decision_id=runtime._next_decision_id(),
        available_rows=runtime.state_store.available_in_flight_rows,
    )
    waits = runtime.planner.observation(observed_decision).wait_reasons
    assert waits[0].request_id == "request-1"
    assert waits[0].reason.value == WaitReasonCode.INCOMPATIBLE_COHORT.value
    runtime.planner.discarded(observed_decision)

    step = runtime.step(now_s=0.1)

    assert step.decision.selected_request_ids == ("request-0",)
    call = execution.calls[-1][2]
    assert isinstance(call, CodePredictorRowsExecutionInput)
    assert call.sampler_route is CodePredictorSamplerRoute.GENERAL


def test_runtime_completion_keeps_executor_owned_output_storage_without_a_second_clone() -> None:
    runtime, _execution = _runtime()
    runtime.admit(_request(0))
    candidates = runtime.planner.candidates(runtime.state_store.requests, now_s=0.0)
    decision = runtime.planner.plan(
        candidates,
        now_s=0.0,
        decision_id=runtime._next_decision_id(),
        available_rows=runtime.state_store.available_in_flight_rows,
    )
    claim = runtime.committer.claim(decision)
    runtime.planner.committed(decision)
    batch = runtime.committer.batches(claim)[0]
    output = TalkerExecutionResult(
        result=TalkerResult(
            tokens=torch.tensor([4]),
            last_hidden=torch.arange(4, dtype=torch.float32).reshape(1, 4),
            logits=torch.arange(32, dtype=torch.float32).reshape(1, 32),
        ),
        next_seen_token_masks=torch.zeros((1, 32), dtype=torch.bool),
        next_sampling_offsets=torch.tensor([512]),
        kv_publications=(_Publication("request-0"),),
    )

    envelope = runtime._talker_completion(
        batch,
        output,
        (runtime.request("request-0"),),
        decode=False,
    )
    delta = envelope.completion.rows[0].delta

    assert delta.token.untyped_storage().data_ptr() == output.result.tokens.untyped_storage().data_ptr()
    assert delta.hidden.untyped_storage().data_ptr() == output.result.last_hidden.untyped_storage().data_ptr()
    assert delta.logits.untyped_storage().data_ptr() == output.result.logits.untyped_storage().data_ptr()
    assert (
        delta.next_seen_token_mask.untyped_storage().data_ptr()
        == output.next_seen_token_masks.untyped_storage().data_ptr()
    )


def test_integrated_rr_rebuilds_membership_and_preserves_two_lane_readiness() -> None:
    runtime, execution = _runtime()
    runtime.admit(_request(0))
    runtime.admit(_request(1))

    prefill = runtime.step(now_s=0.0)
    assert prefill.decision.selected_stage is SynthesisStage.TALKER_PREFILL
    assert prefill.decision.selected_request_ids == ("request-0", "request-1")
    prefill_call = execution.calls[-1][2]
    assert tuple(row.text_token_ids.numel() for row in prefill_call.rows) == (10, 11)
    assert [row.sampling.seed for row in prefill_call.rows] == [100, 101]

    code_predictor = runtime.step(now_s=0.1)
    assert code_predictor.decision.selected_stage is SynthesisStage.CODE_PREDICTOR
    cp_call = execution.calls[-1][2]
    assert [row.seed for row in cp_call.rows] == [100, 101]
    assert list(cp_call.rows[0].offsets) == list(range(32, 512, 32))

    snapshot = runtime.planner.candidates(runtime.state_store.requests, now_s=0.2)
    assert {(work.request_id, work.stage) for work in snapshot} == {
        ("request-0", SynthesisStage.TALKER_DECODE),
        ("request-1", SynthesisStage.TALKER_DECODE),
        ("request-0", SynthesisStage.CODEC),
        ("request-1", SynthesisStage.CODEC),
    }
    assert {
        work.reserve_s
        for work in snapshot
        if work.stage is SynthesisStage.CODEC
    } == {0.010}

    codec = runtime.step(now_s=0.2)
    assert codec.decision.selected_stage is SynthesisStage.CODEC
    assert codec.committed_request_ids == ("request-0", "request-1")
    assert all(
        runtime.request(request_id).codec.visible_pcm_frames > 0
        for request_id in codec.committed_request_ids
    )
    assert all(
        runtime.request(request_id).codec.playback_started_at_s is None
        for request_id in codec.committed_request_ids
    )
    assert all(
        runtime.request(request_id).codec.emitted_duration_s == 0.0
        for request_id in codec.committed_request_ids
    )

    decode = runtime.step(now_s=0.3)
    assert decode.decision.selected_stage is SynthesisStage.TALKER_DECODE
    decode_call = execution.calls[-1][2]
    assert [int(row.text_token_id) for row in decode_call.rows] == [99, 41]
    assert [row.suppress_eos for row in decode_call.rows] == [True, True]
    assert [row.sampling.seed for row in decode_call.rows] == [100, 101]

    runtime.mark_pcm_routed("request-0", pcm_bytes=4, routed_at_s=0.35)
    runtime.mark_pcm_routed("request-0", pcm_bytes=8, routed_at_s=0.40)
    routed = runtime.request("request-0").codec
    assert routed.playback_started_at_s == 0.35
    assert routed.emitted_duration_s == 3.0


def test_ignore_eos_suppresses_the_eos_token_for_every_decode_step() -> None:
    runtime, _execution = _runtime()
    request = _request(0)
    runtime.admit(
        replace(
            request,
            request=replace(
                request.request,
                ignore_eos=True,
                max_new_tokens=360,
            ),
        )
    )
    state = runtime.request("request-0")
    state.generation.generation_step = 300

    assert runtime.input_builder.suppress_decode_eos(state)


def test_talker_commit_advances_the_known_rng_address_without_a_device_scalar_read() -> None:
    class DeviceOffsets:
        def __getitem__(self, index):
            del index
            raise AssertionError("runtime must not synchronize a deterministic RNG offset")

    runtime, execution = _runtime()
    original_prefill = execution.talker_prefill_rows

    def prefill_without_readable_offsets(key, call):
        output = original_prefill(key, call)
        return TalkerExecutionResult(
            result=output.result,
            next_seen_token_masks=output.next_seen_token_masks,
            next_sampling_offsets=DeviceOffsets(),
            kv_publications=output.kv_publications,
        )

    execution.talker_prefill_rows = prefill_without_readable_offsets
    runtime.admit(_request(0))

    runtime.step(now_s=0.0)

    assert runtime.request("request-0").generation.next_sampling_offset == 512


def test_talker_eos_completion_allows_other_ready_work_to_launch_before_scalar_read() -> None:
    class Event:
        def __init__(self) -> None:
            self.ready = False
            self.synchronize_calls = 0

        def query(self) -> bool:
            return self.ready

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            self.ready = True

    events = []

    def event_factory():
        event = Event()
        events.append(event)
        return event

    runtime, execution = _runtime()
    runtime.completion_event_factory = event_factory
    request = _request(0)
    runtime.admit(replace(request, request=replace(request.request, max_new_tokens=4)))

    while not (
        runtime.request("request-0").generation.generation_step == 2
        and runtime.request("request-0").generation.phase.value == "talker_decode"
    ):
        runtime.step(now_s=0.0)

    first_codec = runtime.step(now_s=0.0)
    assert first_codec.decision.selected_stage is SynthesisStage.CODEC

    deferred = runtime.step(now_s=0.0)

    assert deferred.decision.selected_stage is SynthesisStage.TALKER_DECODE
    assert deferred.completions == ()
    assert runtime.request("request-0").generation.claim_token is not None
    assert len(events) == 1

    runtime.admit(_request(1))
    overlapped = runtime.step(now_s=0.0)

    assert overlapped is not None
    assert overlapped.decision.selected_stage is SynthesisStage.TALKER_PREFILL
    assert events[0].synchronize_calls == 0
    events[0].ready = True
    runtime.step(now_s=0.0)
    assert events[0].synchronize_calls == 0
    assert runtime.request("request-0").generation.claim_token is None
    assert any(call[0] == "code_predictor_rows" for call in execution.calls)


def test_executor_failure_releases_plan_without_advancing_request_state() -> None:
    runtime, execution = _runtime()
    runtime.admit(_request(0))

    def fail(key, call):
        del key, call
        raise RuntimeError("captured replay failed")

    execution.talker_prefill_rows = fail
    before = runtime.request("request-0").committed_view()
    try:
        runtime.step(now_s=0.0)
    except RuntimeError as error:
        assert str(error) == "captured replay failed"
    else:
        raise AssertionError("replay failure was swallowed")

    state = runtime.request("request-0")
    assert state.committed_view() == before
    assert state.generation.claim_token is None


def test_admission_resource_failure_does_not_publish_partial_request_state() -> None:
    runtime, execution = _runtime()

    def fail_add(request_id):
        raise RuntimeError(f"no KV ownership for {request_id}")

    execution.add_request = fail_add

    try:
        runtime.admit(_request(0))
    except RuntimeError as error:
        assert str(error) == "no KV ownership for request-0"
    else:
        raise AssertionError("admission resource failure was swallowed")

    assert runtime.state_store.requests == ()


def test_runtime_cancellation_tombstones_inflight_output_without_publication() -> None:
    runtime, execution = _runtime()
    runtime.admit(_request(0))
    candidates = runtime.planner.candidates(runtime.state_store.requests, now_s=0.0)
    decision = runtime.planner.plan(
        candidates,
        now_s=0.0,
        decision_id=runtime._next_decision_id(),
        available_rows=runtime.state_store.available_in_flight_rows,
    )
    claim = runtime.committer.claim(decision)
    runtime.planner.committed(decision)
    batch = runtime.committer.batches(claim)[0]
    states = (runtime.request("request-0"),)
    inputs = runtime.input_builder.build(decision, runtime.state_store)
    submission = execution.submit(decision, inputs)[0]
    envelope = runtime._submission_completion(batch, states, submission.result)
    publication = envelope.kv_publications[0]
    runtime.cancel("request-0")

    runtime._publish_completion(batch, claim, envelope)

    state = runtime.request("request-0")
    assert publication.discarded
    assert not publication.published
    assert state.version == 0
    assert state.is_removable
