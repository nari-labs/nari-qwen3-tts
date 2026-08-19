"""CUDA Graph-only execution for the fixed Qwen3-TTS stage vocabulary."""

from nari_qwen3_tts.executor.build import build_cuda_execution
from nari_qwen3_tts.executor.cache import (
    KVAllocationError,
    KVReservation,
    PagedKVAllocator,
    PendingKVPublication,
    RequestKVState,
    StaleKVPublicationError,
)
from nari_qwen3_tts.executor.code_predictor import CodePredictorExecutor
from nari_qwen3_tts.executor.codec import (
    CodecExecutor,
)
from nari_qwen3_tts.executor.cuda_graph import (
    CapturedCall,
    CaptureDriver,
    CudaGraphPoolFence,
    CudaSubmissionFence,
    SlotBusyError,
    SlotLease,
    SlotLeaseState,
    TorchCaptureDriver,
    stage_rows,
)
from nari_qwen3_tts.executor.executor import Executor
from nari_qwen3_tts.executor.health import (
    CaptureStartupError,
    UncapturedExecutionError,
)
from nari_qwen3_tts.executor.rng import TalkerCodebookAddress
from nari_qwen3_tts.executor.rows import (
    CodecExecutionRow,
    CodecMetadataExecutionRow,
    CodecRowsExecutionInput,
    CodePredictorExecutionRow,
    CodePredictorRowsExecutionInput,
    TalkerDecodeExecutionRow,
    TalkerDecodeRowsExecutionInput,
    TalkerExecutionResult,
    TalkerPrefillExecutionRow,
    TalkerPrefillRowsExecutionInput,
    TalkerSamplingExecutionRow,
)
from nari_qwen3_tts.executor.submission import (
    StageExecutionInput,
    StageExecutionInputs,
    StageExecutionResult,
    StageExecutionSubmission,
    SubmissionFenceError,
    SubmissionWindow,
)
from nari_qwen3_tts.executor.talker import TalkerExecutor, TalkerKVBackend
from nari_qwen3_tts.executor.talker_kv import (
    FlashInferDecodeBinding,
    FlashInferPrefillBinding,
    PagedTalkerKV,
    TalkerAttentionContext,
    TalkerAttentionMetadata,
)

__all__ = [
    "CaptureDriver",
    "CaptureStartupError",
    "CapturedCall",
    "CudaGraphPoolFence",
    "CudaSubmissionFence",
    "CodecExecutor",
    "CodecExecutionRow",
    "CodecMetadataExecutionRow",
    "CodecRowsExecutionInput",
    "CodePredictorExecutor",
    "CodePredictorExecutionRow",
    "CodePredictorRowsExecutionInput",
    "Executor",
    "FlashInferDecodeBinding",
    "FlashInferPrefillBinding",
    "KVAllocationError",
    "KVReservation",
    "PendingKVPublication",
    "PagedKVAllocator",
    "PagedTalkerKV",
    "RequestKVState",
    "SlotBusyError",
    "SlotLease",
    "StaleKVPublicationError",
    "SlotLeaseState",
    "StageExecutionInput",
    "StageExecutionInputs",
    "StageExecutionResult",
    "StageExecutionSubmission",
    "SubmissionFenceError",
    "SubmissionWindow",
    "TalkerCodebookAddress",
    "TalkerAttentionContext",
    "TalkerAttentionMetadata",
    "TalkerKVBackend",
    "TalkerExecutor",
    "TalkerDecodeExecutionRow",
    "TalkerDecodeRowsExecutionInput",
    "TalkerExecutionResult",
    "TalkerPrefillExecutionRow",
    "TalkerPrefillRowsExecutionInput",
    "TalkerSamplingExecutionRow",
    "TorchCaptureDriver",
    "UncapturedExecutionError",
    "build_cuda_execution",
    "stage_rows",
]
