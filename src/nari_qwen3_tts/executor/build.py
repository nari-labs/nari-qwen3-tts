"""Concrete construction of the fixed Qwen3-TTS CUDA executors."""

from __future__ import annotations

import math

import torch

from nari_qwen3_tts.contract.stage import CudaGraphKey
from nari_qwen3_tts.executor.code_predictor import CodePredictorExecutor
from nari_qwen3_tts.executor.codec import CodecExecutor
from nari_qwen3_tts.executor.cuda_graph import CudaGraphPoolFence, TorchCaptureDriver
from nari_qwen3_tts.executor.executor import Executor
from nari_qwen3_tts.executor.optimizations import install_capture_optimizations
from nari_qwen3_tts.executor.talker import TalkerExecutor
from nari_qwen3_tts.executor.talker_kv import PagedTalkerKV
from nari_qwen3_tts.model.checkpoint import LoadedModelAssets
from nari_qwen3_tts.profile import ResolvedProfile


def build_cuda_execution(
    assets: LoadedModelAssets,
    *,
    config: ResolvedProfile,
    required_keys: frozenset[CudaGraphKey],
) -> Executor:
    """Build the complete typed CUDA execution surface without capturing it yet."""
    stages = config.stages
    resources = config.resources
    device = assets.device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production execution requires CUDA")
    model_config = assets.model_config
    talker_config = model_config.talker
    prefill = stages.talker_prefill
    largest_prefill_tokens = max(
        prefill.token_buckets[-1],
        prefill.exact_batch_sizes[-1] * prefill.exact_sequence_lengths[-1],
    )
    cache = PagedTalkerKV(
        num_layers=talker_config.num_hidden_layers,
        num_kv_heads=talker_config.num_key_value_heads,
        num_qo_heads=talker_config.num_attention_heads,
        head_dim=talker_config.head_dim,
        total_pages=resources.kv_pages,
        page_size=resources.kv_page_size,
        scratch_page_count=max(
            math.ceil(largest_prefill_tokens / resources.kv_page_size),
            stages.talker_decode.batch_sizes[-1],
        ),
        workspace_bytes=resources.workspace_bytes,
        device=device,
        dtype=assets.talker.get_input_embeddings().weight.dtype,
    )
    talker_pool = torch.cuda.graphs.graph_pool_handle()
    code_predictor_pool = torch.cuda.graphs.graph_pool_handle()
    codec_pool = torch.cuda.graphs.graph_pool_handle()
    generation_fence = CudaGraphPoolFence(device=device)
    codec_fence = CudaGraphPoolFence(device=device)
    model_dtype = assets.talker.get_input_embeddings().weight.dtype
    talker = TalkerExecutor(
        model=assets.talker,
        config=model_config,
        cache=cache,
        capture_slots=resources.talker_capture_slots,
        driver=TorchCaptureDriver(
            device=device,
            autocast_dtype=model_dtype,
            memory_pool=talker_pool,
        ),
        submission_fence=generation_fence,
    )
    optimizations = install_capture_optimizations(assets, require_talker_fp8=True)
    code_predictor = CodePredictorExecutor(
        model=assets.code_predictor,
        layer0_embedding=assets.code_predictor_layer0_embedding,
        config=model_config,
        max_batch_size=stages.code_predictor.max_batch_size,
        driver=TorchCaptureDriver(
            device=device,
            autocast_dtype=model_dtype,
            memory_pool=code_predictor_pool,
        ),
        submission_fence=generation_fence,
    )
    codec = CodecExecutor(
        model=assets.codec,
        num_code_groups=model_config.num_code_groups,
        cold_frame_sizes=stages.codec.frames.cold,
        device=device,
        driver=TorchCaptureDriver(
            device=device,
            autocast_dtype=None,
            memory_pool=codec_pool,
        ),
        submission_fence=codec_fence,
    )
    return Executor(
        config=config,
        required_keys=required_keys,
        talker=talker,
        code_predictor=code_predictor,
        codec=codec,
        optimizations=optimizations,
    )


__all__ = ["build_cuda_execution"]
