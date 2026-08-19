# Nari Qwen3-TTS

## TL;DR

Nari Qwen3-TTS is a high-performance, single-H100 serving implementation of
[Qwen3-TTS 1.7B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).
It exposes streaming and non-streaming speech generation over HTTP, plus
WebSocket-based input streaming for incremental text input.

It achieves **10 requests per second (RPS)** and **sub-50 ms p95
time-to-first-audio (TTFA)** while maintaining real-time playback on a **single
NVIDIA H100 SXM**. Even at **20 RPS**, it sustains **sub-80 ms p95 TTFA**.

Below is a performance comparison with popular serving engines.

![p95 TTFA under load comparison](docs/assets/p95-ttfa-under-load.png)

> [!NOTE]
> **Methodology**
>
> Read our [blog post](https://nari-labs.com/blog/qwen3-tts-speed-cost-frontier/)
> for the full methodology. For details on the benchmark, see the
> [benchmark repository](https://github.com/nari-labs/benchmarks).

## Run with Docker

The container requires an NVIDIA H100, the NVIDIA Container Toolkit, and a driver compatible with CUDA 13.0. The published image supports Linux x86_64 (`linux/amd64`) H100 hosts. The engine has been tested only with English as the primary language.

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN \
  -e QWEN3_TTS_PROFILE=ttfa \
  -v nari-qwen3-tts-cache:/home/nari/.cache \
  ghcr.io/nari-labs/nari-qwen3-tts:latest
```

Model files and compiled kernels are cached in the named volume. Once model
loading and CUDA Graph capture finish, check readiness with:

```bash
curl --fail http://127.0.0.1:8000/ready
```

To build the image locally instead:

```bash
docker build \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t nari-qwen3-tts:local .
```

## Run with uv

Python and uv versions are pinned in `.python-version` and `pyproject.toml`.
The checked-in `uv.lock` defines the complete environment.

Install the required system packages (Debian/Ubuntu):

```bash
sudo apt-get install -y build-essential libsndfile1 sox
```

```bash
uv sync --frozen --extra codec --extra cuda --extra serving
uv run --frozen nari-qwen3-tts-server --profile ttfa
```

Add `--local-files-only` after the model is cached to prevent downloads at
startup. `--frozen` is intentional: an out-of-date lockfile fails instead of
silently resolving a different CUDA or Python environment.

The distribution name uses hyphens, while Python imports use underscores:

```python
from nari_qwen3_tts import ModelAssetConfig, open_model
```

## Profiles

- `ttfa`: prioritizes time to first audio with latency-oriented scheduling and
  smaller initial Codec chunks.
- `balanced`: the default container profile, balancing first-audio latency and
  sustained request throughput.
- `throughput`: uses larger Codec chunks and batches to prioritize aggregate
  throughput under load.

Select a Docker profile with `QWEN3_TTS_PROFILE`, or pass `--profile` to
`nari-qwen3-tts-server`. You can also set `QWEN3_TTS_MODEL_CACHE_DIR` and
`QWEN3_TTS_LOCAL_FILES_ONLY=1` in Docker.

For advanced tuning, apply a strict partial YAML overlay:

```bash
nari-qwen3-tts-server \
  --local-files-only \
  --engine-config /path/to/engine.yaml
```

The overlay may name its packaged base with `extends: ttfa`, `balanced`, or
`throughput`; alternatively, pass `--profile` and omit `extends`. Unknown keys,
invalid capture lists, and profile/base mismatches fail before model loading.
The fully resolved config and its SHA-256 are printed at startup.

## API and architecture

The service exposes:

- `GET /health`
- `GET /ready`
- `GET /v1/models`
- `POST /v1/audio/speech`
- `WS /v1/audio/speech/ws`

See [WebSocket speech API](docs/websocket.md) for the live-text protocol and
client example.

### HTTP client example

`POST /v1/audio/speech` follows the OpenAI Audio Speech request shape. Nari Qwen3-TTS
supports only the model listed above, `wav` and `pcm` output, and `speed: 1.0`.
It also accepts Nari-specific controls such as `language`.

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "Hello from Nari Labs.",
    "voice": "ryan",
    "language": "english",
    "response_format": "wav",
    "stream": false
  }' \
  --output speech.wav
```

Readiness remains false until CUDA Graph capture and a warm-up TTS request have
both completed.

## Thanks and references

Built by [Nari Labs](https://nari-labs.com). Thanks to the Qwen3-TTS authors
for releasing Qwen3-TTS, and to these projects for their work on
high-performance multimodal and speech serving:

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [SGLang-Omni](https://github.com/sgl-project/sglang-omni)
- [VoxServe](https://github.com/vox-serve/vox-serve)
- [M*](https://github.com/mstar-project/mstar)
