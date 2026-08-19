ARG CUDA_VERSION=13.0.2

FROM ghcr.io/astral-sh/uv:0.12.2 AS uv
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.12.13

LABEL org.opencontainers.image.title="Nari Qwen3-TTS" \
      org.opencontainers.image.description="CUDA Graph-only Qwen3-TTS HTTP/WebSocket server" \
      org.opencontainers.image.source="https://github.com/nari-labs/nari-qwen3-tts"

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/nari-qwen3-tts/.venv \
    PATH=/opt/nari-qwen3-tts/.venv/bin:${PATH} \
    HOME=/home/nari \
    HF_HOME=/home/nari/.cache/huggingface \
    XDG_CACHE_HOME=/home/nari/.cache \
    TORCHINDUCTOR_CACHE_DIR=/home/nari/.cache/torchinductor \
    TRITON_CACHE_DIR=/home/nari/.cache/triton \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 2000 --gid 0 --create-home --home-dir /home/nari --shell /bin/bash nari \
    && mkdir -p /opt/nari-qwen3-tts /home/nari/.cache \
    && chown -R 2000:0 /opt/nari-qwen3-tts /home/nari \
    && chmod -R g+rwX /opt/nari-qwen3-tts /home/nari

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /opt/nari-qwen3-tts
COPY pyproject.toml uv.lock .python-version ./
RUN uv python install "${PYTHON_VERSION}" \
    && uv sync --frozen --no-dev --no-install-project \
        --extra codec --extra cuda --extra serving

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable \
        --extra codec --extra cuda --extra serving \
    && mkdir -p "${HF_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
    && chown -R 2000:0 /home/nari \
    && chmod -R g+rwX /home/nari

# qwen-tts probes the SoX executable at import time. Keep it in the runtime
# image so an otherwise healthy production startup is not degraded or noisy.
RUN apt-get update \
    && apt-get install -y --no-install-recommends sox \
    && rm -rf /var/lib/apt/lists/*

# Keep volatile build provenance after the expensive dependency layers so a
# new Git revision does not force a complete CUDA environment rebuild.
ARG VCS_REF=unknown
LABEL org.opencontainers.image.revision="${VCS_REF}"

COPY docker/entrypoint.sh /usr/local/bin/qwen3-tts-entrypoint
RUN chmod 0755 /usr/local/bin/qwen3-tts-entrypoint

USER nari
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=600s --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:8000/health || exit 1
ENTRYPOINT ["/usr/local/bin/qwen3-tts-entrypoint"]
