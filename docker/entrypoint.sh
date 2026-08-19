#!/bin/sh
set -eu

if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    exec "$@"
fi

model="${QWEN3_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
profile="${QWEN3_TTS_PROFILE:-ttfa}"
host="${QWEN3_TTS_HOST:-0.0.0.0}"
port="${QWEN3_TTS_PORT:-8000}"
model_cache_dir="${QWEN3_TTS_MODEL_CACHE_DIR:-${HF_HOME:-/home/nari/.cache/huggingface}/hub}"

case "${QWEN3_TTS_LOCAL_FILES_ONLY:-0}" in
    1|true|TRUE|yes|YES)
        exec nari-qwen3-tts-server \
            --model "$model" \
            --model-cache-dir "$model_cache_dir" \
            --device cuda:0 \
            --profile "$profile" \
            --host "$host" \
            --port "$port" \
            --local-files-only \
            "$@"
        ;;
    0|false|FALSE|no|NO|'')
        exec nari-qwen3-tts-server \
            --model "$model" \
            --model-cache-dir "$model_cache_dir" \
            --device cuda:0 \
            --profile "$profile" \
            --host "$host" \
            --port "$port" \
            "$@"
        ;;
    *)
        printf 'invalid QWEN3_TTS_LOCAL_FILES_ONLY value: %s\n' "$QWEN3_TTS_LOCAL_FILES_ONLY" >&2
        exit 2
        ;;
esac
