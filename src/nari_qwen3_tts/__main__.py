"""Package capability summary entry point."""

from __future__ import annotations

import json

from nari_qwen3_tts import __version__


def main() -> None:
    print(
        json.dumps(
            {
                "package": "nari-qwen3-tts",
                "version": __version__,
                "capabilities": [
                    "model",
                    "job_separation",
                    "model_optimization",
                    "scheduler",
                    "server",
                ],
                "serving_available": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
