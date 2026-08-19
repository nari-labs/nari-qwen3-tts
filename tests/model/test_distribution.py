from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from conftest import PACKAGE_ROOT


def _copy_package_only(destination: Path) -> Path:
    copied = destination / "nari_qwen3_tts"
    shutil.copytree(
        PACKAGE_ROOT,
        copied,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "build",
            "dist",
            "*.egg-info",
        ),
    )
    return copied


def test_package_only_wheel_install_import_and_run_in_isolation(tmp_path) -> None:
    copied = _copy_package_only(tmp_path)
    wheel_directory = tmp_path / "wheel"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_directory)],
        cwd=copied,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_directory.glob("*.whl"))
    assert wheel.name.startswith("nari_qwen3_tts-")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert "nari_qwen3_tts/__init__.py" in names
    assert all(not name.startswith(("tests/", "benchmarks/")) for name in names)
    removed_paths = {
        "nari_qwen3_tts/contract/state.py",
        "nari_qwen3_tts/engine/core.py",
        "nari_qwen3_tts/model/domain.py",
        "nari_qwen3_tts/planner/codec.py",
    }
    removed_packages = (
        "nari_qwen3_tts/execution/",
        "nari_qwen3_tts/runtime/",
        "nari_qwen3_tts/serving/",
        "nari_qwen3_tts/stages/",
    )
    assert removed_paths.isdisjoint(names)
    assert all(not name.startswith(removed_packages) for name in names)

    environment = tmp_path / "clean-environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
        check=True,
        capture_output=True,
        text=True,
    )
    python = environment / "bin" / "python"
    clean_env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    probe = (
        "import json,nari_qwen3_tts;"
        "print(json.dumps({'version': nari_qwen3_tts.__version__}))"
    )
    subprocess.run(
        [str(python), "-I", "-c", probe],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (environment / "bin" / "nari-qwen3-tts").is_file()
    assert (environment / "bin" / "nari-qwen3-tts-server").is_file()
    result = subprocess.run(
        [str(python), "-I", "-m", "nari_qwen3_tts"],
        cwd=tmp_path,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"capabilities": ["model", "job_separation", "model_optimization", "scheduler", "server"]' in result.stdout
    assert '"serving_available": true' in result.stdout
