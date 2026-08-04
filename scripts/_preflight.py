#!/usr/bin/env python3
"""Fast, non-simulation checks for an Agile SONIC training container."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
import traceback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument(
        "--imports",
        default="torch,accelerate,numpy,h5py,hydra,omegaconf,tensordict,wandb,isaacsim,isaaclab,gear_sonic.train_agent_trl",
        help="Comma-separated Python modules that must import",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    def ok(message: str) -> None:
        print(f"[ OK ] {message}")

    def fail(message: str) -> None:
        failures.append(message)
        print(f"[FAIL] {message}")

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    expected_python = os.getenv("EXPECTED_PYTHON", "3.11")
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python == expected_python:
        ok(f"Python version is {expected_python}")
    else:
        fail(f"Python {expected_python} required; found {actual_python}")

    modules = [name.strip() for name in args.imports.split(",") if name.strip()]
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", None)
            ok(f"import {module_name}" + (f" ({version})" if version else ""))
        except Exception as exc:  # noqa: BLE001
            fail(f"import {module_name}: {exc}")
            traceback.print_exc(limit=1)

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch CUDA runtime: {torch.version.cuda}")
        expected_torch = os.getenv("EXPECTED_TORCH", "2.7.0")
        if torch.__version__.split("+", maxsplit=1)[0] == expected_torch:
            ok(f"PyTorch version is {expected_torch}")
        else:
            fail(f"PyTorch {expected_torch} required; found {torch.__version__}")
        expected_cuda = os.getenv("EXPECTED_CUDA", "12.8")
        if (torch.version.cuda or "").startswith(expected_cuda):
            ok(f"PyTorch uses CUDA {expected_cuda}.x")
        else:
            fail(
                f"PyTorch CUDA {expected_cuda}.x required; "
                f"found {torch.version.cuda or 'none'}"
            )

        if torch.cuda.is_available():
            ok("torch.cuda is available")
        else:
            fail("torch.cuda is unavailable; start the container with GPU access")

        visible = torch.cuda.device_count()
        if visible >= args.gpu_count:
            ok(f"{visible} CUDA device(s) visible; {args.gpu_count} requested")
        else:
            fail(f"{args.gpu_count} GPU(s) requested, but only {visible} visible")

        for index in range(visible):
            props = torch.cuda.get_device_properties(index)
            memory_gib = props.total_memory / (1024**3)
            print(
                f"  cuda:{index}: {props.name}, capability "
                f"{props.major}.{props.minor}, {memory_gib:.1f} GiB"
            )

        if torch.distributed.is_available() and torch.distributed.is_nccl_available():
            ok("torch.distributed NCCL backend is available")
        else:
            fail("torch.distributed NCCL backend is unavailable")
    except Exception as exc:  # noqa: BLE001
        fail(f"PyTorch/CUDA inspection failed: {exc}")
        traceback.print_exc(limit=1)

    if failures:
        print(f"\nPreflight failed with {len(failures)} error(s):")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
