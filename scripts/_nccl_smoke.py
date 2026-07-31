#!/usr/bin/env python3
"""A small all-reduce test for single-node NCCL validation."""

from __future__ import annotations

from datetime import timedelta
import os
import time

import torch
import torch.distributed as dist


def required_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    timeout_seconds = required_int("NCCL_SMOKE_TIMEOUT", 180)
    size_mb = required_int("NCCL_SMOKE_MB", 64)
    iterations = required_int("NCCL_SMOKE_ITERATIONS", 5)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=timeout_seconds),
    )

    value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(
            f"rank {rank}: all-reduce returned {value.item()}, expected {expected}"
        )

    elements = size_mb * 1024 * 1024 // torch.tensor([], dtype=torch.float32).element_size()
    payload = torch.ones(elements, dtype=torch.float32, device=f"cuda:{local_rank}")
    for _ in range(2):
        dist.all_reduce(payload)
        payload.fill_(1)
    torch.cuda.synchronize()
    dist.barrier()

    started = time.perf_counter()
    for _ in range(iterations):
        dist.all_reduce(payload)
        payload.fill_(1)
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - started

    if rank == 0:
        algorithm_gib = (size_mb / 1024) * iterations / elapsed
        print(
            f"NCCL smoke passed: {world_size} ranks, {size_mb} MiB, "
            f"{iterations} iterations, {elapsed:.3f}s, "
            f"{algorithm_gib:.2f} GiB/s rank-local payload rate"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
