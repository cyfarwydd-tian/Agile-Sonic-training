#!/usr/bin/env python3
"""Prove that each required SONIC training module received a real update."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch

POLICY_GROUPS = {
    "H20 encoder": "actor_module.encoders.h20.",
    "teleop encoder": "actor_module.encoders.teleop.",
    "SOMA encoder": "actor_module.encoders.soma.",
    "H20 dynamics decoder": "actor_module.decoders.h20_dyn.",
    "H20 kinematics decoder": "actor_module.decoders.h20_kin.",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        fail(f"missing checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_finite_state(name: str, state: dict[str, torch.Tensor]) -> None:
    if not state:
        fail(f"{name} state dict is empty")
    for key, value in state.items():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            if not torch.isfinite(value).all():
                fail(f"{name} contains NaN/Inf: {key}")


def assert_group_changed(
    name: str,
    prefix: str,
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> tuple[int, float]:
    keys = [key for key in before if key.startswith(prefix) and key in after]
    if not keys:
        fail(f"{name} parameters not found with prefix {prefix!r}")

    changed = 0
    max_delta = 0.0
    for key in keys:
        old = before[key]
        new = after[key]
        if old.shape != new.shape:
            fail(f"shape changed unexpectedly for {key}: {old.shape} -> {new.shape}")
        if not torch.equal(old, new):
            changed += 1
            if old.is_floating_point():
                delta = float((new.float() - old.float()).abs().max())
                if not math.isfinite(delta):
                    fail(f"non-finite parameter delta in {key}")
                max_delta = max(max_delta, delta)
    if changed == 0:
        fail(f"{name} did not update between PPO steps 1 and 2")
    return changed, max_delta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    first = load_checkpoint(run_dir / "model_step_000001.pt")
    second = load_checkpoint(run_dir / "model_step_000002.pt")
    for required in ("policy_state_dict", "value_state_dict", "optimizer_state_dict", "state"):
        if required not in first or required not in second:
            fail(f"checkpoint is missing required key: {required}")

    first_step = getattr(first["state"], "global_step", None)
    second_step = getattr(second["state"], "global_step", None)
    if (first_step, second_step) != (1, 2):
        fail(f"expected checkpoint steps (1, 2), got {(first_step, second_step)}")
    first_transitions = getattr(first["state"], "tot_timesteps", None)
    second_transitions = getattr(second["state"], "tot_timesteps", None)
    if (first_transitions, second_transitions) != (192, 384):
        fail(f"expected real rollout totals (192, 384), got {(first_transitions, second_transitions)}")

    policy_before = first["policy_state_dict"]
    policy_after = second["policy_state_dict"]
    value_before = first["value_state_dict"]
    value_after = second["value_state_dict"]
    assert_finite_state("policy", policy_after)
    assert_finite_state("critic", value_after)

    results = {}
    for name, prefix in POLICY_GROUPS.items():
        results[name] = assert_group_changed(name, prefix, policy_before, policy_after)
    results["critic"] = assert_group_changed("critic", "critic_module.", value_before, value_after)

    optimizer = second["optimizer_state_dict"]
    if not optimizer or not optimizer.get("state") or not optimizer.get("param_groups"):
        fail("optimizer state is empty; no real optimizer update was recorded")
    env_state = second.get("env_state_dict")
    if not isinstance(env_state, dict) or "motion_lib" not in env_state:
        fail("checkpoint does not contain the motion-library environment state")

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        fail(f"resolved Hydra config was not saved: {config_path}")
    config = config_path.read_text(encoding="utf-8")
    for marker in (
        "h20_recon:",
        "h20_soma_latent:",
        "h20_teleop_latent:",
        "active_encoders:",
        "active_decoders:",
        "num_learning_iterations: 2",
        "num_steps_per_env: 8",
        "motion_file: /datasets/robot",
        "soma_motion_file: /datasets/soma",
    ):
        if marker not in config:
            fail(f"resolved config does not contain required training path: {marker}")

    print("SONIC H20 REAL TRAINING SMOKE: PASS")
    print(f"checkpoints: {first_step} -> {second_step}")
    print(f"real simulated transitions: {first_transitions} -> {second_transitions}")
    for name, (changed, max_delta) in results.items():
        print(f"  {name}: {changed} tensors changed, max |delta|={max_delta:.6g}")
    print(f"  optimizer slots: {len(optimizer['state'])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"SONIC H20 REAL TRAINING SMOKE: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
