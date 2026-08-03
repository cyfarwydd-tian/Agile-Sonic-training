#!/usr/bin/env python3
"""Run two real SONIC PPO updates in the Agile training container."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from verify_training import main as verify_training


def run_streaming(command: list[str], log_path: Path) -> None:
    print("+", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"SONIC training exited with status {return_code}")


def main() -> int:
    source = Path("/workspace/sonic-training")
    entrypoint = source / "gear_sonic/train_agent_trl.py"
    fixture = Path("/datasets")
    if not entrypoint.is_file():
        raise RuntimeError(f"SONIC entrypoint not found: {entrypoint}")
    for required in (
        fixture / "robot/neutral_walk_180_R_002__A075.pkl",
        fixture / "soma/neutral_walk_180_R_002__A075.pkl",
    ):
        if not required.is_file():
            raise RuntimeError(f"smoke fixture is missing: {required}")

    run_dir = Path(os.environ.get("RUN_DIR", "/runs/sonic-h20-smoke")) / "training"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(entrypoint),
        "+exp=manager/universal_token/all_modes/sonic_release_h20_test",
        f"experiment_dir={run_dir}",
        "num_envs=24",
        "headless=true",
        "use_wandb=false",
        "manager_env.config.episode_length_s=2.0",
        "manager_env.config.render_results=false",
        "manager_env.commands.motion.num_future_frames=2",
        "manager_env.commands.motion.smpl_num_future_frames=2",
        f"manager_env.commands.motion.motion_lib_cfg.motion_file={fixture / 'robot'}",
        f"manager_env.commands.motion.motion_lib_cfg.soma_motion_file={fixture / 'soma'}",
        "manager_env.commands.motion.motion_lib_cfg.max_num_motions=1",
        "manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load=1",
        "algo.config.num_learning_iterations=2",
        "algo.config.num_steps_per_env=8",
        "algo.config.num_learning_epochs=1",
        "algo.config.num_mini_batches=1",
        "algo.config.save_interval=1",
        "algo.config.eval_frequency=1000000",
        "callbacks.model_save.save_frequency=1",
        "callbacks.model_save.save_last_frequency=1",
    ]
    os.chdir(source)
    run_streaming(command, run_dir / "console.log")

    old_argv = sys.argv
    try:
        sys.argv = ["verify_training.py", str(run_dir)]
        return verify_training()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
