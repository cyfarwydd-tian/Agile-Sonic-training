# Multi-GPU training strategy

## Scope

The supported first production topology is one Linux host with multiple NVIDIA
GPUs, one training process per GPU, PyTorch DistributedDataParallel (DDP), NCCL
collectives and Hugging Face Accelerate as the launcher. H20 is the Agile robot
model; the validated training devices are RTX PRO 6000 Blackwell GPUs.

Multi-node training, RDMA, DeepSpeed and FSDP are outside the current validated
scope. They require separate network, sharded-checkpoint and recovery tests.

## Runtime topology

```text
one host
  -> one container with all selected GPUs
  -> one Accelerate process per GPU
  -> one Isaac/SONIC environment shard per process
  -> NCCL gradient synchronization
  -> shared read-only datasets
  -> shared persistent /runs and /cache mounts
```

NVLink is not required. PCIe peer-to-peer is acceptable when topology and NCCL
stress tests pass. Do not disable P2P or shared-memory transports as a permanent
performance workaround; investigate the driver, NCCL runtime and topology when
the normal paths fail.

The SONIC `num_envs` setting is per rank. The global environment count is
`num_envs × GPU_COUNT`, so scale it deliberately instead of copying a
single-GPU value unchanged. Start with a small per-rank count for acceptance,
then increase it while observing VRAM, host RAM, simulator throughput and
checkpoint latency.

## Pinned software baseline

- Immutable image:
  `ghcr.io/cyfarwydd-tian/agile-sonic-training@sha256:e0e0a1b7f70983ce76cd65e3b0493f641fef445ea512aefcdf64ee6123076800`
- CUDA user space: 12.8.1.
- PyTorch: 2.7.0+cu128 with Blackwell `sm_120` support.
- Effective training NCCL runtime: 2.26.5.
- Verified host driver: 595.58.03. This is a tested combination, not an exact
  requirement; a different compatible driver must pass all gates below.

Pin the private SONIC checkout to a full commit SHA. Do not start a production
job from a dirty source tree or a mutable image tag.

## Host and container requirements

Before launching the container, verify:

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi topo -p2p p
nvidia-ctk --version
```

Use `scripts/run.sh` or `compose.yaml`. They provide the runtime settings that a
minimal `docker run --gpus all` command omits:

- host IPC and sufficiently large `/dev/shm`;
- `IPC_LOCK` and unlimited memlock;
- high file-descriptor and stack limits;
- explicit GPU visibility and worker count;
- persistent source, dataset, run and cache mounts;
- host UID/GID mapping and bounded container logs.

Example:

```bash
scripts/run.sh \
  --image ghcr.io/cyfarwydd-tian/agile-sonic-training@sha256:e0e0a1b7f70983ce76cd65e3b0493f641fef445ea512aefcdf64ee6123076800 \
  --source /srv/agile-sonic/source \
  --datasets /srv/agile-sonic/datasets \
  --runs /srv/agile-sonic/runs \
  --cache /srv/agile-sonic/cache \
  --gpu-request all \
  --gpu-count <GPU_COUNT> \
  --ssh-port 0 \
  --detach
```

For a subset, pass matching values such as `--gpu-request 0,1 --gpu-count 2`.
The launcher rejects mismatched visibility and process counts.

## Acceptance sequence

Run every gate from the non-root training user.

1. Validate the immutable environment, mounts, source revision, limits and all
   visible GPUs:

   ```bash
   SONIC_EXPECTED_REVISION=<40-character-commit-sha> \
   SONIC_REQUIRE_CLEAN=1 \
   agile-sonic-preflight --strict --gpu-count <GPU_COUNT>
   ```

2. Run the normal collective gate, followed by the larger stress gate used for
   the validated two-GPU baseline:

   ```bash
   agile-sonic-nccl-smoke --gpu-count <GPU_COUNT>
   agile-sonic-nccl-smoke \
     --gpu-count <GPU_COUNT> \
     --size-mb 1024 \
     --iterations 50
   ```

3. Run the self-contained one-GPU H20 smoke to verify real Isaac rollouts, PPO,
   auxiliary losses and every active model group. This complements NCCL; it
   does not replace the multi-rank gate.

4. Launch a two-update DDP job with a small per-rank environment count through
   `agile-sonic-launch`. Require both ranks to complete, verify finite non-zero
   changes in the H20/teleop/SOMA encoders, H20 dynamic/kinematic decoders,
   critic and optimizer, and scan logs for CUDA, NCCL, OOM and simulator errors.

   ```bash
   RUN_ID=h20-ddp-acceptance \
   SONIC_EXPECTED_REVISION=<40-character-commit-sha> \
   SONIC_REQUIRE_CLEAN=1 \
   agile-sonic-launch --gpu-count <GPU_COUNT> -- \
     +exp=manager/universal_token/all_modes/sonic_release_h20_smoke \
     ++num_envs=<SMALL_ENVS_PER_RANK> \
     ++algo.config.num_learning_iterations=2
   ```

   Use the exact Hydra keys from the pinned SONIC experiment if they differ;
   never assume an override was accepted without inspecting the resolved
   configuration saved beside the checkpoint.

5. Resume on the same topology and complete at least one more update. The
   current structured configuration requires `+resume=true`, not
   `resume=true`. Confirm that transitions and optimizer state continue from
   the prior checkpoint rather than restarting.

6. Complete a 15–30 minute all-GPU stability run before scaling environment
   count or job duration. Monitor GPU utilization/temperature, VRAM, host RAM,
   disk throughput, NCCL errors, simulator resets, NaN/OOM and checkpoint time.

Only after all six gates pass should the server start the full run.

## Production launch

Use a unique `RUN_ID` for each experiment and keep all outputs below `/runs`.
The launcher creates a rank-consistent rendezvous on port 29500 by default and
sets BF16 mixed precision, asynchronous NCCL error handling and a unique W&B
directory. Override the port when another local job already uses it.

```bash
RUN_ID=<immutable-experiment-id> \
WANDB_MODE=offline \
agile-sonic-launch \
  --gpu-count <GPU_COUNT> \
  --mixed-precision bf16 \
  --main-port <FREE_LOCAL_PORT> \
  -- <PINNED_HYDRA_ARGUMENTS>
```

Persist checkpoints outside the container writable layer. For interruptible
hosts, continuously copy `/runs` to durable storage. Resume first on the same
GPU count; changing world size is a separate acceptance case.

## Failure policy

- Stop immediately on NCCL timeout, illegal memory access, rank divergence,
  non-finite loss/weights, repeated writer failure or corrupt checkpoint.
- Do not accept `NCCL_P2P_DISABLE=1` or `NCCL_SHM_DISABLE=1` as the production
  fix for a local multi-GPU failure.
- A successful import test or one-GPU PPO run does not certify DDP.
- A successful NCCL all-reduce does not certify the Isaac/SONIC training graph.
- Do not move to multi-node, FSDP or DeepSpeed until save/resume semantics and
  failure recovery are tested independently.
