# Vast.ai clean-rebuild audit — instance 46693892

This record validates the remediated Agile SONIC `training` image on a new
Vast.ai container and a new 400 GB disk. It compares the result with discovery
instance `46687432`; no old container filesystem, dependency overlay, source,
Kit cache or checkpoint was reused.

## Scope

- Date: 2026-08-03
- Vast instance: `46693892`
- Machine / host: `105856` / `289275`
- GPU: 1× RTX PRO 6000 WS, 97,887 MiB reported VRAM
- CPU allocation: 128 effective vCPUs
- Disk: new 400 GB allocation
- Host driver: `595.58.03`; marketplace CUDA maximum: `13.2`
- Image commit: `efd9c83465a0448dec855ad4c9088ac19a80cac5`
- Image digest:
  `sha256:6ab2d3aac46c74bb97cae40611262c16163ef1b3aae9cf503ce7e97e2f6b7b59`
- Private SONIC revision:
  `6eb18262cc2c3c707630c401ba3e8349a2579df9`
- Limitation: one GPU validates the complete one-device training path, but not
  multi-GPU NCCL collectives, DDP or inter-GPU P2P.

The old instance and its 400 GB disk were destroyed before this allocation was
created. Vast returned no instance for `46687432` afterward.

## Build and publication

GitHub Actions run
[`30804335746`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/actions/runs/30804335746)
completed successfully.

- Validation job: passed, including shell, Compose, Python and Docker target
  checks.
- Training publish job: 22 minutes 57 seconds.
- Sampled cold-build disk consumption: 56,978,497,536 bytes.
- Minimum remaining runner space: 55,222,448,128 bytes.
- Largest compressed layer: 4,102,248,768 bytes; GHCR margin check passed.
- Candidate promotion and provenance attestation passed.
- Attestation:
  [`38546486`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/attestations/38546486).

The only workflow warnings were GitHub's automatic Node 20-to-24 transition and
the existing advisory about using a larger runner for non-training image
targets. Neither affected this build.

## Clean creation and cold pull

The instance was created from the digest, not a mutable tag. Vast progressed
through layer download, checksum verification and extraction without registry
errors. The complete image took approximately six minutes from instance
creation to `running`/SSH availability on this host. This is a cold-deployment
latency cost of the complete image and should be included in server handoff
planning.

Vast direct-SSH mode again replaced the native image entrypoint. The on-start
hook generated host keys, copied Vast's injected authorization to the image's
non-root user and started sshd without enabling root login.

## Runtime identity and permissions

- Login user: `fangzhengtian`, UID/GID 1000.
- Supplementary group: `video`.
- Authentication: public key only; there is no default password.
- Root SSH and password SSH remain disabled.
- The runtime user has no unrestricted sudo access.
- `/workspace`, `/datasets`, `/runs` and `/cache` were writable as intended.
- `/dev/shm` was 125 GB.
- Vast did not execute the native entrypoint, so
  `/run/agile-sonic/runtime.env` was absent before and after restart.

The SSH session began with soft limits `nofile=1024`, `stack=8192 KiB` and
`memlock=8192 KiB`. It could raise `nofile` to 1,048,576 and stack to 65,536
KiB, but not memlock. Production multi-GPU deployment must set unlimited
memlock at the container runtime/template level.

## Image remediation verification

All three fixes motivated by the first audit were present before source upload:

1. `h5py==3.13.0` imported from the immutable `agile-sonic` environment. No
   `/cache/audit-deps`, `PYTHONPATH` overlay or server-side `pip install` was
   used.
2. `SONIC_EXTRA_KIT_ARGS` defaulted to
   `--portable-root /cache/isaac-portable` despite Vast bypassing entrypoint.
3. Non-root paths resolved to persistent/writable locations:
   - `~/.nv/ComputeCache` → `/cache/nvidia/ComputeCache`
   - `~/.local/share/ov/data` → `/cache/ov-data`
   - `~/.nvidia-omniverse/logs` → `/runs/nvidia-omniverse`

No package-local read-only or permission-denied message appeared in the real
training log. The first launch downloaded about 190 MB of Kit extensions into
`/cache/ov-data`, confirming that the unavoidable first-run download now lands
in the intended cache and survives restart.

## Driver and CUDA execution

- Actual GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.
- Compute capability: 12.0.
- Host driver: `595.58.03`.
- Image runtime: PyTorch `2.7.0+cu128`; its architecture list includes
  `sm_120` and `compute_120`.
- A real 2048×2048 BF16 CUDA matrix operation, loss and backward pass completed
  with finite gradients.
- Isaac selected the GPU through Vulkan and created the physics environments.

There is no evidence that this host driver needs replacement for this image.
A different production driver must still pass the same runtime gates.

## Source and fixture deployment

The user explicitly authorized transfer to instance `46693892`. Rsync uploaded
671 regular files totaling 238,929,129 bytes before compression. The filter
excluded Git history, complete datasets, checkpoints, logs, virtualenvs,
generated package metadata, unrelated assets and non-smoke WBC trees.

The paired fixture hashes matched the checked-in manifest:

```text
robot  9a6df6b558219c9b76f5cf454f5e4175461cdba0947a58bdedf8e4bd82cc625d
soma   f2397a1fce89b0c06d89e80f95a8bbb2d6a917b0bb5b73b9fed912e0987c2dcc
```

No `*.egg-info` or `*.dist-info` directory was transferred, so the generated
metadata/NumPy conflict found in the first audit did not recur.

## Preflight result

The image's base training preflight passed:

- source, dataset, cache and run paths;
- atomic run-directory write/rename;
- `pip check` and required training imports, including Isaac and SONIC;
- CUDA device count and properties;
- NCCL backend availability.

Running the private project's broader `--project-check` also checked TensorRT,
CUDA Python, the complete default motion dataset and excluded teleop/WBC
scripts. Those checks failed because they belong to the `tensorrt/full` image
and full source/data deployment, not this `training` smoke bundle. This is a
profile-design issue in the project checker, not a failure of the exercised
training path. The checker should expose separate `training`, `tensorrt` and
`full` profiles to avoid misleading failures.

## Real H20 training smoke

Status: **passed**

Run directory:
`/runs/sonic-h20-smoke-clean-46693892-attempt1/training`.

- Isaac created 24 real H20 environments on `cuda:0`.
- The H20 MJCF and paired robot/SOMA motion were loaded.
- Two eight-step PPO rollouts completed: real simulated transitions advanced
  from 192 to 384.
- Two optimizer updates and two 107 MB checkpoints completed.
- The verifier found finite, non-zero changes in every required group:
  - H20 encoder: 8 tensors, max delta `0.000112263`;
  - teleop encoder: 10 tensors, max delta `0.000112552`;
  - SOMA encoder: 10 tensors, max delta `0.000112435`;
  - H20 dynamics decoder: 8 tensors, max delta `0.000112652`;
  - H20 kinematics decoder: 8 tensors, max delta `0.000111509`;
  - critic: 8 tensors, max delta `0.000112638`.
- The optimizer checkpoint contained 53 state slots.
- Final result: `SONIC H20 REAL TRAINING SMOKE: PASS`.

No fatal traceback, missing module, permission error, CUDA/NCCL error, OOM or
segmentation fault was found. After exit the GPU returned to 1 MiB, 0% usage,
P8, with no training/Kit process left behind.

Checkpoint SHA-256 values:

```text
f0cb0e2061ba04b5cc44e41a3427187c8f6a4d2ab6081a6d54c41181e41f37b1  model_step_000001.pt
a27801b79a5084eeb86dd4185076cebbbbadecf8aec6f48d1da7e9d08e5193a5  model_step_000002.pt
```

## Stop/start persistence

The instance was stopped to `actual_status=exited`, restarted, and accessed
again through the non-root account. Source, fixture, 190 MB Kit cache and both
checkpoints remained present. Checkpoint hashes were unchanged and the GPU
again reported driver `595.58.03`. The native runtime-state file remained
absent, confirming that the Vast entrypoint caveat is stable rather than an
intermittent image problem.

The instance was stopped after validation to end active GPU charges. Its 400 GB
disk remains allocated for inspection and continues to incur storage charges
until the instance is destroyed.

## Before production multi-GPU training

The image is suitable for the next server acceptance stage, but this one-GPU
audit cannot certify multi-GPU behavior. The RTX PRO 6000 server still needs:

1. host Docker Engine and NVIDIA Container Toolkit validation;
2. unlimited memlock, adequate nofile/stack limits and large shared memory;
3. all intended GPUs visible inside the same container;
4. `agile-sonic-nccl-smoke --gpu-count <GPU_COUNT>` plus topology/P2P review;
5. a two-update DDP smoke with checkpoint save and same-topology resume;
6. a 15–30 minute all-GPU stability/load test;
7. durable mounts and checkpoint/object-storage synchronization.

Do not infer multi-GPU readiness from this single-GPU pass alone.
