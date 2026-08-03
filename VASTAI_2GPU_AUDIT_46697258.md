# Vast.ai two-GPU audit — instance 46697258

This record covers the first two-GPU validation of the Agile SONIC training
container. It is intentionally separate from the one-GPU real-training audit:
the goal here is to identify multi-GPU runtime blockers before deployment to
the production RTX PRO 6000 server.

## Test allocation

- Date: 2026-08-03
- Vast instance: `46697258`
- Offer / machine / host: `46009775` / `43966` / `25384`
- Location: Japan
- Cost at creation: USD 2.5778/hour including a new 400 GB disk
- GPUs: 2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- VRAM: 97,887 MiB per GPU
- Compute capability: 12.0
- CPU allocation: 64 vCPUs; system RAM approximately 257 GB
- Host driver: `595.58.03`; marketplace CUDA maximum: `13.2`
- Image under discovery test:
  `sha256:6ab2d3aac46c74bb97cae40611262c16163ef1b3aae9cf503ce7e97e2f6b7b59`
- NCCL fix image tested with the private workload:
  `sha256:d36a8b1efa37052911f4c0744db036a240bf38016b08c4ef45ee327440003143`
- NCCL fix commit: `5dc641d22a113a7a8a85fdf02c0138db008b189a`
- Runtime-user permission fix: `4655f4545db5b249349bed43a90cb7d7ba8f88ee`
- Final immutable image:
  `sha256:e0e0a1b7f70983ce76cd65e3b0493f641fef445ea512aefcdf64ee6123076800`

## Topology and NVLink

This exact host does **not** provide NVLink. `nvidia-smi nvlink --status`
reported `Device does not have or support Nvlink` for both GPUs.

`nvidia-smi topo -m` reported a `NODE` path between GPU 0 and GPU 1. Both GPUs
are attached to NUMA node 0, and traffic crosses PCIe host bridges rather than
an NVLink fabric. CUDA peer access is available in both directions. Direct
cross-GPU copies measured approximately 45.56 GiB/s from GPU 0 to GPU 1 and
48.37 GiB/s in the reverse direction.

The absence of NVLink is therefore not a functional blocker for DDP. It affects
the performance ceiling for communication-heavy workloads and must be included
in performance expectations.

## Runtime limits

- `/dev/shm`: 125 GB
- `memlock`: 8,192 KiB soft and hard
- `nofile`: 1,024 soft / 1,048,576 hard
- `stack`: 8,192 KiB soft / unlimited hard

The login user can raise `nofile` and stack but cannot raise memlock. The 8 MiB
memlock limit did not prevent the tests below, but it remains below the
production requirement. The final Docker/runtime template must configure
unlimited memlock; this cannot be repaired by an unprivileged process inside
the container.

## NCCL discovery and root cause

The discovery image contains PyTorch `2.7.0+cu128` and NCCL `2.26.2`. On this
dual-Blackwell host, the default two-rank collective either hung during
bootstrap or failed with `CUDA error: an illegal memory access was encountered`.
The failure reproduced with the P2P/IPC and shared-memory transports, including
with CUDA memory and DMA-BUF features disabled.

A socket-only fallback passed when P2P and shared memory were disabled, but at
only 1.66 GiB/s. This confirmed that rank launch and basic distributed process
coordination worked while isolating the defect to the local NCCL transport
path.

An official `nvidia-nccl-cu12==2.26.5` runtime was then preloaded without
changing PyTorch. With NCCL `2.26.5+cuda12.9`, the normal P2P/CUMEM path passed:

```text
2 ranks, 256 MiB, 10 iterations: 0.083 s, 30.13 GiB/s
2 ranks, 1024 MiB, 50 iterations: 1.568 s, 31.89 GiB/s
```

The image fix in commit `5dc641d` preserves the PyTorch dependency metadata but
adds the ABI-compatible NCCL 2.26.5 runtime and automatically enables it for
the `agile-sonic` training environment. CI also calls `ncclGetVersion` during
the build so that an incorrectly loaded runtime fails publication.

The immutable `5dc641d` image was cold-pulled after an in-place Vast recycle.
Without any `/cache` dependency overlay, its built-in runtime passed the same
1 GiB, 50-iteration test in 1.574 seconds at 31.77 GiB/s.

### Non-root profile defect found during cold-image validation

Cold testing exposed one packaging defect that root-only CI did not catch.
BuildKit created `/usr/local/lib/agile-sonic` with mode `0644` while copying the
`0644` helper into a previously absent destination directory. The non-root
login user could not traverse that directory, so `/etc/profile.d/agile-sonic.sh`
failed before it could automatically enable the built-in library.

No server package was installed to work around this. The private workload below
explicitly preloaded the library already present under `/opt/agile-sonic` while
commit `4655f45` added a `0755` directory fix and a build-time gate that sources
the helper as user `fangzhengtian`. A final cold-image test must prove automatic
activation before release handoff.

## Generic DDP training gate

Status: **passed with NCCL 2.26.5**

A two-rank PyTorch DDP job ran ten real forward, backward and AdamW optimizer
steps on rank-specific inputs. Both ranks produced the same final parameter
checksum, and the maximum parameter delta was non-zero:

```text
checksum=-50.56426239
max_delta=0.0010059588
```

This is stronger than an import or initialization check: it exercises gradient
collectives and synchronized parameter updates on both GPUs.

## Authorized private source deployment

The user explicitly authorized upload to instance `46697258`. The filtered
source bundle contained 671 regular files and 238,929,129 uncompressed bytes.
Git history, complete datasets, checkpoints, logs, virtual environments,
generated package metadata and unrelated assets were excluded. The two extra
files found under the remote source directory were Vast's own `onstart.sh` and
`ports.log`.

The paired fixture hashes matched the public manifest:

```text
robot  9a6df6b558219c9b76f5cf454f5e4175461cdba0947a58bdedf8e4bd82cc625d
soma   f2397a1fce89b0c06d89e80f95a8bbb2d6a917b0bb5b73b9fed912e0987c2dcc
```

## Real two-GPU SONIC/H20 training smoke

Status: **passed with the built-in NCCL 2.26.5 library explicitly preloaded**

The production launcher started two Accelerate/DDP ranks in BF16 mode. Rank 0
used `cuda:0` and rank 1 used `cuda:1`; each created 12 Isaac environments and
loaded the real H20 MJCF plus the paired robot/SOMA motion. Two eight-step PPO
rollouts and two optimizer updates completed:

```text
step 1: 192 total real transitions
step 2: 384 total real transitions
```

Both checkpoints and `last.pt` were written once by the main process without a
save race. The strict verifier found finite, non-zero changes in every required
group:

- H20 encoder: 8 tensors, max delta `0.000112448`;
- teleop encoder: 10 tensors, max delta `0.000112547`;
- SOMA encoder: 10 tensors, max delta `0.000112310`;
- H20 dynamics decoder: 8 tensors, max delta `0.000112650`;
- H20 kinematics decoder: 8 tensors, max delta `0.000111341`;
- critic: 8 tensors, max delta `0.000112624`;
- optimizer: 53 populated state slots.

Checkpoint hashes:

```text
b45e7126c58483a6dd1a39cadc5c3bdc2a39b9598822f2d8cadc901855c71c8c  model_step_000001.pt
752d4026e024527879ce89ceb49b7f61f06d1d1749d638261d929f4164d7015d  model_step_000002.pt
```

The fatal-log scan found no CUDA illegal memory access, NCCL error, OOM,
permission failure or segmentation fault. Both GPUs returned to 2 MiB, 0%
utilization and P8, with no orphaned training or Kit process.

## Same-topology checkpoint resume

Status: **passed**

The first invocation used `resume=true`, but Hydra correctly rejected it because
the key is not present in the structured base config; the required syntax is
`+resume=true`. This failed before simulator creation and did not alter the
checkpoint. The corrected two-rank invocation loaded `last.pt` on both ranks,
resumed from step 2 and completed step 3 at 576 total transitions.

The generalized strict verifier confirmed another finite, non-zero update in
all six model groups and 53 optimizer slots between steps 2 and 3. The step-3
checkpoint hash is:

```text
9136e674dc019d2f76a4ee7d6d8ce637982cb40b259088193ddf19393c8739c7  model_step_000003.pt
```

## Final clean-image release gate

Status: **passed**

GitHub Actions run
[`30813887535`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/actions/runs/30813887535)
published commit `4655f45` in 15 minutes 9 seconds. Peak sampled runner disk use
was 54,868,590,592 bytes, minimum remaining space was 57,332,219,904 bytes and
the largest compressed layer was 4,420,997,816 bytes. Promotion and provenance
attestation
[`38569052`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/attestations/38569052)
passed.

The instance was recycled again to the final digest, erasing the earlier source,
cache and checkpoints. After the authorized bundle was re-uploaded, the non-root
profile was tested with `LD_PRELOAD` explicitly unset:

```text
helper directory mode: 0755
helper file mode:      0644
automatic LD_PRELOAD:  /opt/agile-sonic/nccl-runtime/nvidia/nccl/lib/libnccl.so.2
ncclGetVersion:         22605
```

No `/cache` dependency overlay existed. Automatic activation then passed a
1 GiB, 50-iteration NCCL test in 1.574 seconds at 31.76 GiB/s. A second clean
two-rank BF16 SONIC smoke completed the same 192/384 transitions and strict
all-module update checks without any manual NCCL environment setting.

Final checkpoint hashes:

```text
7bd0c2e80f752204666a7ef36f06c633a81afdd323832e2d88e7a773a7b21dfe  model_step_000001.pt
a5929c4b9fb4424fe4a2a783701bd1282c69497967c7ae1c9ecd26fff533d25c  model_step_000002.pt
```

The final fatal-log scan was clean and both GPUs returned to 2 MiB, 0%
utilization and P8. The instance was stopped afterward, then permanently
destroyed on 2026-08-03 with its 400 GB disk. Vast returned `instances: null`;
no uploaded private source, cache, logs or checkpoints remain on the platform.
