# Agile SONIC Training

Agile SONIC Training is the reproducible container and server-launch project for
training the Agile H20 robot with NVIDIA GEAR-SONIC / Whole-Body Control.

> **Hardware naming:** H20 is the robot model. It is not the training GPU. The
> target training server uses NVIDIA RTX PRO 6000 Blackwell GPUs.

The project keeps the useful infrastructure pattern from the original StarVLA
container while replacing its Python and ML environment with the SONIC stack:

```text
CUDA base image
  -> Linux packages and non-root user
  -> key-only SSH
  -> AWS CLI and Alibaba Cloud CLI
  -> Miniforge
  -> SONIC training environments
  -> runtime entrypoint and UID/GID remapping
```

This public repository contains the container environment, CI/CD workflow and
server scripts. It intentionally does **not** contain the private
`sonic-training` checkout, datasets, model assets, checkpoints or credentials.
Those are mounted when the container starts.

## Current verified build

The current verified artifact is the `training` target built from commit
[`efd9c83`](https://github.com/cyfarwydd-tian/Agile-Sonic-training/commit/efd9c83465a0448dec855ad4c9088ac19a80cac5).

- [Successful GitHub Actions run](https://github.com/cyfarwydd-tian/Agile-Sonic-training/actions/runs/30804335746)
- Build and publish time: 22 minutes 57 seconds
- Cold-build sampled disk consumption: 56,978,497,536 bytes
- Minimum remaining runner space: 55,222,448,128 bytes
- Largest compressed layer: 4,102,248,768 bytes
- GHCR layer check, tag promotion and provenance attestation: passed

Pull the exact immutable image:

```bash
docker pull \
  ghcr.io/cyfarwydd-tian/agile-sonic-training@sha256:6ab2d3aac46c74bb97cae40611262c16163ef1b3aae9cf503ce7e97e2f6b7b59
```

The commit-based convenience tag points to the same build:

```bash
docker pull ghcr.io/cyfarwydd-tian/agile-sonic-training:sha-efd9c83-training
```

The package currently permits anonymous manifest access. A GHCR login is not
required for this public build. See [PARTNER_BUILD.md](PARTNER_BUILD.md) for the
short partner handoff.

This digest also passed a clean one-GPU RTX PRO 6000 Blackwell deployment and
real two-update H20 PPO smoke. See
[VASTAI_CLEAN_REBUILD_46693892.md](VASTAI_CLEAN_REBUILD_46693892.md) for the
runtime evidence and remaining multi-GPU gates.

## What is installed

### Base system and operations tooling

| Component | Installed version or purpose |
| --- | --- |
| Base OS | Ubuntu 22.04 |
| CUDA image | CUDA 12.8.1 + cuDNN development image |
| Architecture | Linux `amd64` / `x86_64` |
| Container user | `fangzhengtian`; runtime UID/GID can match the host |
| Miniforge | 26.3.2-2 under `/opt/miniforge3` |
| AWS CLI | 2.36.13 |
| Alibaba Cloud CLI | 3.3.18 |
| SSH | OpenSSH server, public-key only, disabled unless a key is supplied |
| Build tools | GCC/G++, CMake, pkg-config, Ninja, Git and Git LFS |
| Media and graphics | FFmpeg, OpenGL/EGL, Vulkan, X11, Xvfb and GTK libraries |
| Operations | curl, wget, rsync, tmux, jq, sudo and a minimal Vim |

The image also includes USB/udev and common rendering libraries required by
robotics, camera and headless Isaac workflows. The exact apt package list is in
[`files/init-dep.sh`](files/init-dep.sh).

### Verified `training` Python environment

The default Conda environment is `agile-sonic`:

| Component | Version |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.7.0 + CUDA 12.8 |
| TorchVision | 0.22.0 + CUDA 12.8 |
| TorchAudio | 2.7.0 + CUDA 12.8 |
| Triton | 3.3.0 |
| Isaac Sim | 5.1.0, including Kit/SDK/Physics extension caches |
| Isaac Lab | 2.3.2, pinned source revision |
| Hugging Face Accelerate | 1.14.0 |
| Transformers / TRL | 4.57.6 / 0.28.0 |
| Hydra / OmegaConf | 1.3.2 / 2.3.0 |
| TensorDict | 0.7.2 |
| Gymnasium | 1.2.1 |
| h5py | 3.13.0, installed and runtime-verified in the current image |
| MuJoCo | 3.3.2 |
| Open3D / VTK | 0.19.0 / 9.4.2 |
| OpenCV | 4.11.0.86, headless build |
| W&B / TensorBoard | 0.23.1 / 2.20.0 |

The training image also installs the pinned public sources used by the runtime:

- NVIDIA `GR00T-WholeBodyControl`
- Isaac Lab
- SMPLSim
- SMPL-X

The complete direct package list and hard compatibility pins are in:

- [`files/requirements-training.txt`](files/requirements-training.txt)
- [`files/constraints.txt`](files/constraints.txt)
- [`files/agile-sonic.yml`](files/agile-sonic.yml)

Build-time import checks, `pip check`, `pip freeze`, apt packages, toolchain
versions and source revisions are recorded inside the image under
`/opt/agile-sonic/manifests/`.

### Image targets

| Target | Contents | Current status |
| --- | --- | --- |
| `training` | CUDA, PyTorch, Isaac Sim/Lab and SONIC multi-GPU RL stack | Built and published successfully |
| `tensorrt` | `training` plus TensorRT 10.13.3.9, CUDA Python 12.8 and ONNX Runtime 1.27 | Defined; requires a separate publish/runtime validation |
| `full` | `tensorrt` plus isolated tools, data and VLA inference environments | Defined; requires a larger publish/runtime validation |

The `full` target keeps incompatible workloads in separate environments:

| `SONIC_ENV` | Core environment | Purpose |
| --- | --- | --- |
| `training` / `agile-sonic` | Python 3.11, CUDA Torch 2.7 | Isaac/SONIC RL training |
| `tools` / `agile-sonic-tools` | Python 3.10, CPU Torch 2.6 | MuJoCo, teleoperation, cameras, RoboSuite and CycloneDDS |
| `data` / `agile-sonic-data` | Python 3.10, CPU Torch 2.6 | LeRobot data collection and conversion |
| `inference` / `agile-sonic-inference` | Python 3.12, CUDA Torch 2.9 | Isaac-GR00T VLA inference; experimental |

The optional target manifests are under `files/requirements-*.txt` and
`files/agile-sonic-*.yml`.

## Host requirements

For the RTX PRO 6000 Blackwell server, use:

1. Ubuntu 22.04 or 24.04 x86_64.
2. NVIDIA R580 driver, at least `580.65.06` or a validated newer R580 patch.
3. Docker Engine.
4. NVIDIA Container Toolkit configured for Docker.
5. Local NVMe for Docker layers and node-local caches.
6. Sufficient RAM, CPU cores and PCIe bandwidth for the selected GPU count.

The host does not need a separate CUDA Toolkit installation; the CUDA user-space
stack is in the image.

Validate the host before using this project:

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi topo -p2p p
nvidia-ctk --version

docker run --rm --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu22.04 \
  nvidia-smi
```

## Server directory layout

Keep source, datasets, outputs and caches separate:

```text
/srv/agile-sonic/
├── source/       # private sonic-training checkout
├── datasets/     # training data, read-only by default
├── runs/         # checkpoints, Hydra logs and W&B offline runs
└── cache/        # Isaac, Hugging Face, Torch and NVIDIA caches
```

If the private project uses Git LFS, materialize all objects before launching:

```bash
cd /srv/agile-sonic/source
git lfs install
git lfs pull
git lfs ls-files
```

The second column of `git lfs ls-files` must be `*`; `-` means the working tree
still contains pointer text.

## Start the training container

Clone this repository on the server so the validated launcher is available,
then run:

```bash
scripts/run.sh \
  --image ghcr.io/cyfarwydd-tian/agile-sonic-training@sha256:6ab2d3aac46c74bb97cae40611262c16163ef1b3aae9cf503ce7e97e2f6b7b59 \
  --source /srv/agile-sonic/source \
  --datasets /srv/agile-sonic/datasets \
  --runs /srv/agile-sonic/runs \
  --cache /srv/agile-sonic/cache \
  --gpu-request all \
  --gpu-count <GPU_COUNT> \
  --ssh-port 0 \
  --detach
```

Do not replace this with a minimal `docker run --gpus all`. The launcher also
configures host IPC, unlimited memlock, file descriptor and stack limits, GPU
selection, log rotation, persistent paths and runtime UID/GID mapping.

To re-enter the container:

```bash
docker exec --user fangzhengtian -it agile-sonic-training bash -l
```

## Required runtime gates

Run these checks before any production job:

```bash
agile-sonic-preflight --strict
agile-sonic-nccl-smoke --gpu-count <GPU_COUNT>
```

They validate:

- Python, PyTorch, CUDA, Isaac Sim/Lab and important imports.
- Python dependency consistency.
- GPU visibility and device properties.
- source, dataset, cache and run mounts.
- atomic checkpoint-directory writes.
- single-node multi-GPU NCCL all-reduce.

Then run the self-contained SONIC training smoke test on one GPU:

```bash
scripts/sonic-training-smoke.sh \
  --source /srv/agile-sonic/source \
  --runs /srv/agile-sonic/runs \
  --cache /srv/agile-sonic/cache \
  --gpu 0
```

This is a real two-update training test, not an import check. It packages a
paired 120-frame H20 robot/SOMA fixture, runs Isaac rollouts and PPO, and fails
unless every active encoder, decoder, critic and optimizer path changes between
the two checkpoints. See [`smoke/sonic_h20/README.md`](smoke/sonic_h20/README.md)
for its exact coverage. The `h20` name refers to the Agile H20 robot.

Before production training, also verify checkpoint save/restore and complete at
least a 15–30 minute all-GPU stability test. Pin the private project revision:

```bash
SONIC_EXPECTED_REVISION=<40-character-commit-sha> \
SONIC_REQUIRE_CLEAN=1 \
agile-sonic-launch --gpu-count <GPU_COUNT> -- <training arguments...>
```

The first production phase uses Accelerate DDP on one multi-GPU host.
DeepSpeed/FSDP and multi-node RDMA require separate validation.

## Vast.ai validation

Vast.ai is used as a pre-delivery GPU test platform. Select an instance with a
compatible NVIDIA driver, enough GPUs, RAM and local NVMe, then execute the same
host, preflight, NCCL, minimal-training and checkpoint gates described above.

The image does not include private source or datasets, so they must be uploaded
or cloned into persistent Vast.ai storage. Spot instances must continuously sync
`/runs` to durable storage.

See [`VastAI.md`](VastAI.md) for the platform-specific checklist.

## SSH and credentials

Container SSH is optional. It has no default password, forbids root/password
login and starts only when an authorized public key is supplied. Prefer host SSH
plus `docker exec`; do not expose container port 22 unless required.

AWS and Alibaba Cloud CLIs are installed, but credentials are not. Prefer, in
order:

1. Cloud instance roles or short-lived STS credentials.
2. Read-only mounted credential files or Docker secrets.
3. Standard environment variables.

Never put credentials in the Dockerfile, `.env`, Compose files committed to Git,
GitHub Actions logs or public Vast.ai templates. See
[`compose.secrets.yaml.example`](compose.secrets.yaml.example).

W&B defaults to offline mode. Set credentials and an explicit mode only when the
server is allowed to communicate with W&B.

## CI/CD

The workflow in [`.github/workflows/container.yml`](.github/workflows/container.yml)
performs source validation, Buildx checks, registry-cache import/export, candidate
image publication, compressed-layer enforcement, release-tag promotion, SBOM and
provenance attestation.

Pushes to `agent/**` currently publish the `training` target. Other targets retain
a larger disk guard and should be published only after their own capacity and
runtime validation.

## Repository layout

```text
Dockerfile                       multi-stage image definition
compose.yaml                     single-node multi-GPU runtime configuration
files/                           pinned environments and image setup scripts
scripts/run.sh                   validated Docker launcher
scripts/preflight.sh             runtime dependency and mount checks
scripts/nccl-smoke.sh            single-node NCCL test
scripts/launch-multigpu.sh       Accelerate DDP launcher
.github/workflows/container.yml  GitHub Actions build and publication
PARTNER_BUILD.md                 exact partner build handoff
VastAI.md                        Vast.ai deployment notes
VASTAI_CLEAN_REBUILD_46693892.md clean rebuild and real training audit
```

Architecture and dependency decisions are documented in
[`docs/architecture.md`](docs/architecture.md).
