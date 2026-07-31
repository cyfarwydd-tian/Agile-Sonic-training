# syntax=docker/dockerfile:1.7

# Agile SONIC server image.
#
# Stage contract:
#   training -> CUDA/PyTorch/Isaac Sim/Isaac Lab and the multi-GPU RL stack
#   tensorrt -> training plus ONNX/TensorRT export and validation runtimes
#   full     -> tensorrt plus data, teleop, camera, MuJoCo and RoboCasa tooling
#
# The final "default" stage aliases "full", so `docker build .` produces the
# complete partner image. CI may publish each named target independently.

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85
FROM ${CUDA_IMAGE} AS system

ARG TARGETARCH
ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=fangzhengtian
ARG UID=1000
ARG GID=1000
ARG ENABLE_PASSWORDLESS_SUDO=0

# Docker expands ARG values here before bash sees them. This is intentional:
# UID is a readonly bash variable and must not be consumed directly by scripts.
ENV CONTAINER_USER=${USERNAME} \
    CONTAINER_HOME=/home/${USERNAME} \
    IMAGE_USER_UID=${UID} \
    IMAGE_USER_GID=${GID} \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Isaac Sim 5.1 pip wheels and the pinned cloud CLI artifacts below are amd64.
RUN test "${TARGETARCH:-amd64}" = "amd64"

# CUDA -> apt -> user
COPY --chmod=0755 files/init-dep.sh files/init-user.sh /usr/local/sbin/
RUN /usr/local/sbin/init-dep.sh \
    && ENABLE_PASSWORDLESS_SUDO="${ENABLE_PASSWORDLESS_SUDO}" \
       REQUESTED_UID="${IMAGE_USER_UID}" \
       REQUESTED_GID="${IMAGE_USER_GID}" \
       USERNAME="${CONTAINER_USER}" \
       /usr/local/sbin/init-user.sh \
    && rm -f /usr/local/sbin/init-dep.sh /usr/local/sbin/init-user.sh

ENV HOME=/home/${USERNAME} \
    USER=${USERNAME} \
    LOGNAME=${USERNAME} \
    PIP_CACHE_DIR=/cache/pip

# SSH is key-only and disabled at runtime unless a key is supplied.
COPY --chmod=0755 files/init-ssh.sh /usr/local/sbin/
RUN USERNAME="${CONTAINER_USER}" /usr/local/sbin/init-ssh.sh \
    && rm -f /usr/local/sbin/init-ssh.sh
EXPOSE 22

# Pinned, checksum-verified cloud CLIs.
ARG AWS_CLI_VERSION=2.36.13
ARG AWS_CLI_SHA256=a9ac6e52bbdf0bba62e410f7f62aa1a5f5615edb90b126c04cb5e4e3b2984bfc
COPY --chmod=0755 files/init-aws-cli.sh /usr/local/sbin/
RUN AWS_CLI_VERSION="${AWS_CLI_VERSION}" AWS_CLI_SHA256="${AWS_CLI_SHA256}" \
    /usr/local/sbin/init-aws-cli.sh \
    && rm -f /usr/local/sbin/init-aws-cli.sh

ARG ALICLOUD_CLI_VERSION=3.3.18
ARG ALICLOUD_CLI_SHA256=0823286604dbd8beb8d65dd0694d23c913e7c5d5a02b20a3593a4f8a6517f1d4
COPY --chmod=0755 files/init-alicloud-cli.sh /usr/local/sbin/
RUN ALICLOUD_CLI_VERSION="${ALICLOUD_CLI_VERSION}" \
    ALICLOUD_CLI_SHA256="${ALICLOUD_CLI_SHA256}" \
    /usr/local/sbin/init-alicloud-cli.sh \
    && rm -f /usr/local/sbin/init-alicloud-cli.sh

# Miniforge is root-owned under /opt. Runtime UID/GID remapping therefore never
# needs a recursive chown of the large Python/CUDA environment.
ARG MINIFORGE_VERSION=26.3.2-2
ARG MINIFORGE_SHA256=42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94
ENV CONDA_DIR=/opt/miniforge3 \
    CONDA_ENV_NAME=agile-sonic \
    CONDA_PREFIX=/opt/miniforge3/envs/agile-sonic
COPY --chmod=0755 files/init-conda.sh /usr/local/sbin/
RUN MINIFORGE_VERSION="${MINIFORGE_VERSION}" \
    MINIFORGE_SHA256="${MINIFORGE_SHA256}" \
    CONDA_DIR="${CONDA_DIR}" \
    /usr/local/sbin/init-conda.sh \
    && rm -f /usr/local/sbin/init-conda.sh

ENV PATH=${CONDA_PREFIX}/bin:${CONDA_DIR}/condabin:${PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    ISAACLAB_PATH=/opt/IsaacLab \
    SONIC_UPSTREAM_PATH=/opt/sonic-upstream \
    SONIC_WORKSPACE=/workspace/sonic-training \
    SONIC_WORKDIR=/workspace/sonic-training \
    DATASETS_DIR=/datasets \
    RUNS_DIR=/runs \
    PYTHONPATH=/workspace/sonic-training \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg \
    WANDB_CACHE_DIR=/cache/wandb \
    WANDB_MODE=offline \
    NCCL_DEBUG=WARN \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10

COPY --chmod=0755 files/init-training-env.sh /usr/local/sbin/
COPY files/agile-sonic.yml \
     files/agile-sonic-tools.yml \
     files/agile-sonic-data.yml \
     files/agile-sonic-inference.yml \
     files/constraints.txt \
     files/constraints-tools.txt \
     files/constraints-data.txt \
     files/constraints-inference.txt \
     files/requirements-training.txt \
     files/requirements-tensorrt.txt \
     files/requirements-full.txt \
     files/requirements-data.txt \
     files/requirements-inference.txt \
     /opt/agile-sonic/container/
ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.7.0
ARG TORCHVISION_VERSION=0.22.0
ARG TORCHAUDIO_VERSION=2.7.0
ARG TORCH_CUDA=cu128
ARG ISAACSIM_VERSION=5.1.0
ARG ISAACLAB_VERSION=2.3.2
ARG ISAACLAB_REPO=https://github.com/isaac-sim/IsaacLab.git
ARG ISAACLAB_REF=37ddf626871758333d6ed89cf64ad702aef127d0
ARG SONIC_REPO=https://github.com/NVlabs/GR00T-WholeBodyControl.git
ARG SONIC_REF=021df739f0b36e514399f0030e3a195683a46383
ARG SMPLSIM_REPO=https://github.com/ZhengyiLuo/SMPLSim.git
ARG SMPLSIM_REF=b5c08720503ad5fff64050c4d289c42d947fcf8d
ARG SMPLX_REPO=https://github.com/ZhengyiLuo/smplx.git
ARG SMPLX_REF=a5b8e4ac14f79f3f33fd2cf2a16e6f507146b813

# Layer 1/6: Miniforge environment and CUDA-enabled PyTorch.
RUN --mount=type=cache,id=agile-sonic-conda,target=/opt/conda-pkgs \
    --mount=type=cache,id=agile-sonic-pip,target=/root/.cache/pip \
    PYTHON_VERSION="${PYTHON_VERSION}" \
    TORCH_VERSION="${TORCH_VERSION}" \
    TORCHVISION_VERSION="${TORCHVISION_VERSION}" \
    TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION}" \
    TORCH_CUDA="${TORCH_CUDA}" \
    /usr/local/sbin/init-training-env.sh training-base

# Layer 2/6: Isaac Sim application packages and pinned Isaac Lab source.
RUN --mount=type=cache,id=agile-sonic-pip,target=/root/.cache/pip \
    ISAACSIM_VERSION="${ISAACSIM_VERSION}" \
    ISAACLAB_VERSION="${ISAACLAB_VERSION}" \
    ISAACLAB_REPO="${ISAACLAB_REPO}" \
    ISAACLAB_REF="${ISAACLAB_REF}" \
    /usr/local/sbin/init-training-env.sh training-isaac

# Layers 3-5/6: each large Isaac extension-cache distribution gets its own
# registry layer to remain within GHCR's per-layer upload limit.
RUN --mount=type=cache,id=agile-sonic-pip,target=/root/.cache/pip \
    ISAACSIM_VERSION="${ISAACSIM_VERSION}" \
    ISAACSIM_CACHE_PACKAGE=isaacsim-extscache-kit \
    /usr/local/sbin/init-training-env.sh training-isaac-cache
RUN --mount=type=cache,id=agile-sonic-pip,target=/root/.cache/pip \
    ISAACSIM_VERSION="${ISAACSIM_VERSION}" \
    ISAACSIM_CACHE_PACKAGE=isaacsim-extscache-kit-sdk \
    /usr/local/sbin/init-training-env.sh training-isaac-cache
RUN --mount=type=cache,id=agile-sonic-pip,target=/root/.cache/pip \
    ISAACSIM_VERSION="${ISAACSIM_VERSION}" \
    ISAACSIM_CACHE_PACKAGE=isaacsim-extscache-physics \
    /usr/local/sbin/init-training-env.sh training-isaac-cache

# Layer 6/6: SONIC/SMPL sources, RL dependencies, manifests and import gates.
RUN --mount=type=cache,id=agile-sonic-pip,target=/root/.cache/pip \
    TORCH_VERSION="${TORCH_VERSION}" \
    TORCH_CUDA="${TORCH_CUDA}" \
    ISAACSIM_VERSION="${ISAACSIM_VERSION}" \
    ISAACLAB_REF="${ISAACLAB_REF}" \
    SONIC_REPO="${SONIC_REPO}" \
    SONIC_REF="${SONIC_REF}" \
    SMPLSIM_REPO="${SMPLSIM_REPO}" \
    SMPLSIM_REF="${SMPLSIM_REF}" \
    SMPLX_REPO="${SMPLX_REPO}" \
    SMPLX_REF="${SMPLX_REF}" \
    /usr/local/sbin/init-training-env.sh training-project

# Empty mount points are created with final ownership directly; no recursive
# chown is performed over /workspace, Miniforge, Isaac Lab or SONIC sources.
RUN install -d -m 0755 /run/secrets \
    && install -d -m 0755 -o "${CONTAINER_USER}" -g "$(id -gn "${CONTAINER_USER}")" \
      /workspace/sonic-training \
      /datasets \
      /runs \
      /cache \
      /cache/huggingface \
      /cache/pip \
      /cache/torch \
      /cache/xdg \
      /cache/wandb \
      /cache/nvidia/ComputeCache \
      /cache/nvidia/GLCache \
      /cache/ov-data \
      /runs/nvidia-omniverse \
      /runs/wandb

# Runtime-only files are deliberately copied after the heavyweight training
# dependency layer so launcher/entrypoint edits do not invalidate CUDA/Isaac.
COPY --chmod=0755 files/entrypoint.sh /usr/local/sbin/entrypoint.sh
COPY --chmod=0755 files/sonic-env.sh /etc/profile.d/agile-sonic.sh
COPY --chmod=0755 scripts/ /opt/agile-sonic/scripts/
COPY --chmod=0755 files/agile-sonic-preflight.sh /usr/local/bin/agile-sonic-preflight
COPY --chmod=0755 files/agile-sonic-nccl-smoke.sh /usr/local/bin/agile-sonic-nccl-smoke
COPY --chmod=0755 files/agile-sonic-launch.sh /usr/local/bin/agile-sonic-launch

WORKDIR /workspace/sonic-training
ENTRYPOINT ["/usr/local/sbin/entrypoint.sh"]
CMD ["bash"]

FROM system AS training

ARG IMAGE_REVISION=unknown
LABEL org.opencontainers.image.title="Agile SONIC Training" \
      org.opencontainers.image.description="Version-pinned multi-GPU SONIC training environment" \
      org.opencontainers.image.source="https://github.com/cyfarwydd-tian/Agile-Sonic-training" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      com.agile-sonic.target="training"

FROM training AS tensorrt

ARG ONNXRUNTIME_VERSION=1.27.0
ARG TENSORRT_VERSION=10.13.3.9
ARG CUDA_PYTHON_VERSION=12.8.0
RUN --mount=type=cache,id=agile-sonic-conda,target=/opt/conda-pkgs \
    --mount=type=cache,id=agile-sonic-pip,target=/root/.cache/pip \
    ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION}" \
    TENSORRT_VERSION="${TENSORRT_VERSION}" \
    CUDA_PYTHON_VERSION="${CUDA_PYTHON_VERSION}" \
    /usr/local/sbin/init-training-env.sh tensorrt
LABEL com.agile-sonic.target="tensorrt"

FROM tensorrt AS full-tools

ARG SONIC_REF=021df739f0b36e514399f0030e3a195683a46383
ARG CYCLONEDDS_REPO=https://github.com/eclipse-cyclonedds/cyclonedds.git
ARG CYCLONEDDS_REF=5041f3560c088c99e5088b2b8520b69169621196
ARG ROBOSUITE_REPO=https://github.com/xieleo5/robosuite.git
ARG ROBOSUITE_REF=b6aa4a53939ba872382eb32cb8948af0b5b79cdd
ARG TOOLS_TORCH_VERSION=2.6.0
ARG TOOLS_TORCHVISION_VERSION=0.21.0
ARG TOOLS_TORCHAUDIO_VERSION=2.6.0
ARG TOOLS_TORCH_CUDA=cpu
ENV CYCLONEDDS_HOME=/opt/cyclonedds/install \
    CycloneDDS_DIR=/opt/cyclonedds/install/lib/cmake/CycloneDDS \
    CMAKE_PREFIX_PATH=/opt/cyclonedds/install \
    LD_LIBRARY_PATH=/opt/cyclonedds/install/lib:${LD_LIBRARY_PATH}
RUN --mount=type=cache,id=agile-sonic-conda,target=/opt/conda-pkgs \
    --mount=type=cache,id=agile-sonic-pip,target=/root/.cache/pip \
    CYCLONEDDS_REPO="${CYCLONEDDS_REPO}" \
    CYCLONEDDS_REF="${CYCLONEDDS_REF}" \
    ROBOSUITE_REPO="${ROBOSUITE_REPO}" \
    ROBOSUITE_REF="${ROBOSUITE_REF}" \
    SONIC_REF="${SONIC_REF}" \
    TOOLS_TORCH_VERSION="${TOOLS_TORCH_VERSION}" \
    TOOLS_TORCHVISION_VERSION="${TOOLS_TORCHVISION_VERSION}" \
    TOOLS_TORCHAUDIO_VERSION="${TOOLS_TORCHAUDIO_VERSION}" \
    TOOLS_TORCH_CUDA="${TOOLS_TORCH_CUDA}" \
    /usr/local/sbin/init-training-env.sh tools

FROM full-tools AS full-data

ARG DATA_TORCH_VERSION=2.6.0
ARG DATA_TORCHVISION_VERSION=0.21.0
ARG DATA_TORCHAUDIO_VERSION=2.6.0
ARG DATA_TORCH_CUDA=cpu
ARG LEROBOT_REPO=https://github.com/huggingface/lerobot.git
ARG LEROBOT_REF=a445d9c9da6bea99a8972daa4fe1fdd053d711d2
RUN --mount=type=cache,id=agile-sonic-conda,target=/opt/conda-pkgs \
    --mount=type=cache,id=agile-sonic-pip,target=/root/.cache/pip \
    DATA_TORCH_VERSION="${DATA_TORCH_VERSION}" \
    DATA_TORCHVISION_VERSION="${DATA_TORCHVISION_VERSION}" \
    DATA_TORCHAUDIO_VERSION="${DATA_TORCHAUDIO_VERSION}" \
    DATA_TORCH_CUDA="${DATA_TORCH_CUDA}" \
    LEROBOT_REPO="${LEROBOT_REPO}" \
    LEROBOT_REF="${LEROBOT_REF}" \
    /usr/local/sbin/init-training-env.sh data

FROM full-data AS full

ARG INFERENCE_TORCH_VERSION=2.9.0
ARG INFERENCE_TORCHVISION_VERSION=0.24.0
ARG INFERENCE_TORCHAUDIO_VERSION=2.9.0
ARG INFERENCE_TORCH_CUDA=cu128
ARG GROOT_REPO=https://github.com/NVIDIA/Isaac-GR00T.git
ARG GROOT_REF=b9955401d50c92a29258732e3ad6ccd579f1bdc0
RUN --mount=type=cache,id=agile-sonic-conda,target=/opt/conda-pkgs \
    --mount=type=cache,id=agile-sonic-pip,target=/root/.cache/pip \
    INFERENCE_TORCH_VERSION="${INFERENCE_TORCH_VERSION}" \
    INFERENCE_TORCHVISION_VERSION="${INFERENCE_TORCHVISION_VERSION}" \
    INFERENCE_TORCHAUDIO_VERSION="${INFERENCE_TORCHAUDIO_VERSION}" \
    INFERENCE_TORCH_CUDA="${INFERENCE_TORCH_CUDA}" \
    GROOT_REPO="${GROOT_REPO}" \
    GROOT_REF="${GROOT_REF}" \
    /usr/local/sbin/init-training-env.sh inference
LABEL com.agile-sonic.target="full"

# Keep the complete image as the no-target default while preserving the three
# public stage names above.
FROM full AS default
