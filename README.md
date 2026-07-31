# Agile SONIC Training Container

面向 Agile H20 / NVIDIA GEAR-SONIC 的版本固定训练容器。项目保留原始
StarVLA 容器的基础设施思路：

```text
CUDA 基础镜像
  -> apt 与普通用户
  -> SSH
  -> AWS CLI / Alibaba Cloud CLI
  -> Miniforge
  -> SONIC 隔离环境组（training / tools / data / inference）
  -> entrypoint 初始化并降权运行
```

但 Python/CUDA/Isaac 依赖已经改为 SONIC 训练所需的版本组合，不能与原
StarVLA 的 Python 3.10 / PyTorch 2.8 环境混用。

本仓库只负责公开的容器环境和运维脚本，不包含私有的 `sonic-training`
源码、训练数据、模型或云凭据。源码和数据在运行容器时分别挂载。
依赖冲突、分层理由和原 Docker 范式的具体改进见
[容器架构与依赖决策](docs/architecture.md)。

## Version matrix

默认训练环境：

- Ubuntu 22.04
- CUDA 12.8.1 + cuDNN development image
- Python 3.11
- PyTorch 2.7.0 + cu128
- TorchVision 0.22.0 + cu128
- TorchAudio 2.7.0 + cu128
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2
- Conda 环境名：`agile-sonic`
- 容器用户：`fangzhengtian`（UID/GID 可配置）

`full` 不是把所有包强行安装到同一个 Python 解释器，而是在同一镜像中提供：

| `SONIC_ENV` | Python / 核心栈 | 用途 |
| --- | --- | --- |
| `training` / `agile-sonic` | Python 3.11、Torch 2.7、Isaac Sim/Lab | 默认；正式 RL 多卡训练 |
| `tools` / `agile-sonic-tools` | Python 3.10、CPU Torch 2.6 | MuJoCo、teleop、camera、Unitree、RoboSuite |
| `data` / `agile-sonic-data` | Python 3.10、CPU Torch 2.6 | LeRobot 数据采集与转换 |
| `inference` / `agile-sonic-inference` | Python 3.12、Torch 2.9 | Isaac-GR00T VLA 推理；当前标记为 experimental |

必须隔离的原因不是偏好，而是上游的硬约束相互冲突：例如 Isaac 训练要求
Python 3.11 / Torch 2.7 / `packaging==23.0`，固定的 LeRobot 版本要求
Torch `<2.7` / `packaging>=24.2`，当前 Isaac-GR00T 又要求 Python 3.12 /
Torch 2.9。entrypoint 默认选择 `training`，也可在启动时设置
`SONIC_ENV=tools|data|inference`。

镜像提供三个构建目标：

| Target | 用途 |
| --- | --- |
| `training` | 多卡 RL 训练、Isaac Sim/Lab、Open3D 与训练检查 |
| `tensorrt` | `training` 加 ONNX/TensorRT 导出与推理 |
| `full` | 聚合全部隔离环境；加入数据、VLA inference、teleop、camera、RoboSuite 等工具 |

正式训练建议固定镜像 digest，而不是只使用 `latest`。

## Host prerequisites

服务器需要：

1. NVIDIA 驱动能够支持 CUDA 12.8。
2. Docker Engine、BuildKit/buildx。
3. NVIDIA Container Toolkit，并已为 Docker runtime 配置 GPU。
4. 足够的本地 NVMe 空间。镜像、Isaac 缓存、数据和 checkpoint 应分开估算。

重要边界：Isaac Sim 5.1 官方页面目前已标记该版本停止支持，列出的 Linux
测试驱动是 `580.65.06`，H20 也不在其官方 GPU 支持表内；该页面还明确说明
没有 RT Core 的 A100/H100 不受支持。因此本项目的 H20 + headless 训练组合必须
视为目标硬件验证路径，不能仅凭镜像成功构建就宣称受官方支持。上线前必须在实际
H20 服务器跑完本文的 GPU、NCCL 和最小 PPO 门禁。

基础检查：

```bash
nvidia-smi
docker run --rm --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
docker buildx version
```

## Build

从仓库根目录执行：

```bash
scripts/build.sh --target training --load
scripts/build.sh --target full --load
```

指定镜像名、tag、UID/GID：

```bash
scripts/build.sh \
  --image ghcr.io/cyfarwydd-tian/agile-sonic-training \
  --target full \
  --tag local \
  --uid "$(id -u)" \
  --gid "$(id -g)" \
  --load
```

推送镜像：

```bash
scripts/build.sh \
  --image ghcr.io/cyfarwydd-tian/agile-sonic-training \
  --target full \
  --tag dev \
  --push
```

如服务器构建需要代理，可通过标准的 `HTTP_PROXY`、`HTTPS_PROXY` 和
`NO_PROXY` 环境变量传入。代理只作为构建参数使用，不写入最终镜像。

## Directory layout on a training server

推荐把四类数据明确隔离：

```text
/srv/agile-sonic/
├── source/       # 私有 sonic-training checkout
├── datasets/     # 只读训练数据
├── runs/         # checkpoint、日志、W&B offline 数据
└── cache/        # 每台节点自己的 Isaac/HF/Torch/NVIDIA 缓存
```

私有源码使用 Git LFS 保存模型、motion、mesh 和二进制资产。服务器 clone 后必须先：

```bash
cd /srv/agile-sonic/source
git lfs install
git lfs pull
git lfs ls-files  # 第二列应为 *；出现 - 表示工作树仍是 pointer
```

镜像内 preflight 会检查当前工作树是否仍有 LFS pointer；未拉全资产时会在启动训练前
直接失败，而不是等 Isaac 加载到一半才报错。

不要把大量小 PKL 数据放进 Docker image。多节点训练时，每台节点应把数据同步到
本地 NVMe，并使用 manifest、文件数量和 SHA-256 门禁确认内容一致。

## Run a container

```bash
scripts/run.sh \
  --image ghcr.io/cyfarwydd-tian/agile-sonic-training:latest \
  --source /srv/agile-sonic/source \
  --datasets /srv/agile-sonic/datasets \
  --runs /srv/agile-sonic/runs \
  --cache /srv/agile-sonic/cache \
  --gpu-request all \
  --detach
```

默认容器名为 `agile-sonic-training`。源码挂载到
`/workspace/sonic-training`，数据挂载到 `/datasets`，输出挂载到 `/runs`，
缓存挂载到 `/cache`。训练时 `/datasets` 默认只读；只有数据采集或同步任务才显式
增加 `--datasets-rw`。Compose 的对应开关是 `DATASETS_READ_ONLY=false`。

`scripts/run.sh` 会让非 root 调用者创建的输出目录保持原 ownership；root 运维时优先
继承 `SUDO_UID/SUDO_GID`，否则默认使用 1000:1000，并且只调整本次新建的精确目录。
entrypoint 默认不会 `chown` 已存在的 bind mount，避免破坏 NFS/root-squash 或 ACL。
简单本地磁盘确需修复挂载根 ownership 时才显式设置
`FIX_MOUNT_OWNERSHIP=1`；共享盘应在宿主机用 ACL/UID 映射处理。

选择镜像内的其他隔离环境：

```bash
scripts/run.sh --env tools \
  --source /srv/agile-sonic/source --datasets /srv/agile-sonic/datasets
scripts/run.sh --env data --datasets-rw \
  --source /srv/agile-sonic/source --datasets /srv/agile-sonic/datasets
scripts/run.sh --env inference \
  --source /srv/agile-sonic/source --datasets /srv/agile-sonic/datasets
```

重新进入：

```bash
docker exec --user fangzhengtian -it agile-sonic-training bash -l
echo "$CONDA_DEFAULT_ENV"  # 默认 agile-sonic
```

运行脚本会配置 GPU、共享内存、memlock、nofile、stack、日志轮转和缓存路径。
正式训练不要退回只有 `--gpus all` 的最小 `docker run` 命令。
entrypoint 会把本次不含凭据的 `RUN_ID/RUN_DIR/WANDB_DIR` 写到
`/run/agile-sonic/runtime.env`，登录 shell 和镜像内 launcher 会自动读取，因此
后续 `docker exec` 不会意外切换到另一个 run 目录。

使用 Compose 选择两张卡时，`NVIDIA_VISIBLE_DEVICES` 只写普通逗号列表，不能使用
Docker CLI 的 `device=...` 语法：

```bash
SONIC_SOURCE=/srv/agile-sonic/source \
DATASETS_DIR=/srv/agile-sonic/datasets \
NVIDIA_VISIBLE_DEVICES=0,1 \
GPU_COUNT=2 \
docker compose up -d sonic-training
```

也可以先把 `.env.example` 复制为 `.env`，只填写这些非敏感路径和运行参数；云密钥
仍必须走后文的只读文件或 Compose secrets。

## SSH

容器保留 SSH，但默认执行以下安全策略：

- 不创建默认密码。
- 禁止 root SSH 登录。
- 禁止 SSH 密码认证。
- 只在提供 authorized key 时启动 sshd。
- host key 在容器第一次启动时生成，不烘焙进镜像。

推荐从文件传入公钥：

```bash
SSH_AUTHORIZED_KEYS_FILE=/secure/sonic_authorized_keys \
scripts/run.sh \
  --source /srv/agile-sonic/source \
  --datasets /srv/agile-sonic/datasets \
  --ssh-port 2222 \
  --detach
```

也兼容 `SSH_AUTHORIZED_KEYS` 和旧变量 `SSH_USER_PUBLIC_KEY`。在可直接 SSH
登录宿主机的服务器上，优先使用宿主机 SSH 加 `docker exec`，无需暴露容器 22 端口。

## AWS and Alibaba Cloud credentials

优先顺序：

1. 云实例角色、RAM role、STS 等短期身份。
2. 只读挂载的 credentials 文件或 Docker secret。
3. 标准环境变量。
4. 仅为兼容旧流程保留的变量名。

AWS 标准变量：

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN
AWS_REGION / AWS_DEFAULT_REGION
AWS_SHARED_CREDENTIALS_FILE
AWS_CONFIG_FILE
```

Alibaba Cloud 标准变量：

```text
ALIBABA_CLOUD_ACCESS_KEY_ID
ALIBABA_CLOUD_ACCESS_KEY_SECRET
ALIBABA_CLOUD_SECURITY_TOKEN
ALIBABA_CLOUD_REGION_ID
```

entrypoint 支持相应的 `_FILE` 变量，也兼容 `AWS_KEY_ID`、`AWS_KEY`、
`ALICLOUD_KEY_ID`、`ALICLOUD_KEY` 和 `ALICLOUD_REGION`。不要把真实密钥写入
Dockerfile、Compose 文件、GitHub Actions 日志或仓库。

Compose 场景可复制 `compose.secrets.yaml.example` 到仓库外的安全目录，填写的
只是宿主机文件路径，不是凭据内容；删除未使用的 secret 声明/挂载（SSH 还需删除
对应 environment 项）后再启动。基础 `compose.yaml` 不发布 SSH 端口，只有这个
包含 authorized key 的 override 才默认把宿主机 `2222` 映射到容器 22：

```bash
SONIC_SOURCE=/srv/agile-sonic/source \
DATASETS_DIR=/srv/agile-sonic/datasets \
docker compose \
  -f compose.yaml \
  -f /secure/compose.secrets.yaml \
  up -d sonic-training
```

不要把凭据写入 `.env`；它适合放镜像名、挂载路径和 GPU 数量等非敏感配置。

## Preflight

启动容器并进入其 shell 后，在正式训练前依次执行镜像自带的命令：

```bash
agile-sonic-preflight
agile-sonic-nccl-smoke --gpu-count 2
```

门禁包含：

- Python、PyTorch、CUDA、Isaac Sim/Lab 和关键包版本。
- `pip check`。
- GPU 数量和每张 GPU 的基本信息。
- 挂载路径与 `/runs` 原子写入。
- 单机多卡 NCCL all-reduce。

随后使用两张卡跑最小 SONIC smoke：

```bash
agile-sonic-launch --gpu-count 2 -- \
  +exp=manager/universal_token/all_modes/sonic_release_h20_smoke \
  headless=true \
  use_wandb=false \
  ++algo.config.num_learning_iterations=2
```

launcher 在未收到显式 `base_dir=...` 时会自动使用
`/runs/<RUN_ID>`，checkpoint、Hydra 日志和 W&B 目录不会落入源码挂载。生产任务还应
把私有源码固定到完整 commit，并启用 clean checkout 门禁：

```bash
SONIC_EXPECTED_REVISION=<40-character-commit-sha> \
SONIC_REQUIRE_CLEAN=1 \
agile-sonic-launch --gpu-count 2 -- <training arguments...>
```

当挂载的私有源码含 `scripts/env_checks/check_training_container.py` 且镜像是
`tensorrt/full` target 时，launcher 的默认 preflight 会自动执行这份当前源码的
H20 dependency/asset 检查；纯 `training` target 因不含 TensorRT，只执行基础门禁并
给出提示。镜像构建时的 import smoke 针对固定公开 fallback，不能替代这一步。

`num_envs` 是每个 rank 的环境数量。将单卡正式实验迁移到 N 卡做严格 A/B 时，
先把每卡 `num_envs` 调整为原值除以 N，避免无意中把全局 rollout 放大 N 倍。

第一阶段只使用 Accelerate DDP。DeepSpeed/FSDP 需要另外验证分片 checkpoint
保存、恢复和 optimizer state 后才能启用。

## GitHub CI/CD and GHCR

`.github/workflows/container.yml` 会：

- 对 pull request 运行静态验证和 Docker build check。
- 对 `main`、`agent/**` 和语义版本 tag 的 push 自动构建。
- 使用 Buildx 和 GHCR registry cache。
- 将镜像推送到：

```text
ghcr.io/cyfarwydd-tian/agile-sonic-training
```

主要 tag 约定：

```text
latest                 # main 的 full 镜像
latest-full
latest-training
latest-tensorrt
sha-<commit>
<branch>-<target>
vX.Y.Z
```

GitHub 标准 public Linux runner 只有约 14 GB 可用 SSD，不能完成本项目实测约
85 GB 量级的 Isaac/full
这种大镜像。静态校验可以继续使用 hosted runner；正式发布前应把仓库变量
`DOCKER_RUNNER` 设置为有充足磁盘的 self-hosted runner label。发布 job 默认要求
Docker 数据盘至少有 250 GB 可用空间（需要调整时设置纯数字字节值
`DOCKER_MIN_FREE_BYTES`）；容量不足会在下载大依赖前明确失败。合作伙伴拉取 private
GHCR package 前需要：

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin
docker pull ghcr.io/cyfarwydd-tian/agile-sonic-training@sha256:<digest>
```

GHCR 还限制每个压缩 layer 不超过 10 GB，并要求单次 layer upload 在 10 分钟内
完成。Dockerfile 因此把训练环境、Torch、Isaac Sim 组件和项目依赖拆成独立 layer；
workflow 会先推唯一的 `candidate-*` tag，确认最大 layer 小于 9.5 GB 后才提升
branch/SHA/release tag。首次在大盘 runner 上完成真实构建后，仍应人工检查 layer
blob 大小和上传日志。

发布后应在目标 GPU 服务器完成 preflight、NCCL、两次 PPO update、checkpoint
保存/恢复和至少 15 分钟全卡压力测试，再开始正式训练。

## Reproducibility and maintenance

- 基础镜像、Miniforge、Python/CUDA/PyTorch、Isaac、云 CLI 和 Git 依赖必须固定版本。
- 当前直接依赖和源码 commit 已固定，但传递 pip/apt 依赖尚未全部 hash-lock；
  `pip freeze` 是构建产物审计记录，不等同于逐字节可重建。
- 依赖升级通过独立 PR 进行，并重新生成环境检查记录。
- `full` 用于开箱即用；生产训练若不需要 camera/teleop/TensorRT，优先拉取
  `training` target 以降低传输和磁盘压力。
- 私有 SONIC 源码、数据与 checkpoint 永远通过挂载提供，不进入公开镜像层。
- 构建生成的 `pip freeze`、`pip check`、公开源码 commit、apt 与 toolchain 记录位于
  `/opt/agile-sonic/manifests/`，用于服务器验收和问题复现。
