# Vast.ai deployment notes

Vast.ai 只作为可选 GPU 供应商。正式训练仍应遵守本仓库 README 中的镜像、
挂载、preflight、NCCL 和 checkpoint 门禁。

## Before renting

选择实例时至少核对：

- GPU 型号、显存和 GPU 数量。
- NVIDIA driver/CUDA compatibility。
- GPU 间拓扑、PCIe/NVLink 和 NCCL 带宽。
- CPU 核数、内存和本地 NVMe。
- 实例最长可用时间、可靠性和上下行带宽。

完整 SONIC 环境、数据、缓存与 checkpoint 会占用大量空间。正式任务推荐至少
准备 300 GB，较大数据集建议 400–500 GB 或更多；不要沿用旧文档中的 32 GB 示例。

## Image

CI 发布的镜像位于：

```text
ghcr.io/cyfarwydd-tian/agile-sonic-training
```

优先使用 immutable digest：

```text
ghcr.io/cyfarwydd-tian/agile-sonic-training@sha256:<digest>
```

若 GHCR package 为 private，应在 Vast.ai 控制台中使用只具备
`read:packages` 权限的专用 token 配置 registry credential。不要把 token 放入
实例启动命令、公开模板或训练日志。

## Storage and source

容器镜像不包含私有 `sonic-training` 源码和训练数据。实例启动后按以下稳定路径准备：

```text
/workspace/sonic-training   private source checkout
/datasets                  read-only training data
/runs                      checkpoints, logs and W&B offline runs
/cache                     node-local Isaac/HF/Torch/NVIDIA caches
```

数据同步结束后，先验证 manifest、文件数量和 SHA-256，再启动训练。Spot/interruptible
实例必须把 `/runs` 定期同步到持久对象存储。

## SSH

镜像只允许 SSH public-key authentication，不提供默认密码。建议由 Vast.ai 平台负责
宿主机 SSH；如必须暴露容器 sshd，使用 `SSH_AUTHORIZED_KEYS_FILE` 或
`SSH_AUTHORIZED_KEYS`，并只映射平台分配的端口。

## Required gates

实例准备完成后依次运行：

```bash
agile-sonic-preflight
agile-sonic-nccl-smoke --gpu-count <visible-gpu-count>
```

然后执行 2-GPU、2-update SONIC smoke，验证 checkpoint 保存和同拓扑恢复。只有
环境版本、GPU 数量、NCCL、数据门禁、写盘、恢复以及短时全卡压力测试全部通过后，
才启动正式训练。

训练进程应由本仓库的多卡 launcher 和外部 manager 管理。不要仅依赖 SSH session、
`nohup` 或容器可写层保存状态。实例销毁前确认所有 checkpoint、日志和 W&B offline
目录已经同步完成。

## Verified clean rebuild: instance 46693892

The remediated immutable `training` image was cold-pulled onto a new 400 GB disk
and tested on 2026-08-03 on one NVIDIA RTX PRO 6000 Blackwell GPU. Full evidence
and the comparison with the first audit are in
[`VASTAI_CLEAN_REBUILD_46693892.md`](VASTAI_CLEAN_REBUILD_46693892.md). The
original discovery audit remains in
[`VASTAI_AUDIT_46687432.md`](VASTAI_AUDIT_46687432.md).

Verified host/runtime combination:

```text
Host driver:       595.58.03
GPU:               RTX PRO 6000 Blackwell Max-Q, compute capability 12.0
Image CUDA/nvcc:   12.8 / 12.8.93
PyTorch:           2.7.0+cu128, including sm_120
cuDNN / NCCL:      9.7.1 / 2.26.2
```

Across the discovery and clean-rebuild audits, PyTorch FP32/BF16 kernels, cuDNN
forward/backward, a locally compiled `sm_120` CUDA extension,
Isaac/Vulkan/PhysX and a real two-update SONIC H20 PPO smoke all passed. The
clean-rebuild smoke used only dependencies baked into the new digest. This host
does not need its driver replaced for the current image.

### Current image behavior

Digest
`sha256:6ab2d3aac46c74bb97cae40611262c16163ef1b3aae9cf503ce7e97e2f6b7b59`
contains `h5py==3.13.0`; do not install a server-side overlay. It also defaults
Kit to `--portable-root /cache/isaac-portable` and bakes the non-root
ComputeCache, Omniverse data and log links. These settings remained effective
even when Vast bypassed the native entrypoint, and the prior package-local
read-only errors did not recur.

Raise the limits that the non-root session is allowed to raise:

```bash
ulimit -n 1048576
ulimit -s 65536
```

The tested Vast container retained a hard memlock limit of only 8 MiB. A
one-GPU smoke passed, but this is not acceptable evidence for production NCCL or
GPUDirect. Multi-GPU deployment must request unlimited memlock from the
container runtime/template.

### Vast SSH launch-mode caveat

Vast direct-SSH mode replaces the image entrypoint. Because this image does not
bake shared SSH host keys and disables root login, the Vast on-start hook must:

1. run `ssh-keygen -A`;
2. copy Vast's injected authorization to
   `/home/fangzhengtian/.ssh/authorized_keys` with user ownership and mode 0600;
3. start `/usr/sbin/sshd`;
4. preserve `ACCEPT_EULA=Y`, `PRIVACY_CONSENT=Y` and
   `SONIC_ENV=agile-sonic` for login sessions.

Do not bake host private keys into the image and do not enable SSH root login.
A Vast-specific private template/on-start hook is the appropriate solution.

### Source sync hygiene

Exclude generated package metadata in addition to normal large/local trees:

```text
**/*.egg-info/
**/*.dist-info/
**/__pycache__/
```

`gear_sonic/pyproject.toml` currently declares `numpy==1.26.4`, while Isaac Sim
5.1 pins `numpy==1.26.0`. SONIC runs correctly as mounted source, but transferring
a generated `gear_sonic.egg-info` makes `pip check` report this conflict.

The first H20 launch also downloaded about 190 MB of Kit extensions. Persist
the user Omniverse extension store, allow first-start egress, or precache the
exact training experience before using an offline server.
