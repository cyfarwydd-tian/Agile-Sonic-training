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
