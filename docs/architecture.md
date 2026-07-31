# Agile SONIC 容器架构与依赖决策

## 目标

本项目把 StarVLA 容器中成熟的服务器初始化顺序保留下来：

```text
CUDA -> apt -> 普通用户 -> SSH -> AWS/Aliyun CLI
     -> Miniforge -> SONIC environments -> entrypoint -> gosu
```

同时把镜像内容、运行目录和多卡门禁改成 SONIC 训练所需的结构。公开镜像只包含
环境和固定版本的公开上游源码；私有 `sonic-training`、数据、checkpoint 和凭据均在
运行时挂载。

## 为什么不能只有一个 Python 环境

SONIC 仓库本身已把 training、sim/teleop、data collection、camera 和 inference
拆成不同 venv。主要依赖存在以下硬冲突：

| 能力 | Python / Torch | 关键约束 |
| --- | --- | --- |
| Isaac 训练 | Python 3.11 / Torch 2.7 cu128 | NumPy 1.26.0、Packaging 23、Gymnasium 1.2.1 |
| LeRobot 数据 | Python 3.10 / Torch `<2.7` | Packaging `>=24.2`、Gymnasium 0.29.1 |
| 当前 Isaac-GR00T | Python 3.12 / Torch 2.9 cu128 | Triton 3.5、Gymnasium 1.2.2、较新 TensorRT |
| RoboSuite/RoboCasa | 独立 tools 环境 | MuJoCo 3.3.2、NumPy 1.26.4、CycloneDDS 0.10.2 |

因此 `full` 的含义是“一个镜像内聚合多个隔离环境”，而不是让 pip 忽略冲突。
默认环境始终是 `agile-sonic`；entrypoint 根据 `SONIC_ENV` 选择其他环境。

## 构建层次

```text
system
  └─ training
       └─ tensorrt
            └─ full-tools
                 └─ full-data
                      └─ full
```

- `training` 是正式多卡 PPO 训练的最小支持镜像。
- `tensorrt` 只增加导出/验证工具，不改变训练版本组合。
- `full` 为合作伙伴提供完整工具箱，但会明显增加下载和磁盘成本。
- 无 `--target` 构建时默认得到 `full`；生产训练可以明确拉取更小的
  `latest-training`。

## 上游源码策略

Isaac Lab、SONIC fallback、SMPLSim、smplx、CycloneDDS、RoboSuite、LeRobot 和
Isaac-GR00T 都固定到 commit，而不是在构建时跟随 branch HEAD。

Isaac Lab core、SMPLSim、smplx、RoboSuite 和 RoboCasa 使用 `.pth` 暴露固定源码，而不安装
它们有冲突的 distribution metadata。需要的运行依赖由各自环境的 constraints
显式安装，并在构建结束执行 `pip check` 和 import smoke。私有源码挂载路径
`/workspace/sonic-training` 位于 `PYTHONPATH` 首位，因此会覆盖镜像内的公开
fallback。

## 原 Docker 范式中需要修正的点

1. 不在镜像中设置默认密码，也不把 SSH key 或云密钥写进 layer。
2. sshd 默认不启动；只有提供 authorized key 或显式启用时才启动。
3. 下载的 Miniforge、AWS CLI、Aliyun CLI 使用固定版本和 SHA-256 校验。
4. 不再对 Miniforge、Isaac、源码或整个 workspace 做递归 `chown`，避免制造几十
   GB 的重复 layer；运行时只调整普通用户和小型可写目录。
5. 私有源码不再 `COPY .` 进镜像，避免每次代码变化使庞大依赖层失效。
6. 缓存、数据和训练输出固定挂载到 `/cache`、`/datasets`、`/runs`。数据默认只读。
7. 默认使用 key-only SSH、短期云身份或只读 secret file；兼容旧变量只为迁移。
8. 多卡训练先使用 Accelerate DDP，并在正式训练前执行 GPU、依赖、文件系统和 NCCL
   smoke。DeepSpeed/FSDP 要等 checkpoint 分片和恢复行为单独验证后再启用。

## 不默认烘焙的硬件组件

UltraLeap runtime 需要接受厂商许可并添加外部 apt 仓库；ROS Desktop 也不是 PPO
训练依赖且会显著扩大依赖求解面。它们不进入默认 training/full Python 环境。
XRoboToolkit 的仓库内 x86_64 SDK、Unitree SDK、camera 和 data Python 依赖可以由
`full` 提供；具体相机、VR/手部追踪设备仍必须在目标服务器做 USB/device mapping
和硬件 smoke。

## CI/CD 边界

Pull request 在标准 GitHub runner 上完成 shell、Compose 和 Dockerfile 静态检查。
镜像发布由 push/tag 触发并写入 GHCR。标准 public Linux runner 的可用 SSD 很小，
完整 Isaac/full 镜像应通过 `DOCKER_RUNNER` 指向大盘 self-hosted runner；否则触发
成功并不等于远程构建具备足够磁盘。

GHCR 对单个 layer 有 10 GB 大小限制，并对单次上传设有 10 分钟超时。训练环境、
Torch、Isaac Sim 主组件、Isaac extension cache 和项目依赖必须分层安装；首次真实
构建还要检查 manifest 中的 layer blob 大小，不能只依赖 Dockerfile 静态检查。

每个发布镜像都应记录 source revision、目标名称、digest、SBOM/provenance，以及
各环境实际安装清单。合作伙伴部署时使用 digest，而不是可变的 `latest`。

## H20 支持边界

Isaac Sim 5.1 的官方 requirements 页面现已标记为停止支持；官方列出的测试 GPU
是 RTX 系列，并明确将没有 RT Core 的 A100/H100 列为不支持。H20 没有出现在该
版本的支持矩阵中。因此这里的 H20 + headless 方案是针对 SONIC 项目的工程验证
路径，不是 NVIDIA 官方兼容性声明。每个新驱动、镜像 digest 和服务器型号组合都
必须重新执行 preflight、NCCL、最小 PPO、checkpoint 恢复与稳定性测试。
