# LeRobot 框架与 pi0.5 使用指南

> 本文档面向已有 Python/PyTorch 基础的读者，重点解释 LeRobot 框架的设计逻辑和 pi0.5 模型的特点，而非重复介绍通用深度学习知识。
>
> 版本参考：lerobot==0.6.1，pi0.5（`lerobot/pi05`）

---

## 一、pi0.5 模型特点

### 1.1 架构概览

pi0.5 是一个 **VLA（Vision-Language-Action）模型**，由三个部分组成：

```
图像（多路摄像头）  ──→  SigLIP 视觉编码器  ──┐
语言指令（文本）   ──→  Gemma2 语言模型     ──→  Transformer 主干  ──→  Flow Matching 动作头  ──→  动作序列
机器人状态（关节角）──→  线性投影层         ──┘
```

- **视觉编码器**：SigLIP（Google，对比学习预训练），将每帧图像编码为 token 序列
- **语言主干**：Gemma2（2B 参数量级），理解任务指令并与视觉 token 交叉注意力
- **动作头**：Flow Matching（连续正则化流），从高斯噪声迭代去噪，生成动作分布的样本

Flow Matching 与 Diffusion Policy 的关系：两者思路相似（都是从噪声中恢复信号），Flow Matching 使用直线流轨迹，推理步数更少（通常 10 步，比 DDPM 的 100 步快得多）。

### 1.2 输入空间

| 输入键（LeRobot 约定） | 形状 | 说明 |
|----------------------|------|------|
| `observation.images.<camera_name>` | `(B, C=3, H=224, W=224)` | 每路摄像头一个键，RGB，已归一化到 `[-1, 1]` |
| `observation.state` | `(B, state_dim)` | 机器人本体感受，LIBERO 中 `state_dim=8`（7 个关节角 + 1 个夹爪宽度） |
| `task` | `list[str]`，长度 B | 自然语言任务描述，如 `"pick up the red cube"` |

摄像头名称由数据集决定，LIBERO 常见配置：
- `agentview_rgb`：第三人称全景摄像头
- `eye_in_hand_rgb`：腕部摄像头

### 1.3 输出空间：Action Chunk

pi0.5 **不输出单步动作**，而是一次性预测未来 N 步的动作序列，称为 **Action Chunk**：

```
输出形状：(B, chunk_size, action_dim)
```

- `chunk_size`（LeRobot 中对应 `n_action_steps`）：默认 **50**，即每次推理给出未来 50 步的动作
- `action_dim`：取决于机器人，LIBERO 为 **7**（6 维末端执行器位姿 + 1 维夹爪）

**为什么要用 Action Chunk？**

单步推理（观测→动作→观测→动作）存在两个问题：
1. 摄像头帧率（~30 Hz）远低于控制频率（~200 Hz），每步都要重新推理成本高
2. 模型在时序上只看到当前帧，难以规划需要多步协调的动作

Action Chunk 让模型一次规划 N 步，执行时只取前几步（Temporal Ensemble 策略），下一次推理时再更新——在延迟和平滑性之间取得平衡。

---

## 二、LeRobot 框架设计

### 2.1 核心抽象层次

```
LeRobotDataset          ← 数据层：统一格式加载各种机器人数据集
    ↓
Policy（如 PI05Policy）  ← 模型层：统一接口，屏蔽模型内部差异
    ↓
lerobot_train.py        ← 训练层：数据加载、优化、日志、checkpoint
    ↓
PeftConfig（可选）       ← 适配层：LoRA 等参数高效微调
```

### 2.2 Policy 统一接口

所有 LeRobot Policy 都继承自 `PreTrainedPolicy`，暴露以下方法：

```python
# 推理（不更新权重）
action = policy.select_action(batch: dict) -> Tensor

# 训练前向（计算 loss）
output = policy.forward(batch: dict) -> PolicyOutput

# 保存/加载（HuggingFace 标准接口）
policy.save_pretrained(path)
policy = Policy.from_pretrained(path_or_hub_id, torch_dtype=...)
```

`select_action` 和 `forward` 接受同一个 `batch` 字典，键名与数据集的观测/动作键一一对应。

### 2.3 配置系统（draccus）

LeRobot 用 `draccus` 库做配置管理，核心思想是**类型安全的 dataclass**——所有超参都是 Python 类，不是裸字典。

配置通过命令行参数传入，格式为 `--section.key=value`：

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=pi05 \
  --dataset.repo_id=lerobot/libero_spatial_no_noops \
  --training.num_steps=50000
```

也可以用 JSON 文件配合命令行覆盖（见下文训练脚本部分）。

---

## 三、训练脚本详解

### 3.1 `lerobot_train` 的核心参数

以下是与本项目相关的常用参数，按功能分组：

**数据集**
```bash
--dataset.repo_id=lerobot/libero_spatial_no_noops
# 从 HuggingFace Hub 下载 LIBERO 数据集（会缓存到 HF_HOME/hub/）

--dataset.image_transforms.enable=true
# 启用图像增强（随机裁剪、色彩抖动），有助于泛化
```

**模型**
```bash
--policy.type=pi05
# 指定使用 pi0.5 策略

--policy.path=lerobot/pi05
# 预训练权重来源（Hub ID 或本地路径）
```

**训练**
```bash
--training.num_steps=50000      # 总训练步数
--training.batch_size=4         # 每卡 batch size
--training.lr=1e-4              # 学习率
--training.grad_clip_norm=10    # 梯度裁剪（防止梯度爆炸）
--training.save_freq=5000       # 每 N 步保存一次 checkpoint
--training.log_freq=100         # 每 N 步打印一次日志
```

**精度（V100 关键）**
```bash
--training.mixed_precision=no   # V100 不支持 bf16，必须关闭
```
> 注意：mixed_precision 在训练脚本中控制，accelerate 中也需对应设置。两者都设为 `no` / `false`，以 accelerate 的设置为准（它是上层控制器）。

**LoRA**
```bash
--peft.method_type=LORA
--peft.r=32
--peft.lora_alpha=32
```

**输出**
```bash
--output_dir=outputs/pi05_lora_libero
--wandb.project=pi05_v100        # W&B 项目名
```

### 3.2 多卡训练：accelerate 的角色

```bash
accelerate launch \
    --num_processes=6 \        # 启动 6 个进程，分别对应 6 块 GPU
    --mixed_precision=no \     # 关闭混合精度（V100 fp32）
    -m lerobot.scripts.lerobot_train \
    # ... 以上 lerobot 参数
```

**accelerate 做了什么：**
1. 以 `torchrun` 方式启动多个进程，每个进程绑定一块 GPU
2. 自动包装模型为 `DistributedDataParallel (DDP)`
3. 自动同步梯度（每步结束后 all-reduce）
4. 统一管理 `optimizer.step()` 和 mixed precision

**DDP 数据并行的逻辑：**
- 每块 GPU 持有**完整的模型副本**
- `batch_size=4` 且 `num_processes=6` → 实际有效 batch size = **24**
- 梯度在所有 GPU 之间平均，等效于用 24 个样本计算一次梯度更新

---

## 四、LoRA 微调原理与配置

### 4.1 LoRA 在 pi0.5 中作用于哪里

LeRobot 的 PEFT 实现通过 HuggingFace `peft` 库，将 LoRA 应用于 **Transformer 主干的注意力层**（Query、Key、Value、Output 矩阵）。视觉编码器和动作头默认冻结。

### 4.2 关键超参解释

```json
{
  "peft": {
    "method_type": "LORA",
    "r": 32,
    "lora_alpha": 32
  }
}
```

- **`r`（rank，秩）**：LoRA 分解的低秩维度。r=32 表示在原始权重矩阵 W 旁边插入两个矩阵 A∈ℝ^{d×32} 和 B∈ℝ^{32×d}，权重更新 ΔW = BA。r 越大，表达能力越强，但训练参数越多。
- **`lora_alpha`**：实际缩放系数 = `lora_alpha / r`。`r=32, alpha=32` → 缩放 = 1.0。若设 `alpha=64, r=32` → 缩放 = 2.0，相当于放大学习率。通常先保持 `alpha == r`。

### 4.3 显存估算（V100 32GB，fp32）

| 阶段 | 估算显存占用/卡 |
|------|--------------|
| 模型参数（fp32） | ~14–16 GB |
| 梯度（仅 LoRA 参数，fp32） | ~0.5–1 GB |
| 优化器状态（Adam，LoRA 参数） | ~1–2 GB |
| Activation（batch=4，图像序列） | ~6–8 GB |
| **合计** | **~22–27 GB** |

batch=4 在 32 GB 卡上有一定余量；如果 OOM，先将 batch 降到 2。

---

## 五、环境变量参考

运行脚本前可能需要设置以下变量（在 shell 中 `export`，不要写入代码文件）：

| 变量 | 用途 | 示例 |
|------|------|------|
| `HF_TOKEN` | 下载 gated 模型（pi0.5 需要申请访问权限） | `export HF_TOKEN=hf_xxx` |
| `HF_HOME` | HuggingFace 缓存根目录（权重、数据集） | `export HF_HOME=/persistent/hf_cache` |
| `WANDB_API_KEY` | W&B 训练监控 | `export WANDB_API_KEY=xxx` |
| `CUDA_VISIBLE_DEVICES` | 限制使用哪些 GPU（如只用 GPU 0,1） | `export CUDA_VISIBLE_DEVICES=0,1` |

> `HF_HOME` 由 `install.sh` 写入 `~/.bashrc`（持久化）。
> `HF_TOKEN`、`WANDB_API_KEY` 含敏感信息，**只在 shell 会话中 `export`，不写进任何文件**。

`os.getenv` 的读取来源即进程启动时继承的 shell 环境，层级为：
`/etc/environment` → `~/.bashrc` → 当前 shell `export` → 命令行内联 `VAR=val python ...`（后者覆盖前者）。

---

## 六、数据集：LIBERO

LIBERO 在 HuggingFace Hub 上有多个子集，均以 LeRobot 格式存储：

| Hub ID | 内容 | 任务数 |
|--------|------|-------|
| `lerobot/libero_spatial_no_noops` | 同一物体，不同空间位置 | 10 |
| `lerobot/libero_object_no_noops` | 同一位置，不同物体 | 10 |
| `lerobot/libero_goal_no_noops` | 不同目标状态 | 10 |
| `lerobot/libero_90_no_noops` | 多样化任务 | 90 |

`no_noops` 版本过滤掉了无动作帧，训练效率更高，推荐使用。

LeRobot 数据集的观测键可以这样查看：

```python
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset("lerobot/libero_spatial_no_noops")
print(ds.features)          # 所有键和形状
print(ds.meta.robot_type)   # 机器人类型
print(ds.num_frames)        # 总帧数
```

---

## 七、快速命令速查

```bash
# 验证镜像环境
python /usr/local/share/v100-image/check_environment.py --require-gpu

# 查看 lerobot_train 所有参数
python -m lerobot.scripts.lerobot_train --help

# 查看 pi0.5 模型 config
python -c "
from lerobot.policies.pi05 import PI05Policy, PI05Config
cfg = PI05Config()
print(cfg)
"

# 查看数据集结构
python -c "
from lerobot.datasets import LeRobotDataset
ds = LeRobotDataset('lerobot/libero_spatial_no_noops')
print(ds.features)
"

# 启动训练（单卡，调试用）
CUDA_VISIBLE_DEVICES=0 python -m lerobot.scripts.lerobot_train \
  --policy.type=pi05 \
  --policy.path=lerobot/pi05 \
  --dataset.repo_id=lerobot/libero_spatial_no_noops \
  --training.num_steps=200 \
  --training.batch_size=2 \
  --training.mixed_precision=no \
  --peft.method_type=LORA \
  --peft.r=32 \
  --peft.lora_alpha=32 \
  --output_dir=outputs/debug_run
```
