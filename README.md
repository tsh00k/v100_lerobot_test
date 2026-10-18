# pi05_v100 — pi0.5 在 V100 上的推理与 LoRA 微调

## 项目背景

### 什么是 pi0.5？
π0.5 (pi0.5) 是 Physical Intelligence (π) 公司开发的机器人基础模型。
它是一个**视觉-语言-动作模型 (VLA)**：
- 输入：摄像头图像 + 机器人本体状态 + 自然语言指令
- 输出：机器人动作序列（关节角度 / 末端执行器位姿）

### 什么是 LoRA？
LoRA（Low-Rank Adaptation）是一种**参数高效微调方法**。
大模型有数十亿参数，直接全量微调需要巨大的显存和计算资源。
LoRA 的思路：冻结原始权重，只训练两个小矩阵 A 和 B（它们的乘积是权重更新量）。
- 训练参数量：原来的 ~0.1%–1%
- 显存占用：大幅降低
- 效果：接近全量微调

### 什么是 LIBERO？
LIBERO 是一个**机器人操作基准数据集**，包含多个任务套件：
- `libero_spatial`：考验空间感知（同一物体，不同位置）
- `libero_object`：考验物体泛化（同一位置，不同物体）
- `libero_goal`：考验目标泛化
- `libero_90`：90 个不同任务

---

## 项目结构

```
pi05_v100/
├── .gitignore               # git 忽略规则
├── README.md                # 本文件
├── Makefile                 # 快捷入口：make check / infer / smoke / train
├── install.sh               # 项目初始化（建目录、HF 缓存指向持久化存储、可选 editable lerobot）
├── docs/
│   ├── lerobot_pi05_guide.md   # LeRobot 框架与 pi0.5 使用说明
│   └── proxy_and_network.md    # 反向隧道 / 代理配置
└── scripts/
    ├── 1_check_setup.py     # 步骤一：验证环境
    ├── 2_inference.py       # 步骤二：加载 pi0.5，走 pre/post-processor 跑推理
    └── 3_train_lora.sh      # 步骤三：多卡 LoRA 微调（超参写在脚本里，环境变量可覆盖）
```

运行环境（Python / PyTorch / LeRobot 0.6.1 / peft）由 `v100_mirror` 镜像提供，本仓库不负责装依赖。

> 参考项目结构来自：
> - [huggingface/lerobot](https://github.com/huggingface/lerobot)（`src/lerobot/scripts/`、`examples/`）
> - [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)（`scripts/`、`examples/libero/`）

---

## 典型深度学习项目架构说明

| 目录/文件 | 作用 | 类比 |
|-----------|------|------|
| `Dockerfile` / `pyproject.toml` | 声明依赖，确保可复现（本项目在镜像里） | 食谱的"原材料清单" |
| `configs/` 或启动脚本里的参数 | 超参数，与代码分离 | 旋钮面板，不改代码就能调参 |
| `scripts/` | 对外暴露的入口脚本 | 程序的"操作手册" |
| `src/` 或主库（LeRobot） | 核心模型、数据、训练逻辑 | 引擎 |
| `outputs/` | 训练产物（不提交 git） | 实验记录 |

---

## 快速开始

> 前提：已挂载镜像并 SSH 进入容器。`conda activate lerobot` 已在 `.bashrc` 中自动生效。
> 镜像已包含 Python 3.12、PyTorch 2.7.1+cu118、lerobot 0.6.1、peft 等全套依赖。

### 1. 克隆本项目并初始化
```bash
git clone <this-repo> && cd pi05_v100

# 基础初始化（建 outputs/ 和 data/ 目录）
bash install.sh

# 可选：把 HF 模型/数据集缓存指向持久化存储（强烈建议，避免重启后重新下载）
HF_CACHE_DIR=/persistent/hf_cache bash install.sh

# 可选：同时切换 lerobot 为 editable 安装（方便阅读和修改 LeRobot 源码）
HF_CACHE_DIR=/persistent/hf_cache EDITABLE_LEROBOT=1 bash install.sh
```

### 2. 验证环境
```bash
python scripts/1_check_setup.py
```

### 3. 运行推理（验证模型可正常加载）
```bash
python scripts/2_inference.py
```

### 4. 启动 LoRA 微调
```bash
# 可选：设置 W&B 密钥（用于训练监控）
export WANDB_API_KEY=your_key_here

bash scripts/3_train_lora.sh
```

---

## 关于 V100 和 fp32

Tesla V100 **不支持 bfloat16 (bf16)**（Ampere 架构之后才有），只支持 fp32 和 fp16。
新一代 A100/H100 常用 bf16 进行训练（比 fp32 快、比 fp16 数值更稳定）。
本项目统一使用 **fp32**，牺牲一定速度换取数值稳定性和 V100 兼容性。
