# LeRobot 框架与 pi0.5 使用指南

> 面向已会 PyTorch 的读者，只讲 LeRobot 和 pi0.5 本身。
> 版本：`lerobot==0.6.1`，模型 `lerobot/pi05_base`（LoRA 起点）/ `lerobot/pi05_libero_base`（诊断对象），数据集 `HuggingFaceVLA/libero`。
> 文中参数名、默认值均对照服务器上安装的 0.6.1 源码核实；源码路径前缀 `P=/opt/conda/envs/lerobot/lib/python3.12/site-packages/lerobot`。
> 遇到不确定的接口，**先查源码或 `--help`，不要猜**。

---

## 一、pi0.5 模型

### 1.1 结构

```
图像 (多相机) ─▶ SigLIP 视觉编码器 ─┐
prompt tokens ─────────────────────▶ PaliGemma (gemma_2b) ─▶ KV cache ─┐
                                                                         ▼
                          噪声动作 + 时间步 ─▶ action expert (gemma_300m) ─▶ 速度场 ─▶ 迭代去噪 ─▶ action chunk
```

- **VLM 主干**：PaliGemma（`paligemma_variant='gemma_2b'`），处理图像和语言。
- **Action expert**：一个独立的小 Gemma（`action_expert_variant='gemma_300m'`），和主干共享注意力，专门输出动作。
- **动作头**：Flow Matching。推理时从高斯噪声出发，积分 `num_inference_steps=10` 步得到动作。
- 总参数量 **4.14B**；fp32 下光权重就占约 16.5 GB。

pi0.5 和 pi0 的一个关键区别：**state 不是用连续向量输入的，而是离散化成文本 token 拼进 prompt**（见 2.3）。

### 1.2 关键配置（`PI05Config`，`P/policies/pi05/configuration_pi05.py`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `chunk_size` | 50 | 一次预测未来多少步动作（openpi 里叫 `action_horizon`）。本仓库 LoRA 用 **10** |
| `n_action_steps` | 50 | 每个 chunk 实际执行多少步再重新推理。本仓库 LoRA 用 **5**；必须 ≤ `chunk_size` |
| `max_state_dim` / `max_action_dim` | 32 / 32 | 内部统一 pad 到 32 维（LIBERO 实际 state 8 维、action 7 维） |
| `num_inference_steps` | 10 | flow matching 去噪步数 |
| `image_resolution` | (224, 224) | 图像在模型内部 resize+pad 到这个尺寸 |
| `tokenizer_max_length` | 200 | prompt 最大 token 数 |
| `dtype` | `float32` | V100 保持 float32 |
| `gradient_checkpointing` | False | 训练时建议打开，省显存 |
| `freeze_vision_encoder` / `train_expert_only` | False / False | 全量微调时可用来冻结部分模块 |
| `optimizer_lr` | 2.5e-5 | 默认学习率（全量微调用的） |
| `normalization_mapping` | VISUAL: IDENTITY，STATE/ACTION: QUANTILES | state/action 用分位数归一化到 [-1, 1] |
| `push_to_hub` | **True** | 不关的话，没设 `repo_id` 时训练启动会直接报错 |

### 1.3 输入输出（以 `pi05_libero_base` 为例）

运行 `scripts/2_inference.py` 会把这些打印出来（`policy.config.input_features` / `output_features`）：

| 键 | 形状 | 说明 |
|---|---|---|
| `observation.images.image` | (3, 256, 256) | 第三人称相机 |
| `observation.images.image2` | (3, 256, 256) | 腕部相机 |
| `observation.images.empty_camera_0` | (3, 224, 224) | 占位相机，不提供时模型会自动填空图 |
| `observation.state` | (8,) | 本体状态 |
| `task` | str | 自然语言任务 |
| **输出** `action` | (7,) | 末端位姿增量 6 维 + 夹爪 1 维 |

### 1.4 Action chunk 与 `select_action`

- `policy.predict_action_chunk(batch)` → `(B, 50, 7)`：一次前向得到完整 chunk。模型内部按 32 维计算，返回前已经裁回真实的 action 维度，但**还没有反归一化**。
- `policy.select_action(batch)` → `(B, 7)`：内部维护一个动作队列，队列空了才推理一次 chunk，之后每次调用弹出一步。所以在 50 步里只有第 1 步真正跑了模型。
- 每个 episode 开始前要调用 `policy.reset()` 清空队列。

---

## 二、LeRobot 的三段式推理管线

```
raw frame ──preprocessor──▶ batch ──policy──▶ action ──postprocessor──▶ 真实尺度动作
```

### 2.1 为什么不能直接把张量喂给 policy

policy 只认**已经处理好**的 batch：state 已归一化、prompt 已分词、数据已经在 GPU 上。这些处理都由 processor pipeline 完成，并且和 checkpoint 一起保存（`policy_preprocessor.json` / `policy_postprocessor.json`，归一化统计量也在里面）。

```python
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy

policy = PI05Policy.from_pretrained(ckpt).to("cuda").eval()
pre, post = make_pre_post_processors(
    policy.config, pretrained_path=ckpt,
    preprocessor_overrides={"device_processor": {"device": "cuda"}},
)
batch = pre(frame)            # frame：无 batch 维的原始观测 + "task"
action = post(policy.select_action(batch))
```

### 2.2 pi0.5 preprocessor 的步骤（`P/policies/pi05/processor_pi05.py`）

1. `rename_observations`：键名对齐
2. `add_batch_dim`：加 batch 维
3. `RelativeActionsProcessorStep`：可选，相对动作（默认关闭）
4. `normalize`：state/action 做 QUANTILES 归一化
5. `Pi05PrepareStateTokenizerProcessorStep`：state 离散化成 256 档，拼出 prompt
6. `TokenizerProcessorStep`：用 **`google/paligemma-3b-pt-224`** 的 tokenizer 分词 → `observation.language.tokens` / `attention_mask`
7. `to_device`

postprocessor：`unnormalize` → `AbsoluteActions` → `to_cpu`。

### 2.3 pi0.5 的 prompt 格式

```
Task: pick up the black bowl and place it on the plate, State: 128 131 97 255 0 12 200 64;
Action: 
```

state 归一化到 [-1, 1] 后用 `np.digitize` 分成 256 档，再写成文本。所以 **normalize 必须排在 tokenizer 前面**，手动 tokenize 一定会出错。

### 2.4 原始 frame 的格式要求

- 图像：float，CHW，值域 **[0, 1]**（模型内部会转成 [-1, 1]）
- state：真实物理量（不要自己归一化）
- `task`：字符串
- 不要自己加 batch 维

### 2.5 gated 依赖

`google/paligemma-3b-pt-224` 是 gated repo（需要人工审批）。推理和训练都要用它的 tokenizer，所以要先在网页上申请访问，并且 `export HF_TOKEN=...`。

---

## 三、训练：`lerobot-train`

入口：`lerobot-train` 等价于 `python -m lerobot.scripts.lerobot_train`。
配置类：`TrainPipelineConfig`（`P/configs/train.py`），由 draccus 解析，命令行写法是 `--a.b=value`。

### 3.1 参数分组（真实字段名）

**顶层**（注意：没有 `training.` 前缀）
```
--batch_size=4          # 每卡
--steps=30000
--save_freq=5000
--log_freq=50
--num_workers=4
--seed=1000
--output_dir=...        # 必须是不存在的目录，否则报 FileExistsError（除非 --resume=true）
--job_name=...
--resume=false
```

**dataset**
```
--dataset.repo_id=HuggingFaceVLA/libero
```

**policy**（`--policy.path` 加载预训练模型，其余 `--policy.*` 覆盖它的 config）
```
--policy.path=lerobot/pi05_libero_base
--policy.dtype=float32
--policy.gradient_checkpointing=true
--policy.optimizer_lr=1e-4
--policy.scheduler_decay_steps=30000
--policy.push_to_hub=false
--policy.chunk_size=10
--policy.n_action_steps=5
```
`chunk_size` 同时决定训练时每个样本取多少步动作（`action_delta_indices = range(chunk_size)`），并和 `n_action_steps` 一起写进 checkpoint 的 `config.json`，评测时自动沿用，不需要再像诊断 `pi05_libero_base` 时那样用 `EXTRA_ARGS` 覆盖。

`use_policy_training_preset=True`（默认）表示优化器和学习率调度直接用 policy config 里的 `optimizer_*` / `scheduler_*`，所以学习率要通过 `--policy.optimizer_lr` 改。

**peft**（`PeftConfig`，`P/configs/default.py`）
```
--peft.method_type=LORA
--peft.r=32
--peft.lora_alpha=32      # 默认 None → peft 库默认 8，r=32 时缩放只有 0.25，建议显式设成 r
--peft.target_modules=... # 不填就用 pi05 的默认值，见 3.3
```

**wandb**
```
--wandb.enable=false
```

精度：**不由 lerobot 参数控制**，由 `accelerate launch --mixed_precision=no` 控制（训练循环里用的是 `accelerator.autocast()`）。

### 3.2 多卡：accelerate

```bash
accelerate launch --multi_gpu --num_processes=6 --mixed_precision=no \
    -m lerobot.scripts.lerobot_train  <上面的参数>
```
- 每张卡一个进程，各自持有一份完整模型（DDP）
- 有效 batch = `batch_size × num_processes`
- lerobot 内部会创建 `Accelerator`，用它做 backward、梯度裁剪和 autocast

### 3.3 pi0.5 默认的 LoRA 作用位置

`PI05Policy._get_default_peft_targets()`（`modeling_pi05.py`）：

```python
target_modules = r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))"
```

也就是说默认**只在 action expert 的 q/v 投影和动作相关的投影层上加 LoRA**，PaliGemma 主干和视觉编码器保持冻结。想让主干也参与适配，可以用 `--peft.target_modules` 自己写正则。

### 3.4 显存（V100 32GB，fp32）

- 权重约 16.5 GB/卡，这部分是固定开销
- LoRA 参数和对应的优化器状态都很小
- 剩下的显存主要用在激活值上：`gradient_checkpointing=true` 能显著降低，`batch_size` 从 4 开始试，OOM 就减半
- 以上是估算，实际以冒烟测试的 `nvidia-smi` 为准

### 3.5 chunk_size 为什么用 10（源码确认 + 推断）

- openpi 的 `Pi0Config` 默认 `action_horizon=50`（`src/openpi/models/pi0_config.py`），LeRobot 的 `pi05_base` 配置也是 `chunk_size=50`，所以 `pi05_base` 很可能是按 50 预训练的（推断：源码只有默认值）。
- openpi 官方的 `pi05_libero` 从 `pi05_base` 权重出发，显式设成 `action_horizon=10` 微调，评测时 `replan_steps=5`（`src/openpi/training/config.py`、`examples/libero/main.py`）。
- chunk 长度不影响任何权重的形状，所以微调时改短没问题。本仓库照搬 10 / 5，方便与 openpi 官方结果（96.9）对比；LeRobot 锚点用的是 50。

### 3.6 本仓库的训练脚本 `scripts/3_train_lora.sh`

脚本把上面的参数封装成环境变量（`make train` / `make smoke` 的命令行变量会原样传进去）：

| 变量 | 默认 | 对应参数 / 含义 |
|---|---|---|
| `NUM_GPUS` | 6 | `accelerate --num_processes` |
| `BATCH_SIZE` | 4 | `--batch_size`（每卡） |
| `STEPS` | 30000 | `--steps`，同时设 `--policy.scheduler_decay_steps` |
| `LR` | 1e-4 | `--policy.optimizer_lr` |
| `LORA_R` | 32 | `--peft.r`，`lora_alpha` 取同值 |
| `LORA_TARGETS` | default | `default` / `expert` / `all`，见脚本注释 |
| `CHUNK_SIZE` / `N_ACTION_STEPS` | 10 / 5 | `--policy.chunk_size` / `--policy.n_action_steps` |
| `POLICY_PATH` | lerobot/pi05_base | `--policy.path`；为 pi05_base 时自动加相机 rename |
| `EPISODES` | 全部 | `--dataset.episodes`，如 `[0,1,2]` |
| `SAVE_FREQ` / `LOG_FREQ` | 5000 / 50 | 保存与日志间隔 |
| `WANDB` | true | 需要 `WANDB_API_KEY` |
| `MONITOR` | cpu（smoke 为 full） | 资源监控，见下 |

**耗时与资源监控**：结果写在 `<OUTPUT_DIR>.monitor/`（`OUTPUT_DIR` 必须由 lerobot 自己创建，所以放在同级目录）。训练结束自动运行 `scripts/summarize_train.py` 打印并写入 `summary.md`。

- `train.log`：lerobot 的完整输出。其中 `updt_s` / `data_s` / `mem_gb` 是 lerobot 自带的，取所有 rank 的最大值。
- `MONITOR=full`：每 5 秒采一次 nvidia-smi（每张卡的显存和利用率）、CPU 利用率、负载、主机内存。
- `MONITOR=cpu`：每 30 秒只采 CPU 和主机内存。GPU 情况 wandb 的 system 面板已经有了。
- `MONITOR=off`：只保留 `train.log` 和墙钟时间。
- `MON_INTERVAL` 可改采样间隔，`TARGET_STEPS`（默认 30000）用于估算正式训练耗时。
- 主机指标是整机的，包括其他用户的进程。

---

## 四、数据集：`HuggingFaceVLA/libero`

LeRobot 格式，公开（非 gated），合并了 LIBERO 的 4 个套件：

- 1693 个 episode，273,465 帧，10 fps
- features：`observation.images.image`（256×256×3）、`observation.images.image2`、`observation.state`（8）、`action`（7）、`task_index` 等

键名和 `pi05_libero_base` 的输入完全一致，所以不需要 rename。`pi05_base` 的相机键名是 `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`，训练脚本会自动加 `--rename_map`：`image → base_0_rgb`，`image2 → left_wrist_0_rgb`，缺失的右腕相机由模型填空图并 mask。

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("HuggingFaceVLA/libero")
print(ds.meta.features)
print(ds[0]["task"])
```

> 注：网上常见的 `lerobot/libero_spatial_no_noops` 等名字在 Hub 上不存在。

---

## 五、环境变量

| 变量 | 用途 |
|---|---|
| `HF_HOME` | HF 缓存根目录，模型在 `$HF_HOME/hub`，数据集在 `$HF_HOME/lerobot` 或 hub 下 |
| `HF_TOKEN` | gated repo（paligemma）需要；只在会话里 export |
| `HTTPS_PROXY` | 走反向隧道，见 `proxy_and_network.md` |
| `HF_HUB_OFFLINE=1` | 文件都已缓存后可以设上，禁止联网检查，启动更快 |
| `CUDA_VISIBLE_DEVICES` | 限定用哪几张卡 |

---

## 六、常用命令

都在服务器上、在 tmux 里运行。先准备环境：

```bash
export PATH=/opt/conda/envs/lerobot/bin:$PATH
export HF_HOME=/root/data1/tsh/hf_cache
export HF_TOKEN=...            # paligemma tokenizer
export WANDB_API_KEY=...
make net-check                 # 需要联网时先检查隧道；全部缓存好后可改用 export HF_HUB_OFFLINE=1
```

**第 0 步：冒烟测试，测每步耗时**（6 卡 20 步、前 10 个 episode、`MONITOR=full`）

```bash
make smoke
# 想让估算更准，可以多跑几步，并按计划的正式步数估算：
make smoke STEPS=60 TARGET_STEPS=30000
```

结束时会打印 `s/step` 表格（也在 `outputs/pi05_smoke_*.monitor/summary.md`）。用稳态 s/step 定正式训练的步数预算；改了 `BATCH_SIZE` 或 `LORA_TARGETS` 要重新测，因为每步耗时会变。

**第 1 步：主实验**（从 `pi05_base` 出发，r=32，默认作用位置，chunk 10 / 5）

```bash
make train JOB_NAME=r32_default                         # 步数默认 30000；按 smoke 的结果用 STEPS=... 调整
```

**第 2 步：消融**（其余参数与主实验相同）

```bash
make train JOB_NAME=r8_default  LORA_R=8                # rank
make train JOB_NAME=r64_default LORA_R=64
make train JOB_NAME=r32_expert  LORA_TARGETS=expert     # 作用位置
make train JOB_NAME=r32_all     LORA_TARGETS=all
```

**每次训完：评测**（chunk / 执行步数从 checkpoint 自动读取）

```bash
make eval POLICY_PATH=outputs/<run>/checkpoints/last/pretrained_model NAME=<run>
LOCAL_CHECKPOINT=outputs/<run>/checkpoints/last/pretrained_model python scripts/2_inference.py   # 单步推理检查
```

其他：

```bash
lerobot-train --help                                    # 全部训练参数
STEPS=200 NUM_GPUS=1 bash scripts/3_train_lora.sh       # 不经过 make，单卡排查
make train ... MONITOR=off                              # 关闭资源采样
make train ... CHUNK_SIZE=50 N_ACTION_STEPS=10          # 改回 LeRobot 默认的 chunk（用于对照）
```

---

## 附录：背景概念

**LoRA**：冻结原始权重 W，只训练低秩增量 ΔW = BA（A ∈ ℝ^{r×d}，B ∈ ℝ^{d×r}），缩放系数为 `lora_alpha / r`。可训练参数通常只占原模型的约 0.1%–1%，优化器状态也随之大幅减少。

**LIBERO 套件**：`libero_spatial`（物体相同、摆放位置不同）、`libero_object`（摆放相同、物体不同）、`libero_goal`（场景相同、目标不同）、`libero_10`（长时序任务，也叫 Long）。

**V100 与 fp32**：V100（Volta 架构）没有 bf16，只有 fp32 和 fp16；fp16 的数值范围窄，训练容易溢出。本项目统一用 fp32，牺牲速度换取稳定性。

**一般深度学习项目的分层**

| 目录 / 文件 | 作用 |
|---|---|
| `Dockerfile` / `pyproject.toml` | 声明依赖，保证可复现（本项目由镜像提供） |
| `configs/` 或启动脚本里的参数 | 超参数，与代码分离 |
| `scripts/` | 对外的入口脚本 |
| `src/` 或主库（这里是 LeRobot） | 模型、数据、训练逻辑 |
| `outputs/` | 训练产物，不提交 git |
