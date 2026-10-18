# LeRobot 框架与 pi0.5 使用指南

> 面向已会 PyTorch 的读者，只讲 LeRobot 和 pi0.5 本身。
> 版本：`lerobot==0.6.1`，模型 `lerobot/pi05_libero_base`，数据集 `HuggingFaceVLA/libero`。
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
| `chunk_size` | 50 | 一次预测未来 50 步动作 |
| `n_action_steps` | 50 | 每个 chunk 实际执行多少步再重新推理 |
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
```
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

---

## 四、数据集：`HuggingFaceVLA/libero`

LeRobot 格式，公开（非 gated），合并了 LIBERO 的 4 个套件：

- 1693 个 episode，273,465 帧，10 fps
- features：`observation.images.image`（256×256×3）、`observation.images.image2`、`observation.state`（8）、`action`（7）、`task_index` 等

键名和 `pi05_libero_base` 的输入完全一致，所以不需要 rename。

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

```bash
lerobot-train --help                                   # 所有训练参数
STEPS=200 NUM_GPUS=1 bash scripts/3_train_lora.sh      # 冒烟测试
bash scripts/3_train_lora.sh                           # 6 卡正式训练
LOCAL_CHECKPOINT=outputs/<run>/checkpoints/last/pretrained_model python scripts/2_inference.py
```
