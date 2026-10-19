# π0.5 LoRA on LIBERO with 6× V100 (fp32)

在 6 张 Tesla V100-32GB 上，以 fp32 精度对 4.1B 参数的 VLA 模型 [π0.5](https://www.physicalintelligence.company/blog/pi05) 做 LoRA 微调。从通用预训练权重 `lerobot/pi05_base` 出发，在 [LIBERO](https://libero-project.github.io/) 仿真基准上评测成功率，并与官方全量微调的结果对比。

- V100 不支持 bf16，4.1B 模型在 fp32 下做全量微调，单卡放不下一份完整的模型和优化器状态（约 66 GB），所以只训练 LoRA。
- 框架：[LeRobot](https://github.com/huggingface/lerobot) 0.6.1；数据集：[`HuggingFaceVLA/libero`](https://huggingface.co/datasets/HuggingFaceVLA/libero)（4 个套件，1693 条示教）。

<!-- TODO: rollout GIF（每个套件一个） -->

## 结果

LIBERO 成功率（%），每个套件 10 个任务 × 50 个 episode。

| 方法 | 起点 | 可训练参数 | Spatial | Object | Goal | Long | 平均 |
|---|---|---|---|---|---|---|---|
| openpi 全量微调（官方报告） | pi05_base | 100% | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| LeRobot 全量微调（官方报告）¹ | pi05_libero | 100% | 97.0 | 99.0 | 98.0 | 96.0 | 97.5 |
| LeRobot 全量微调（本仓库评测）¹ | pi05_libero | 100% | TBD | TBD | TBD | TBD | TBD |
| **LoRA r=32，default（本仓库）** | pi05_base | TBD | TBD | TBD | TBD | TBD | TBD |

¹ `lerobot/pi05_libero_finetuned`：在 openpi 的 pi05_libero 基础上又全量微调 6k 步（8×H100，bf16）。用本仓库的评测流程评测它，并和官方报告的数字对比，以此检验评测流程本身，见[评测流程的锚点](#评测流程的锚点)。

### 消融：LoRA rank

LoRA 位置固定为 `default`。

| r | 可训练参数 | 显存 / 卡 | s / step | Spatial | Object | Goal | Long | 平均 |
|---|---|---|---|---|---|---|---|---|
| 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 消融：LoRA 作用位置

r = 32。

| 位置 | 覆盖的层 | LoRA 层数 | 可训练参数 | 显存 / 卡 | s / step | 平均 |
|---|---|---|---|---|---|---|
| `default` | action expert 的 q/v + 动作投影（LeRobot 默认） | 38 | TBD | TBD | TBD | TBD |
| `expert` | action expert 全部 attn + MLP + 动作/时间投影 | 130 | TBD | TBD | TBD | TBD |
| `all` | `expert` + PaliGemma 语言主干（视觉编码器冻结） | 256 | TBD | TBD | TBD | TBD |

`expert` 和 `all` 只差语言主干，这一对比用来回答"VLM 主干要不要一起适配"。

<!-- TODO: 结论（每张表 2–3 句），wandb 报告链接 -->

## 训练设置

| 项 | 值 |
|---|---|
| 硬件 | 6 × Tesla V100-SXM2-32GB，DDP |
| 精度 | fp32（`accelerate --mixed_precision=no`），gradient checkpointing |
| 起点 | `lerobot/pi05_base` |
| batch | 4 / 卡 × 6 = 24 |
| 步数 | TBD |
| 学习率 | 1e-4，warmup 1000 步，cosine 衰减 |
| LoRA | `lora_alpha = r` |
| 归一化 | state / action 用 LIBERO 数据集统计量做 QUANTILES 归一化 |
| 相机映射 | `image → base_0_rgb`，`image2 → left_wrist_0_rgb`，`right_wrist_0_rgb` 置空并 mask |

### 评测流程的锚点

社区里有好几例报告，说 LeRobot 下的 LIBERO 评测复现不出官方数字（从 0% 到 71% 都有）。所以先评测一个官方 checkpoint，确认本仓库的评测流程是可信的，然后才评测自己训的模型。

### 诊断：`pi05_libero_base` 为什么在 LeRobot 里评测出 0%

`lerobot/pi05_libero_base` 是 openpi 官方 `pi05_libero` 的 PyTorch 转换版，官方报告的平均成功率是 96.9%，但社区在 LeRobot 下评测只得到约 0%（[lerobot#2533](https://github.com/huggingface/lerobot/issues/2533)）。

**假设**：这个 checkpoint 的 processor 没有携带归一化统计量。

- `policy_preprocessor.json` / `policy_postprocessor.json` 里 normalizer 的 `features` 为空，也没有 stats 文件，`norm_map` 写的还是 `MEAN_STD`（pi0.5 用的是 `QUANTILES`）。
- LeRobot 的 normalizer 遇到没有统计量的键会原样返回。训练时 `lerobot-train` 会用数据集统计量覆盖，`lerobot-eval` 则不会。
- 结果是两头都错：state 没有归一化就被离散化成 256 档拼进 prompt；LIBERO 的 state 有一维原始值约为 3，超出 [-1, 1]，全部落进最后一档。模型输出的 action 也没有反归一化，直接当成真实动作执行。

**第二个不一致：prompt 格式。** openpi 训练 `pi05_libero` 时设置了 `discrete_state_input=False`（openpi `training/config.py`），所以 prompt 只有任务文本加一个换行，模型从没见过 state。LeRobot 的 pi0.5 processor 则固定拼成 `Task: <task>, State: <8 个数>;\nAction: `。用 openpi 自带的 `paligemma_tokenizer.model` 核对过：

| prompt | 示例任务的 token 数 |
|---|---|
| openpi `pi05_libero` | 14 |
| LeRobot 默认 | 55 |
| LeRobot 去掉 state 拼接 + 末尾换行 | 14（与 openpi 逐个 token 一致） |

仿真这一侧，LeRobot 和 openpi 是一致的：开局等待 10 步、图像旋转 180°、state 取 eef 位置 + axis-angle + 夹爪。

**验证**：权重不变，只重建 processor（[`scripts/attach_norm_stats.py`](scripts/attach_norm_stats.py)，构建方式与 `lerobot-train` 一致），一次只改一个变量：

| checkpoint | 统计量 | prompt | 开环 L1² | Spatial | Object | Goal | Long | 平均 |
|---|---|---|---|---|---|---|---|---|
| `pi05_libero_base`（原样） | 无 | LeRobot | TBD | TBD | TBD | TBD | TBD | TBD |
| + LIBERO 统计量 | ✓ | LeRobot | TBD | 87.6 | 90.6 | 80.6 | 25.2 | 71.0 |
| + LIBERO 统计量 + openpi prompt | ✓ | openpi | TBD | TBD | TBD | TBD | TBD | TBD |
| openpi 官方报告 | ✓ | openpi | — | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |

² 不跑仿真：在数据集的 200 帧上，比较预测的首步动作与示教动作，取逐维 L1 误差的均值（真实尺度）。

补上统计量后，平均成功率达到 71.0%，但 Long 只有 25.2%（10 个任务分别为 30 / 22 / 8 / 58 / 10 / 80 / 0 / 12 / 4 / 28）。

<!-- TODO: 结论。其余已知差异：openpi 每 5 步重新规划（这里 n_action_steps=10）；动作统计量来自 physical-intelligence/libero 而非 HuggingFaceVLA/libero；bf16 → fp32 -->

这一点对本仓库自己的实验也有影响：从 `pi05_base` 训练时，统计量会写进保存的 checkpoint，所以评测自己训的模型不受第一个问题影响。prompt 格式则在训练和评测时保持一致，都用 LeRobot 默认格式。

## 复现

运行环境（Python 3.12、PyTorch 2.7.1+cu118、LeRobot 0.6.1 + `[libero,peft,pi,training]`）由镜像提供；渲染用 OSMesa，环境问题见 [docs/troubleshooting.md](docs/troubleshooting.md)。

```bash
export HF_TOKEN=...          # google/paligemma-3b-pt-224 是 gated repo，需先在网页上申请访问
export WANDB_API_KEY=...

make check                   # 检查环境
make data                    # 下载 LIBERO 数据集（35 GB）
make eval                    # 评测官方 checkpoint（评测流程锚点）

make norm-stats                            # 诊断：pi05_libero_base + LIBERO 统计量
make norm-stats OPENPI_PROMPT=1 CHECK=200  # 再加 openpi prompt，并做三者开环对比
make eval POLICY_PATH=lerobot/pi05_libero_base NAME=libero_base_raw
make eval POLICY_PATH=outputs/ckpt/pi05_libero_base_with_stats NAME=libero_base_stats
make eval POLICY_PATH=outputs/ckpt/pi05_libero_base_openpi_prompt NAME=libero_base_openpi_prompt

make train JOB_NAME=r32_default                         # 主实验
make train JOB_NAME=r8_default  LORA_R=8                # rank 消融
make train JOB_NAME=r64_default LORA_R=64
make train JOB_NAME=r32_expert  LORA_TARGETS=expert     # 位置消融
make train JOB_NAME=r32_all     LORA_TARGETS=all

make eval POLICY_PATH=outputs/<run>/checkpoints/last/pretrained_model NAME=<run>
```

快速冒烟测试：`make smoke`（6 卡 20 步，只用 10 条示教）、`make eval SUITES=libero_spatial TASK_IDS=[0] N_EPISODES=2`。全部参数见 `make help`。

## 仓库结构

```
├── Makefile                  # 入口：check / data / infer / smoke / train / eval
├── install.sh                # 初始化：建目录、HF 缓存路径、可选 editable lerobot
├── scripts/
│   ├── 1_check_setup.py      # 环境检查
│   ├── 2_inference.py        # 单步推理（pre/post-processor 管线）
│   ├── 3_train_lora.sh       # LoRA 训练（rank / 作用位置 / 起点可调）
│   ├── 4_eval_libero.sh      # LIBERO 评测，每个套件占一张卡
│   ├── summarize_eval.py     # 汇总 eval_info.json → markdown 表格
│   └── attach_norm_stats.py  # 给缺统计量的 checkpoint 补上数据集统计量 + 开环检查
└── docs/
    ├── lerobot_pi05_guide.md # LeRobot 与 pi0.5 的使用笔记
    └── troubleshooting.md    # 安装 / 渲染问题记录
```

## 参考

- Physical Intelligence, *π0.5: a Vision-Language-Action Model with Open-World Generalization*, [arXiv:2504.16054](https://arxiv.org/abs/2504.16054)
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)，[huggingface/lerobot](https://github.com/huggingface/lerobot)
- Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, NeurIPS 2023
