# π0.5 的 state 输入：openpi 与 LeRobot 的差异

> 对照版本：openpi commit `215abfb`，lerobot 0.6.1。文中行号都基于这两个版本。
> 标注说明：**[源码]** 指读代码确认过；**[实测]** 指在服务器上跑过；**[推断]** 指没有直接证据。

## 1. 模型拿到 state 的两种方式

pi0 系列模型有两种获取机器人本体状态（state，LIBERO 中为 8 维：eef 位置 3 维 + axis-angle 3 维 + 夹爪 2 维）的方式：

| 方式 | 做法 | 使用者 |
|---|---|---|
| 连续向量 | `state_proj` 线性层把 state 投影成一个 token，送进 action expert | pi0 |
| 离散文本 | 归一化到 [-1, 1] 后分成 256 档，写成文字拼进 prompt：`Task: <task>, State: 12 200 ...;\nAction: ` | pi0.5（可关闭） |

pi0.5 **没有**连续 state 这个入口：

- **[源码]** openpi `models/pi0.py:93-97`：`state_proj` 只在 `if config.pi05: ... else:` 的 else 分支里创建；`:151` 用 state token 的代码包在 `if not self.pi05:` 里；`:229` 只是从 state 的形状里取 batch size。PyTorch 版 `models_pytorch/pi0_pytorch.py:103-107`、`:244` 写法相同。
- **[实测]** `lerobot/pi05_libero_base` 的权重里没有 `state_proj`，与动作相关的只有 `action_in_proj`、`action_out_proj`、`time_mlp_in`、`time_mlp_out`。

所以对 pi0.5 来说，**state 唯一的入口就是 prompt**。prompt 里没有 state，模型就完全看不到 state。

## 2. openpi：默认带 state，唯独 `pi05_libero` 关掉

- **[源码]** `models/pi0_config.py:33`：`discrete_state_input` 字段的注释写着 "not used directly by the model, but it is read by the ModelTransformFactory"。`:40-41`：没有显式设置时，取值等于 `pi05`，也就是 pi0.5 默认带 state。
- **[源码]** `training/config.py:134` 把这个值传给 `TokenizePrompt`。`transforms.py` 的 `TokenizePrompt` 在它为 `False` 时直接令 `state = None`。
- **[源码]** `models/tokenizer.py` 的 `tokenize`：
  - 有 state：`"Task: {text}, State: {s};\nAction: "`，前面加 BOS
  - 无 state：BOS + 任务文本 + 单独编码的 `"\n"`
- **[源码]** `training/config.py` 里所有 `pi05=True` 的配置中，**只有 `pi05_libero`（`:745`）显式写了 `discrete_state_input=False`**。`pi05_aloha`、`pi05_droid`、`pi05_droid_finetune`、`pi05_full_droid_finetune`、`pi05_aloha_pen_uncap` 和 polaris 的 droid 配置都用默认值（带 state）。
- **[源码]** openpi 的 LIBERO 评测脚本（`examples/libero/main.py`）会把 `observation/state` 发给 policy，但 `TokenizePrompt` 会把它丢掉。所以 `pi05_libero` 在**训练和评测时都不带 state**。

**未知**：
- openpi 没有公开预训练代码，`pi05_base` 预训练时 prompt 里是否带 state 无法从源码确认。**[推断]** 其他微调配置都默认带，说明大概率带。
- 为什么 LIBERO 要关掉，源码里没有注释说明，也没在 issue 里找到解释。**[推断]** 可能的原因：LIBERO 中机械臂和夹爪始终在两路相机画面里；去掉 state 可以避免模型依赖本体状态走捷径。

## 3. LeRobot：写死带 state，但跟 checkpoint 走

- **[源码]** `PI05Config` 里没有与 `discrete_state_input` 对应的字段。
- **[源码]** `policies/pi05/processor_pi05.py` 的 `make_pi05_pre_post_processors` 总会在 normalize 之后插入 `Pi05PrepareStateTokenizerProcessorStep`，也就是把 state 拼进 prompt。
- **[源码]** processor 的步骤列表保存在 checkpoint 的 `policy_preprocessor.json` 里，统计量保存在旁边的 `.safetensors` 中。加载时按 JSON 重建：
  - `lerobot-eval`（`scripts/lerobot_eval.py:771-774`）只覆盖 `device_processor` 和 `rename_observations_processor`
  - `lerobot-train`（`scripts/lerobot_train.py:357-376`）覆盖 device、rename，以及 normalizer 的 features / norm_map / stats（非 resume 时）
  - 两者都**不改步骤列表**
- **[实测]** `lerobot/pi05_base` 在 Hub 上的 `policy_preprocessor.json` 里含有 `pi05_prepare_state_tokenizer_processor_step`。`lerobot/pi05_libero_base` 也一样。

## 4. 不一致带来的影响

`lerobot/pi05_libero_base` 是 openpi `pi05_libero` 的转换版，训练时不带 state，但 LeRobot 给它配的 processor 会带 state：

| prompt | 示例任务 token 数 |
|---|---|
| openpi `pi05_libero` | 14 |
| LeRobot 默认（带 state） | 55 |
| LeRobot 去掉 state 步骤 + 末尾补换行 | 14，与 openpi 逐 token 一致 |

**[实测]** 示例任务："put both the alphabet soup and the tomato sauce in the basket"。openpi 一侧用它自带的 `paligemma_tokenizer.model`（sentencepiece）编码。

评测时，模型会看到 41 个训练时从没见过的 token。

**[实测]** 只补统计量、保留 LeRobot 默认 prompt 时，LIBERO 平均成功率 71.0%（Spatial 87.6 / Object 90.6 / Goal 80.6 / Long 25.2），openpi 官方报告的是 96.9%。改用 openpi prompt 后的结果：TBD。

## 5. 对本仓库实验的影响

- 从 `pi05_base` 做 LoRA：训练和评测都用 checkpoint 里保存的同一份 processor（带 state），**格式一致，不需要改代码**。
- 想复现 openpi 的做法（不带 state）：用 `scripts/attach_norm_stats.py --openpi-prompt` 的方式，另存一份去掉该步骤的 processor。它只改 checkpoint 里的 JSON，不改 LeRobot 源码。**[实测]** 重新加载后步骤列表和 token 数都正确。
- 可做的消融：从 `pi05_base` 出发，比较 LoRA 微调**带 state**（LeRobot 默认）和**不带 state**（openpi 对 LIBERO 的做法）的效果。
