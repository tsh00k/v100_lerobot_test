"""
2_inference.py —— 加载 pi0.5，用 LIBERO 格式的观测做推理

【学习要点】
- VLA 模型的"观测空间"和"动作空间"是什么
- 如何从 HuggingFace Hub 加载预训练模型
- 推理时为什么要用 torch.no_grad()
- 什么是 batch dimension（批维度）

【运行方式】
  python scripts/2_inference.py

  # 指定模型来源（默认使用 HuggingFace Hub 上的官方权重）
  HF_MODEL_ID=lerobot/pi05 python scripts/2_inference.py

  # 使用本地已下载的权重（离线环境）
  LOCAL_CHECKPOINT=outputs/my_lora_checkpoint python scripts/2_inference.py
"""

import os
import torch
from transformers import AutoTokenizer

# ——— 可配置项（通过环境变量覆盖）———
MODEL_ID        = os.getenv("HF_MODEL_ID", "lerobot/pi05_libero_base")
LOCAL_CKPT      = os.getenv("LOCAL_CHECKPOINT", "")   # 如果填了，优先使用本地权重
DEVICE          = os.getenv("DEVICE", "cuda:0")
DTYPE_STR       = os.getenv("DTYPE", "float32")       # V100 用 float32
TASK            = os.getenv("TASK", "pick up the red block and place it in the bin")


def load_policy_and_tokenizer(model_id: str, local_ckpt: str, device: str, dtype: torch.dtype):
    """
    加载 pi0.5 策略模型和对应的 tokenizer。

    【为什么需要 tokenizer】
    pi0.5 是语言条件模型（VLA），推理时必须提供 tokenized 的任务描述。
    tokenizer 把自然语言字符串（"pick up the red block"）转成 token id 张量，
    模型才能理解任务意图。

    【tokenizer 从哪里加载】
    pi0.5 基于 PaliGemma，tokenizer 文件随模型权重一起缓存在 HF_HOME/hub/ 下，
    local_files_only=True 确保不联网，直接读缓存。
    """
    from lerobot.policies.pi05 import PI05Policy

    checkpoint = local_ckpt if local_ckpt else model_id
    print(f"加载模型：{checkpoint}")
    print(f"  设备：{device}，精度：{dtype}")

    policy = PI05Policy.from_pretrained(checkpoint, torch_dtype=dtype)
    policy = policy.to(device).eval()

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  参数量：{n_params / 1e9:.2f}B")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    print(f"  tokenizer 加载完成，vocab_size={tokenizer.vocab_size}")

    return policy, tokenizer


def make_dummy_libero_observation(tokenizer, task: str, device: str, dtype: torch.dtype) -> dict:
    """
    构造符合 lerobot/pi05_libero_base 期望的虚拟观测。

    【键名和形状来源】
    运行脚本后打印的 input_features 即完整列表，以该输出为准：
      observation.images.image          (3, 256, 256)  主摄像头
      observation.images.image2         (3, 256, 256)  第二摄像头
      observation.images.empty_camera_0 (3, 224, 224)  占位摄像头（填零）
      observation.state                 (8,)            本体状态

    【语言 token】
    pi0.5 是语言条件模型，必须提供两个额外键：
      observation.language.tokens          → tokenizer 编码后的 token id
      observation.language.attention_mask  → 1 表示真实 token，0 表示 padding

    tokenizer_max_length=200 来自 PI05Config 默认值。
    """
    # tokenize 任务描述
    # padding="max_length"：统一填充到固定长度（模型要求定长输入）
    # truncation=True：超长截断
    encoded = tokenizer(
        task,
        return_tensors="pt",
        padding="max_length",
        max_length=200,
        truncation=True,
    )

    obs = {
        "observation.images.image": torch.zeros(1, 3, 256, 256, dtype=dtype, device=device),
        "observation.images.image2": torch.zeros(1, 3, 256, 256, dtype=dtype, device=device),
        "observation.images.empty_camera_0": torch.zeros(1, 3, 224, 224, dtype=dtype, device=device),
        "observation.state": torch.zeros(1, 8, dtype=dtype, device=device),
        # 语言 token（必须项）
        "observation.language.tokens": encoded["input_ids"].to(device),
        "observation.language.attention_mask": encoded["attention_mask"].to(device),
    }
    return obs


def run_inference(policy, observation: dict):
    """
    执行一步推理，返回预测的动作。

    【torch.no_grad() 的作用】
    训练时，PyTorch 会记录每次计算的"计算图"（用于反向传播求梯度）。
    推理时不需要梯度，用 no_grad() 禁用这个记录，可以：
    1. 节省约 30–50% 的显存
    2. 加快计算速度
    """
    with torch.no_grad():
        # select_action 是 LeRobot 策略的统一接口
        # 输入：observation dict；输出：action tensor
        action = policy.select_action(observation)
    return action


def main():
    dtype = getattr(torch, DTYPE_STR)

    policy, tokenizer = load_policy_and_tokenizer(MODEL_ID, LOCAL_CKPT, DEVICE, dtype)

    print("\n模型期望的输入特征（键名和形状）：")
    for k, v in policy.config.input_features.items():
        print(f"  {k}: shape={v.shape}, type={v.type}")

    print(f"\n任务描述：{TASK!r}")
    print("\n构造虚拟观测（实际使用时替换为真实环境数据）...")
    obs = make_dummy_libero_observation(tokenizer, TASK, DEVICE, dtype)
    for key, val in obs.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    print("\n运行推理...")
    action = run_inference(policy, obs)

    print(f"\n推理结果：")
    print(f"  action shape : {action.shape}")
    print(f"  action dtype : {action.dtype}")
    print(f"  action values: {action[0].cpu().numpy()}")
    print("\n✅ 推理成功")


if __name__ == "__main__":
    main()
