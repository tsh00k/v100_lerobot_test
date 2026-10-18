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

# ——— 可配置项（通过环境变量覆盖）———
MODEL_ID        = os.getenv("HF_MODEL_ID", "lerobot/pi05_base")
LOCAL_CKPT      = os.getenv("LOCAL_CHECKPOINT", "")   # 如果填了，优先使用本地权重
DEVICE          = os.getenv("DEVICE", "cuda:0")
DTYPE_STR       = os.getenv("DTYPE", "float32")       # V100 用 float32


def load_policy(model_id: str, local_ckpt: str, device: str, dtype: torch.dtype):
    """
    加载 pi0.5 策略模型。

    【LeRobot 的模型加载方式】
    LeRobot 里每个策略（policy）都继承自 HuggingFace 的 PreTrainedModel，
    因此可以用熟悉的 .from_pretrained() 接口加载。
    """
    from lerobot.policies.pi05 import PI05Policy

    checkpoint = local_ckpt if local_ckpt else model_id
    print(f"加载模型：{checkpoint}")
    print(f"  设备：{device}，精度：{dtype}")

    policy = PI05Policy.from_pretrained(checkpoint, torch_dtype=dtype)

    # 把模型移动到 GPU，并切换到评估模式
    # eval() 的作用：关闭 Dropout、BatchNorm 的训练行为，确保推理结果确定性
    policy = policy.to(device).eval()

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  参数量：{n_params / 1e9:.2f}B")
    return policy


def make_dummy_libero_observation(device: str, dtype: torch.dtype) -> dict:
    """
    构造一个符合 LIBERO 格式的虚拟观测。

    【观测空间说明】
    LIBERO 环境提供两路摄像头图像 + 机器人本体状态：
    - agentview_rgb   : 第三人称视角摄像头（看整个场景）
    - eye_in_hand_rgb : 腕部摄像头（看末端执行器附近）
    - state           : 本体感受（proprioception），即各关节角度、速度等

    【数组形状说明（PyTorch 约定）】
    大多数深度学习框架期望输入是一个"批"（batch），即多个样本同时处理。
    即使只有 1 个样本，也要保留批维度：
      - 图像：(batch=1, channels=3, height=224, width=224)
      - 状态：(batch=1, state_dim=8)

    【注意】
    实际使用时，这里的 state_dim 和图像尺寸必须与数据集的观测空间严格一致。
    可以通过 LeRobot 的数据集加载器自动获取这些维度（见 README）。
    """
    obs = {
        # 图像：随机噪声（实际使用时换成真实摄像头帧）
        "observation.images.agentview_rgb": torch.zeros(
            1, 3, 224, 224, dtype=dtype, device=device
        ),
        "observation.images.eye_in_hand_rgb": torch.zeros(
            1, 3, 224, 224, dtype=dtype, device=device
        ),
        # 本体状态：LIBERO 机械臂通常是 7 个关节 + 1 个夹爪 = 8 维
        "observation.state": torch.zeros(1, 8, dtype=dtype, device=device),
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

    policy = load_policy(MODEL_ID, LOCAL_CKPT, DEVICE, dtype)

    print("\n构造虚拟观测（实际使用时替换为真实环境数据）...")
    obs = make_dummy_libero_observation(DEVICE, dtype)
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
