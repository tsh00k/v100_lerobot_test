"""
2_inference.py —— 加载 pi0.5（LIBERO 版），用虚拟观测跑一次推理

LeRobot 的标准推理管线（三段式）：

    raw frame ──preprocessor──▶ batch ──policy.select_action──▶ action ──postprocessor──▶ 真实尺度动作

preprocessor（从 checkpoint 的 policy_preprocessor.json 加载）依次做：
    1. 重命名键 / 加 batch 维
    2. 用数据集统计量归一化 state（QUANTILES 归一化到 [-1, 1]）
    3. 把 state 离散化成 256 档，拼进 prompt：
         "Task: <task>, State: <s1 s2 ...>;\nAction: "
    4. 用 PaliGemma tokenizer（google/paligemma-3b-pt-224）分词
         → observation.language.tokens / attention_mask
    5. 搬到 GPU
postprocessor 把模型输出反归一化回真实动作尺度，并搬回 CPU。

所以：不要手动 tokenize，也不要手动归一化，交给 processor。

前置条件：
    - google/paligemma-3b-pt-224 是 gated repo，需在 HF 网页申请访问并 export HF_TOKEN
    - 首次运行需要联网（走隧道）：export HTTPS_PROXY=socks5h://127.0.0.1:17897

运行：
    python scripts/2_inference.py
    TASK="put the bowl on the plate" python scripts/2_inference.py
    LOCAL_CHECKPOINT=outputs/xxx/checkpoints/last/pretrained_model python scripts/2_inference.py
"""

import os

import torch

MODEL_ID   = os.getenv("HF_MODEL_ID", "lerobot/pi05_libero_base")
LOCAL_CKPT = os.getenv("LOCAL_CHECKPOINT", "")
DEVICE     = os.getenv("DEVICE", "cuda:0")
TASK       = os.getenv("TASK", "pick up the black bowl and place it on the plate")


def load(checkpoint: str, device: str):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    print(f"加载模型：{checkpoint}")
    policy = PI05Policy.from_pretrained(checkpoint)
    policy.config.device = device
    policy = policy.to(device).eval()
    print(f"  参数量：{sum(p.numel() for p in policy.parameters()) / 1e9:.2f}B，dtype={policy.config.dtype}")

    # preprocessor 内的 to_device 步骤默认用保存时的 device，这里覆盖成当前 device
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def make_dummy_frame(policy) -> dict:
    """
    构造一帧"原始"观测（未归一化、无 batch 维），键名取自 policy.config.input_features。
      - 图像：float32，CHW，值域 [0, 1]
      - state：真实物理量（preprocessor 会负责归一化）
      - task：自然语言字符串
    名字带 empty_camera 的是占位相机，不提供时模型会自动用空图填充。
    """
    frame = {}
    for key, feat in policy.config.input_features.items():
        if "empty_camera" in key:
            continue
        frame[key] = torch.rand(*feat.shape) if key.startswith("observation.images.") else torch.zeros(*feat.shape)
    frame["task"] = TASK
    return frame


def main():
    checkpoint = LOCAL_CKPT or MODEL_ID
    policy, preprocessor, postprocessor = load(checkpoint, DEVICE)

    print("\n模型期望的输入特征：")
    for k, v in policy.config.input_features.items():
        print(f"  {k}: {tuple(v.shape)}  {v.type}")
    print("模型输出特征：")
    for k, v in policy.config.output_features.items():
        print(f"  {k}: {tuple(v.shape)}  {v.type}")
    print(f"chunk_size={policy.config.chunk_size}, n_action_steps={policy.config.n_action_steps}, "
          f"num_inference_steps={policy.config.num_inference_steps}")

    frame = make_dummy_frame(policy)
    batch = preprocessor(frame)
    print("\npreprocessor 输出的 batch：")
    for k, v in batch.items():
        desc = f"{tuple(v.shape)} {v.dtype}" if isinstance(v, torch.Tensor) else repr(v)
        print(f"  {k}: {desc}")

    policy.reset()  # 清空 action 队列（每个 episode 开始时都要调用）
    with torch.inference_mode():
        # 一次前向得到整个 chunk：(B, chunk_size, action_dim)，未反归一化
        chunk = policy.predict_action_chunk(batch)
        # select_action：内部维护一个队列，队列空时推理一个 chunk，之后每次弹出一步
        action = policy.select_action(batch)
    action = postprocessor(action)

    print(f"\naction chunk（模型原始输出）: {tuple(chunk.shape)}")
    print(f"单步 action（反归一化后）  : {tuple(action.shape)}")
    print(f"  {action[0].numpy()}")
    print("\n✅ 推理成功")


if __name__ == "__main__":
    main()
