"""
给不带归一化统计量的 checkpoint（如 lerobot/pi05_libero_base）补上 LIBERO 数据集统计量，另存为本地 checkpoint。

背景：lerobot/pi05_libero_base 的 policy_preprocessor/postprocessor 里 normalizer 的 features 为空、
没有 stats 文件，而 NormalizerProcessorStep 遇到没有统计量的键会原样返回（normalize_processor.py）。
训练时 lerobot-train 会用数据集统计量覆盖，但 lerobot-eval 不会 → 评测时 state 不归一化、action 不反归一化。

做法（与 lerobot_train.py 构建 processor 的方式一致）：
  - 权重和 config.json 原样复用（软链接，不复制）
  - normalizer / unnormalizer 用 HuggingFaceVLA/libero 的统计量，norm_map 用 policy config 的（QUANTILES）

用法：
  python scripts/attach_norm_stats.py                       # 生成 outputs/ckpt/pi05_libero_base_with_stats
  python scripts/attach_norm_stats.py --check 200           # 另外做一次开环检查（不跑仿真）
  python scripts/attach_norm_stats.py --check 200 --no-build
"""

import argparse
import os
import shutil
from pathlib import Path

import torch

DATASET = "HuggingFaceVLA/libero"


def build(src_dir: Path, out_dir: Path, cfg, stats) -> None:
    from lerobot.policies.factory import make_pre_post_processors

    features = {**cfg.input_features, **cfg.output_features}
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=str(src_dir),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {"stats": stats, "features": features, "norm_map": cfg.normalization_mapping},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": stats, "features": cfg.output_features, "norm_map": cfg.normalization_mapping,
            },
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_dir / "config.json", out_dir / "config.json")
    link = out_dir / "model.safetensors"
    if not link.exists():
        os.symlink((src_dir / "model.safetensors").resolve(), link)
    pre.save_pretrained(out_dir, config_filename="policy_preprocessor.json")
    post.save_pretrained(out_dir, config_filename="policy_postprocessor.json")
    print(f"已生成 {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print("  ", p.name)


def open_loop_check(src_dir: Path, out_dir: Path, n_frames: int) -> None:
    """同一份权重、两套 processor，在数据集帧上比较预测动作与示教动作的 L1 误差（归一化前的真实尺度）。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(str(src_dir))
    policy.config.device = "cuda"
    policy = policy.to("cuda", dtype=torch.float32).eval()

    variants = {}
    for name, path in [("原始（无统计量）", src_dir), ("补统计量", out_dir)]:
        variants[name] = make_pre_post_processors(
            policy.config, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )

    ds = LeRobotDataset(DATASET, episodes=list(range(10)))
    idx = torch.linspace(0, len(ds) - 1, n_frames).long().tolist()
    errs = {k: [] for k in variants}
    for i in idx:
        frame = ds[i]
        gt = frame["action"].float()
        for name, (pre, post) in variants.items():
            policy.reset()
            with torch.inference_mode():
                act = post(policy.select_action(pre(dict(frame))))[0].float().cpu()
            errs[name].append((act - gt).abs())

    print(f"\n开环检查：{len(idx)} 帧（{DATASET} 前 10 条示教），预测首步动作 vs 示教动作，逐维 L1 误差均值")
    print("示教动作逐维 |a| 均值（量级参考）:", [round(v, 3) for v in torch.stack([ds[i]["action"] for i in idx]).abs().mean(0).tolist()])
    for name, e in errs.items():
        e = torch.stack(e)
        print(f"  {name:10s} 平均 {e.mean():.4f}   逐维 {[round(v, 3) for v in e.mean(0).tolist()]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="lerobot/pi05_libero_base")
    ap.add_argument("--out", default="outputs/ckpt/pi05_libero_base_with_stats")
    ap.add_argument("--check", type=int, default=0, help="开环检查的帧数，0 = 不做")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    import lerobot.policies.pi05  # noqa: F401  注册 pi05 配置类
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    src_dir = Path(args.src) if Path(args.src).is_dir() else Path(snapshot_download(args.src))
    out_dir = Path(args.out)

    if not args.no_build:
        cfg = PreTrainedConfig.from_pretrained(str(src_dir))
        stats = LeRobotDatasetMetadata(DATASET).stats
        print("norm_map:", cfg.normalization_mapping)
        print("state q01/q99:", stats["observation.state"]["q01"], stats["observation.state"]["q99"])
        build(src_dir, out_dir, cfg, stats)

    if args.check:
        open_loop_check(src_dir, out_dir, args.check)


if __name__ == "__main__":
    main()
