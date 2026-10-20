"""
给不带归一化统计量的 checkpoint（如 lerobot/pi05_libero_base）补上 LIBERO 数据集统计量，另存为本地 checkpoint。

背景：lerobot/pi05_libero_base 的 policy_preprocessor/postprocessor 里 normalizer 的 features 为空、
没有 stats 文件，而 NormalizerProcessorStep 遇到没有统计量的键会原样返回（normalize_processor.py）。
训练时 lerobot-train 会用数据集统计量覆盖，但 lerobot-eval 不会 → 评测时 state 不归一化、action 不反归一化。

做法（与 lerobot_train.py 构建 processor 的方式一致）：
  - 权重和 config.json 原样复用（软链接，不复制）
  - normalizer / unnormalizer 用 HuggingFaceVLA/libero 的统计量，norm_map 用 policy config 的（QUANTILES）

--openpi-prompt：openpi 训练 pi05_libero 时 discrete_state_input=False，prompt 只是 "<task>\n"；
LeRobot 的 pi05 processor 固定拼成 "Task: <task>, State: <8 个数>;\nAction: "。该选项去掉 state 拼接步骤、
在 task 末尾补换行，token 序列与 openpi 逐个一致（已用 openpi 的 paligemma_tokenizer.model 核对）。

--openpi-stats：统计量换成 openpi 训练 pi05_libero 时实际用的那份（从 physical-intelligence/libero 计算，
随 checkpoint 发布在 gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json）。
它和 HuggingFaceVLA/libero 的统计量差别很大（action 的 q01–q99 区间，HF 只有 openpi 的 46%–96%），
而 LeRobot 与 openpi 的 QUANTILES 公式一致（2·(x−q01)/(q99−q01) − 1），所以可以直接替换 q01/q99。
openpi 文件里没有 min/max，LeRobot 的 QUANTILES 模式也不用。

用法：
  python scripts/attach_norm_stats.py                       # 生成 outputs/ckpt/pi05_libero_base_with_stats
  python scripts/attach_norm_stats.py --openpi-prompt       # 生成 outputs/ckpt/pi05_libero_base_openpi_prompt
  python scripts/attach_norm_stats.py --openpi-prompt --openpi-stats   # 生成 outputs/ckpt/pi05_libero_base_openpi_full
  python scripts/attach_norm_stats.py --check 200           # 另外做一次开环检查（不跑仿真）
  python scripts/attach_norm_stats.py --check 200 --no-build
"""

import argparse
import os
import shutil
from pathlib import Path

import torch

DATASET = "HuggingFaceVLA/libero"
OPENPI_STATS_URL = (
    "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
)


def load_openpi_stats(cache: Path) -> dict:
    """下载 openpi 的 norm_stats.json，转成 LeRobot 的 stats 键名。下载走 HTTPS_PROXY。"""
    import json

    import numpy as np
    import requests

    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(OPENPI_STATS_URL, timeout=60)
        r.raise_for_status()
        cache.write_bytes(r.content)
    raw = json.loads(cache.read_text())["norm_stats"]
    return {
        lerobot_key: {k: np.asarray(v, dtype=np.float32) for k, v in raw[openpi_key].items()}
        for lerobot_key, openpi_key in [("observation.state", "state"), ("action", "actions")]
    }


def use_openpi_prompt(pre) -> None:
    from lerobot.policies.pi0.processor_pi0 import Pi0NewLineProcessor
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor.tokenizer_processor import TokenizerProcessorStep

    steps = [s for s in pre.steps if not isinstance(s, Pi05PrepareStateTokenizerProcessorStep)]
    assert len(steps) == len(pre.steps) - 1, "没找到 pi05 的 state 拼接步骤"
    i = next(i for i, s in enumerate(steps) if isinstance(s, TokenizerProcessorStep))
    steps.insert(i, Pi0NewLineProcessor())
    pre.steps = steps


def build(src_dir: Path, out_dir: Path, cfg, stats, openpi_prompt: bool = False) -> None:
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
    if openpi_prompt:
        use_openpi_prompt(pre)
    print("preprocessor steps:", [type(s).__name__ for s in pre.steps])
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


def open_loop_check(src_dir: Path, ckpts: dict[str, Path], n_frames: int) -> None:
    """同一份权重、多套 processor，在数据集帧上比较预测首步动作与示教动作的 L1 误差（真实尺度）。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(str(src_dir))
    policy.config.device = "cuda"
    policy = policy.to("cuda", dtype=torch.float32).eval()

    variants = {}
    for name, path in [("原始（无统计量）", src_dir), *[(k, v) for k, v in ckpts.items() if v.is_dir()]]:
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
        print(f"  {name:16s} 平均 {e.mean():.4f}   逐维 {[round(v, 3) for v in e.mean(0).tolist()]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="lerobot/pi05_libero_base")
    ap.add_argument("--out", default=None)
    ap.add_argument("--openpi-prompt", action="store_true", help="prompt 改成 openpi pi05_libero 的格式（不含 state）")
    ap.add_argument("--openpi-stats", action="store_true", help="统计量用 openpi 发布的 norm_stats.json")
    ap.add_argument("--check", type=int, default=0, help="开环检查的帧数，0 = 不做")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    default_outs = {
        "补统计量": Path("outputs/ckpt/pi05_libero_base_with_stats"),
        "补统计量+openpi prompt": Path("outputs/ckpt/pi05_libero_base_openpi_prompt"),
        "openpi 统计量+prompt": Path("outputs/ckpt/pi05_libero_base_openpi_full"),
    }

    import lerobot.policies.pi05  # noqa: F401  注册 pi05 配置类
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    src_dir = Path(args.src) if Path(args.src).is_dir() else Path(snapshot_download(args.src))
    if args.openpi_stats and not args.openpi_prompt:
        ap.error("--openpi-stats 需要和 --openpi-prompt 一起用（只对 openpi 的完整设置做对照）")
    key = ("openpi 统计量+prompt" if args.openpi_stats
           else "补统计量+openpi prompt" if args.openpi_prompt else "补统计量")
    out_dir = Path(args.out) if args.out else default_outs[key]

    if not args.no_build:
        cfg = PreTrainedConfig.from_pretrained(str(src_dir))
        if args.openpi_stats:
            stats = load_openpi_stats(Path("outputs/ckpt/openpi_pi05_libero_norm_stats.json"))
        else:
            stats = LeRobotDatasetMetadata(DATASET).stats
        print("norm_map:", cfg.normalization_mapping)
        print("action q01/q99:", stats["action"]["q01"], stats["action"]["q99"])
        print("state q01/q99:", stats["observation.state"]["q01"], stats["observation.state"]["q99"])
        build(src_dir, out_dir, cfg, stats, openpi_prompt=args.openpi_prompt)

    if args.check:
        ckpts = {**default_outs, key: out_dir}
        open_loop_check(src_dir, ckpts, args.check)


if __name__ == "__main__":
    main()
