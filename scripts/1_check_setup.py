"""
1_check_setup.py —— 验证项目所需环境是否完整

【与镜像自带 check_environment.py 的分工】
  /usr/local/share/v100-image/check_environment.py
      → 验证镜像本身是否完好（torch 版本、所有依赖导入、GPU FP32 计算）

  本脚本（1_check_setup.py）
      → 在上面的基础上，验证本项目的额外需求：
        1. 调用镜像脚本做全量检查
        2. 确认 peft 可用（LoRA 微调依赖）
        3. 打印训练会用到的 GPU 显存信息（帮助判断 batch size）

【运行方式】
  python scripts/1_check_setup.py
"""

import subprocess
import sys


def run_image_check():
    """调用镜像内置的环境检查脚本"""
    image_check = "/usr/local/share/v100-image/check_environment.py"
    print(f">>> 调用镜像内置检查：{image_check}\n")
    result = subprocess.run(
        [sys.executable, image_check, "--require-gpu"],
        text=True,
    )
    if result.returncode != 0:
        print("\n❌ 镜像环境检查失败，请确认容器已正确挂载并启动。")
        sys.exit(1)


def check_peft():
    print("\n>>> 检查 peft（LoRA 支持）...")
    try:
        import peft
        print(f"    ✓ peft {peft.__version__}")
    except ImportError:
        print("    ❌ peft 未找到")
        print("       镜像构建时应已包含，如缺失请检查 requirements-lerobot.txt")
        sys.exit(1)


def print_gpu_memory_summary():
    """打印各卡显存，帮助判断训练时 batch size 的上限"""
    import torch
    print("\n>>> GPU 显存摘要（fp32 训练参考）")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"    cuda:{i}  {props.name}  {total_gb:.0f} GB")
    print("\n    fp32 下 pi0.5 推理约占 ~8-12 GB/卡")
    print("    LoRA 微调（batch=4）约占 ~20-24 GB/卡，32 GB 卡可满足")


if __name__ == "__main__":
    run_image_check()
    check_peft()
    print_gpu_memory_summary()
    print("\n✅ 项目环境检查通过，可以开始运行脚本。")
