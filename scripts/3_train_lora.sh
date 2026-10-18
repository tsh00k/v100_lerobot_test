#!/usr/bin/env bash
# ===================================================================
# 3_train_lora.sh —— 多卡 LoRA 微调 pi0.5（LIBERO 数据集，fp32）
#
# 【学习要点】
# - 为什么用 accelerate launch 而不是直接 python
# - num_processes 对应 GPU 数量
# - 什么是 mixed_precision，为什么 V100 要关掉
# - 如何通过环境变量灵活调参
#
# 【运行前准备】
#   1. 确保环境已激活：conda activate pi0_v100
#   2. 确保 install.sh 已运行
#   3. （可选）登录 W&B：wandb login
#
# 【运行方式】
#   bash scripts/3_train_lora.sh
#
#   # 调整 GPU 数量：
#   NUM_GPUS=2 bash scripts/3_train_lora.sh
#
#   # 先用少量步数验证流程（~5 分钟）：
#   STEPS=200 bash scripts/3_train_lora.sh
# ===================================================================

set -euo pipefail

# ——— 可调参数（环境变量覆盖）———————————————
NUM_GPUS="${NUM_GPUS:-6}"                        # 使用 GPU 的数量
CONFIG_FILE="${CONFIG_FILE:-configs/lora_libero.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi05_lora_libero}"
STEPS="${STEPS:-50000}"                          # 覆盖 config 里的 steps（调试时设 200）
# ——————————————————————————————————————————

echo "======================================================="
echo " pi0.5 LoRA 微调启动"
echo "   GPU 数量   : ${NUM_GPUS}"
echo "   配置文件   : ${CONFIG_FILE}"
echo "   输出目录   : ${OUTPUT_DIR}"
echo "   训练步数   : ${STEPS}"
echo "======================================================="

mkdir -p "${OUTPUT_DIR}"

# ——— 核心启动命令 ———————————————————————————
#
# accelerate launch：HuggingFace 提供的多 GPU 启动器
#   作用：在多块 GPU 上启动同一个训练脚本，自动做数据并行（DDP）
#   类比：像导演一样，同时叫 N 个演员（GPU 进程）执行同一个剧本
#
# --num_processes：启动的进程数 = 使用的 GPU 数
#
# --mixed_precision=no：关闭混合精度，使用纯 fp32
#   说明：mixed_precision="bf16" 在 A100/H100 上可加速 2-3 倍，
#          但 V100 不支持 bf16，必须关掉，否则报错
#
# python -m lerobot.scripts.lerobot_train：LeRobot 的主训练脚本
#   等价于：cd lerobot && python src/lerobot/scripts/lerobot_train.py
#
accelerate launch \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=no \
    -m lerobot.scripts.lerobot_train \
        --config "${CONFIG_FILE}" \
        --steps "${STEPS}" \
        --output_dir "${OUTPUT_DIR}"

# ——— 训练完成后提示 ———————————————————————————
echo ""
echo "======================================================="
echo " ✅ 训练完成"
echo "   checkpoint 保存在：${OUTPUT_DIR}"
echo ""
echo "   【下一步】用微调后的权重做推理："
echo "   LOCAL_CHECKPOINT=${OUTPUT_DIR} python scripts/2_inference.py"
echo "======================================================="
