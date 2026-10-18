#!/usr/bin/env bash
# ===================================================================
# install.sh —— 项目初始化脚本（在 remote 上 git clone 之后运行一次）
#
# 【前提】已挂载镜像并进入容器（conda activate lerobot 已自动生效）
#
# 【镜像已提供，无需重装】
#   Python 3.12, PyTorch 2.7.1+cu118, lerobot[pi,training]==0.6.1,
#   peft, accelerate, wandb, transformers 等全套依赖
#
# 【本脚本只做三件事】
#   1. 建项目目录（outputs/、data/）
#   2. 可选：把 HuggingFace 缓存指向持久化存储（避免每次重启重新下载）
#   3. 可选：将 lerobot 切换为 editable 安装（方便阅读和修改源码）
#
# 【运行方式】
#   bash install.sh
#
#   # 同时挂载持久化存储并切换 editable 模式：
#   HF_CACHE_DIR=/persistent/hf_cache EDITABLE_LEROBOT=1 bash install.sh
# ===================================================================

set -euo pipefail

HF_CACHE_DIR="${HF_CACHE_DIR:-}"          # 持久化存储路径（留空则不修改）
EDITABLE_LEROBOT="${EDITABLE_LEROBOT:-0}" # 设为 1 则将 lerobot 切换为 editable 安装

# ——— Step 1：建项目目录 ——————————————————————————————
echo ">>> 创建项目目录..."
mkdir -p outputs data
echo "    outputs/  — 训练 checkpoint 和日志"
echo "    data/     — 本地数据集（从 HF Hub 下载后缓存于此）"

# ——— Step 2：持久化 HuggingFace 缓存（可选）——————————
# 说明：lerobot/pi05 模型权重 + LIBERO 数据集合计数十 GB。
# 如果不配置，每次容器重启都要重新下载。
# 配置后，下载一次永久可用（存在持久化存储上）。
if [[ -n "${HF_CACHE_DIR}" ]]; then
    echo ""
    echo ">>> 配置 HuggingFace 缓存 → ${HF_CACHE_DIR}"
    mkdir -p "${HF_CACHE_DIR}"
    # 写入 .bashrc，之后每次进容器自动生效
    EXPORT_LINE="export HF_HOME=${HF_CACHE_DIR}"
    if ! grep -qF "${EXPORT_LINE}" /root/.bashrc 2>/dev/null; then
        echo "${EXPORT_LINE}" >> /root/.bashrc
        echo "    已写入 /root/.bashrc"
    else
        echo "    已存在，跳过"
    fi
    export HF_HOME="${HF_CACHE_DIR}"
else
    echo ""
    echo ">>> 跳过 HF 缓存配置（如需持久化，运行：HF_CACHE_DIR=/your/path bash install.sh）"
fi

# ——— Step 3：editable 模式安装 lerobot（可选）——————————
# 说明：镜像中 lerobot 是固定版本的普通安装（pip install lerobot==0.6.1）。
# 切换为 editable 安装后，可以：
#   - 直接在本地修改 lerobot 源码并立即生效（无需重新 pip install）
#   - 方便学习和调试 LeRobot 内部实现
# 代价：覆盖镜像里的版本，之后 git pull lerobot/ 即可更新。
if [[ "${EDITABLE_LEROBOT}" == "1" ]]; then
    echo ""
    echo ">>> 切换 lerobot 为 editable 安装..."
    if [[ ! -d "lerobot" ]]; then
        git clone https://github.com/huggingface/lerobot.git
    else
        echo "    lerobot/ 目录已存在，跳过克隆"
    fi
    # --no-deps：依赖镜像里已装好的，只替换 lerobot 本身
    pip install --no-deps -e "lerobot[pi,training]"
    echo "    ✓ lerobot editable 安装完成（源码位于 ./lerobot/）"
fi

# ——— 完成 ————————————————————————————————————————
echo ""
echo "✅ 初始化完成。运行以下命令验证环境："
echo "   python /usr/local/share/v100-image/check_environment.py --require-gpu"
