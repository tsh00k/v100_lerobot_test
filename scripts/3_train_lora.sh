#!/usr/bin/env bash
# ===================================================================
# 3_train_lora.sh —— pi0.5 LoRA 微调（LIBERO，fp32，多卡 DDP）
#
# 参数全部对照 lerobot==0.6.1 源码核实：
#   顶层字段：lerobot/configs/train.py  TrainPipelineConfig
#   LoRA 字段：lerobot/configs/default.py  PeftConfig
#   pi05 字段：lerobot/policies/pi05/configuration_pi05.py  PI05Config
# 查看全部参数：lerobot-train --help
#
# 前置：
#   export HF_TOKEN=hf_xxx                              # paligemma tokenizer 是 gated
#   export HTTPS_PROXY=socks5h://127.0.0.1:<端口>       # 首次下载需要；端口以服务器上 ss -tlnp | grep sshd 为准
#   在 tmux 里运行！
#
# 用法（推荐通过 Makefile）：
#   make smoke                       # 冒烟测试：6 卡 20 步
#   make train                       # 正式训练：6 卡
#   STEPS=200 NUM_GPUS=1 bash scripts/3_train_lora.sh   # 也可直接调用
#
# wandb：默认开启，checkpoint 不作为 artifact 上传（单个 >16GB）
# ===================================================================

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-6}"
BATCH_SIZE="${BATCH_SIZE:-4}"          # 每卡；有效 batch = BATCH_SIZE × NUM_GPUS
STEPS="${STEPS:-30000}"
LR="${LR:-1e-4}"                       # LoRA 常用比全量微调（pi05 默认 2.5e-5）更大的 lr
LORA_R="${LORA_R:-32}"
# 动作 chunk：采用 openpi 的 LIBERO 配方（pi05_libero 从 pi05_base 出发，action_horizon 50 → 10，评测每 5 步重新规划）
#   PI05Config 默认 chunk_size=50、n_action_steps=50；二者都会写进 checkpoint 的 config.json，评测时自动沿用
#   约束：n_action_steps ≤ chunk_size（configuration_pi05.py 中校验）
CHUNK_SIZE="${CHUNK_SIZE:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-5}"
JOB_NAME="${JOB_NAME:-pi05_lora_libero}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)}"  # 必须是不存在的目录
SAVE_FREQ="${SAVE_FREQ:-5000}"         # 最后一步无论如何都会保存
LOG_FREQ="${LOG_FREQ:-50}"
WANDB="${WANDB:-true}"                 # 需要 WANDB_API_KEY（或 wandb login）
WANDB_PROJECT="${WANDB_PROJECT:-pi05_v100}"
EPISODES="${EPISODES:-}"               # 如 "[0,1,2]"；只下载/使用这些 episode。留空 = 全量（35 GB）

# pi05_base：通用预训练，未见过 LIBERO（实验的起点）
# pi05_libero_base：openpi 官方已在 LIBERO 上全量微调过的 pi05_libero（只适合做对照）
POLICY_PATH="${POLICY_PATH:-lerobot/pi05_base}"

EPISODES_ARG=()
[[ -n "${EPISODES}" ]] && EPISODES_ARG=(--dataset.episodes="${EPISODES}")

# pi05_base 的相机键名是 base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb，数据集是 image / image2。
# 与 openpi 的 LIBERO 映射一致：主相机→base，腕部→left_wrist，right_wrist 缺失时模型自动填空图并 mask。
DEFAULT_RENAME='{"observation.images.image": "observation.images.base_0_rgb", "observation.images.image2": "observation.images.left_wrist_0_rgb"}'
RENAME_ARG=()
if [[ -n "${RENAME_MAP:-}" ]]; then
    RENAME_ARG=(--rename_map="${RENAME_MAP}")
elif [[ "${POLICY_PATH}" == "lerobot/pi05_base" ]]; then
    RENAME_ARG=(--rename_map="${DEFAULT_RENAME}")
fi

# LoRA 作用位置（peft 对字符串做 re.fullmatch；命中数已对照 pi05 模块名核实）
#   default：LeRobot 默认，action expert 的 q/v + action_in/out_proj            （38 层）
#   expert ：action expert 全部 attn+mlp + 4 个动作/时间投影                     （130 层）
#   all    ：expert + PaliGemma 语言主干全部 attn+mlp；视觉编码器仍冻结          （256 层）
# expert 与 all 只差语言主干，用来做"要不要适配 VLM"的消融。
LORA_TARGETS="${LORA_TARGETS:-default}"
_PROJ="action_in_proj|action_out_proj|time_mlp_in|time_mlp_out"
_BLOCK='(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'
case "${LORA_TARGETS}" in
    default) TARGETS_ARG=() ;;
    expert)  TARGETS_ARG=(--peft.target_modules="(.*\.gemma_expert\..*\.${_BLOCK}|model\.(${_PROJ}))") ;;
    all)     TARGETS_ARG=(--peft.target_modules="(.*\.(gemma_expert|language_model)\..*\.${_BLOCK}|model\.(${_PROJ}))") ;;
    *) echo "❌ LORA_TARGETS 只能是 default / expert / all"; exit 1 ;;
esac

# 单卡不加 --multi_gpu
MULTI_GPU_FLAG=""
[[ "${NUM_GPUS}" -gt 1 ]] && MULTI_GPU_FLAG="--multi_gpu"

if [[ "${WANDB}" == "true" && -z "${WANDB_API_KEY:-}" && ! -f ~/.netrc ]]; then
    echo "❌ WANDB=true 但没有找到 WANDB_API_KEY 或 ~/.netrc。export WANDB_API_KEY=... 或 WANDB=false"
    exit 1
fi

echo "GPUs=${NUM_GPUS} batch/卡=${BATCH_SIZE} steps=${STEPS} lr=${LR} r=${LORA_R} targets=${LORA_TARGETS} chunk=${CHUNK_SIZE}/${N_ACTION_STEPS} wandb=${WANDB}(${WANDB_PROJECT})"
echo "base：${POLICY_PATH}  rename：${RENAME_ARG[*]:-无}"
echo "episodes：${EPISODES:-全部}"
echo "输出目录：${OUTPUT_DIR}"

# 耗时和资源监控：OUTPUT_DIR 必须由 lerobot 自己创建，所以监控文件放在同级的 .monitor 目录
#   full：GPU（nvidia-smi）+ CPU + 主机内存，默认每 5 秒采样（make smoke 用这个）
#   cpu ：只采 CPU + 主机内存，默认每 30 秒（正式训练默认；GPU 情况 wandb 的 system 面板已有）
#   off ：不采样，只保留 train.log 和墙钟时间
# 每次采样只是读 /proc 或调一次 nvidia-smi，开销远小于一个训练 step
MONITOR="${MONITOR:-cpu}"
case "${MONITOR}" in
    full) MON_INTERVAL="${MON_INTERVAL:-5}" ;;
    cpu)  MON_INTERVAL="${MON_INTERVAL:-30}" ;;
    off)  ;;
    *) echo "❌ MONITOR 只能是 full / cpu / off"; exit 1 ;;
esac
MON_DIR="${OUTPUT_DIR}.monitor"
mkdir -p "${MON_DIR}"
echo "监控：${MONITOR}  目录：${MON_DIR}"
MON_PID=""
if [[ "${MONITOR}" != "off" ]]; then
    (
        while true; do
            t=$(date +%s)
            if [[ "${MONITOR}" == "full" ]]; then
                nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
                    | sed "s/^/${t},/; s/ //g" >> "${MON_DIR}/gpu.csv"
            fi
            # host.csv：时间, 已用内存 MiB, 1 分钟负载, /proc/stat 中 cpu 总行的各项 jiffies（汇总时做差分得利用率）
            echo "${t},$(free -m | awk '/^Mem:/{print $3}'),$(cut -d' ' -f1 /proc/loadavg),$(awk '/^cpu /{$1=""; print substr($0,2)}' /proc/stat | tr ' ' ',')" \
                >> "${MON_DIR}/host.csv"
            sleep "${MON_INTERVAL}"
        done
    ) &
    MON_PID=$!
    trap 'kill ${MON_PID} 2>/dev/null || true' EXIT
fi
T0=$(date +%s)

# accelerate 负责：起 N 个进程、DDP 包装、梯度同步、混合精度
#   --mixed_precision=no：V100 无 bf16，全程 fp32
accelerate launch \
    ${MULTI_GPU_FLAG} \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=no \
    -m lerobot.scripts.lerobot_train \
    \
    --dataset.repo_id=HuggingFaceVLA/libero \
    "${EPISODES_ARG[@]}" \
    "${RENAME_ARG[@]}" \
    \
    --policy.path="${POLICY_PATH}" \
    --policy.dtype=float32 \
    --policy.gradient_checkpointing=true \
    --policy.optimizer_lr="${LR}" \
    --policy.scheduler_decay_steps="${STEPS}" \
    --policy.push_to_hub=false \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    \
    --peft.method_type=LORA \
    --peft.r="${LORA_R}" \
    --peft.lora_alpha="${LORA_R}" \
    "${TARGETS_ARG[@]}" \
    \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --log_freq="${LOG_FREQ}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    \
    --wandb.enable="${WANDB}" \
    --wandb.project="${WANDB_PROJECT}" \
    --wandb.disable_artifact=true \
    2>&1 | tee "${MON_DIR}/train.log"

echo $(( $(date +%s) - T0 )) > "${MON_DIR}/wall_s"
[[ -n "${MON_PID}" ]] && kill "${MON_PID}" 2>/dev/null || true
echo ""
python scripts/summarize_train.py "${MON_DIR}" "${TARGET_STEPS:-30000}" || true

echo ""
echo "✅ 训练完成：${OUTPUT_DIR}"
echo "推理：LOCAL_CHECKPOINT=${OUTPUT_DIR}/checkpoints/last/pretrained_model python scripts/2_inference.py"
