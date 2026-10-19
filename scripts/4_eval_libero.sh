#!/usr/bin/env bash
# ===================================================================
# 4_eval_libero.sh —— 在 LIBERO 仿真里评测成功率（lerobot-eval，lerobot==0.6.1）
#
# 每个套件一个进程、一张 GPU，并行跑；结果在 ${OUT_ROOT}/<suite>/eval_info.json，
# 每个任务最多录 10 个 episode 的 mp4（${OUT_ROOT}/<suite>/videos/）。
#
# 前置：
#   - 无头渲染用 OSMesa（见 docs/troubleshooting.md）
#   - 首次运行需联网：export HTTPS_PROXY=socks5h://127.0.0.1:<端口>
#
# 用法（推荐通过 Makefile）：
#   make eval                                        # 官方基线，四个套件，每任务 50 episode
#   make eval SUITES=libero_spatial TASK_IDS=[0] N_EPISODES=2   # 测时间
#   make eval POLICY_PATH=outputs/<run>/checkpoints/last/pretrained_model NAME=lora_r32
# ===================================================================

set -euo pipefail

POLICY_PATH="${POLICY_PATH:-lerobot/pi05_libero_finetuned_v044}"
NAME="${NAME:-$(basename "${POLICY_PATH}")}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
N_EPISODES="${N_EPISODES:-50}"          # 每个任务；官方评测为 50
N_ENVS="${N_ENVS:-10}"                  # 每个进程里并行的仿真环境数（CPU 渲染，按核数调）
TASK_IDS="${TASK_IDS:-}"                # 如 "[0]"；留空 = 套件内全部 10 个任务
GPUS="${GPUS:-0 1 2 3 4 5}"
SEED="${SEED:-1000}"
OUT_ROOT="${OUT_ROOT:-outputs/eval/${NAME}_$(date +%Y%m%d_%H%M%S)}"

export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

TASK_IDS_ARG=()
[[ -n "${TASK_IDS}" ]] && TASK_IDS_ARG=(--env.task_ids="${TASK_IDS}")
RENAME_ARG=()
[[ -n "${RENAME_MAP:-}" ]] && RENAME_ARG=(--rename_map="${RENAME_MAP}")

read -r -a SUITE_ARR <<< "${SUITES}"
read -r -a GPU_ARR <<< "${GPUS}"
if (( ${#SUITE_ARR[@]} > ${#GPU_ARR[@]} )); then
    echo "❌ 套件数 (${#SUITE_ARR[@]}) 多于 GPU 数 (${#GPU_ARR[@]})"; exit 1
fi

mkdir -p "${OUT_ROOT}"
echo "policy=${POLICY_PATH}  suites=${SUITES}  episodes/任务=${N_EPISODES}  n_envs=${N_ENVS}"
echo "输出：${OUT_ROOT}"

PIDS=()
for i in "${!SUITE_ARR[@]}"; do
    suite="${SUITE_ARR[$i]}"; gpu="${GPU_ARR[$i]}"
    # lerobot-eval 要求 output_dir 不存在，所以只建日志、不建 suite 目录
    log="${OUT_ROOT}/${suite}.log"
    echo "  ${suite} → GPU ${gpu}  (日志 ${log})"
    CUDA_VISIBLE_DEVICES="${gpu}" lerobot-eval \
        --policy.path="${POLICY_PATH}" \
        --policy.dtype=float32 \
        --policy.device=cuda \
        --env.type=libero \
        --env.task="${suite}" \
        "${TASK_IDS_ARG[@]}" \
        "${RENAME_ARG[@]}" \
        --eval.n_episodes="${N_EPISODES}" \
        --eval.batch_size="${N_ENVS}" \
        --seed="${SEED}" \
        --output_dir="${OUT_ROOT}/${suite}" \
        > "${log}" 2>&1 &
    PIDS+=($!)
done

FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done

python scripts/summarize_eval.py "${OUT_ROOT}" || true
(( FAIL == 0 )) || { echo "❌ 有套件失败，查看 ${OUT_ROOT}/*.log"; exit 1; }
