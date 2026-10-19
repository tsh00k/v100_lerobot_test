#!/usr/bin/env bash
# ===================================================================
# 4_eval_libero.sh —— 在 LIBERO 仿真里评测成功率（lerobot-eval，lerobot==0.6.1）
#
# 每个分片一个进程、一张 GPU，并行跑；结果在 ${OUT_ROOT}/<分片>/eval_info.json，
# 每个任务最多录 10 个 episode 的 mp4（${OUT_ROOT}/<分片>/videos/）。
#
# 分片：SUITES 里写 "libero_10:0-4" 表示只跑该套件的任务 0..4。默认把最慢的 libero_10 拆成两片
# （失败的 episode 会跑满 520 步，单片要 6–7 小时），共占 5 张卡。summarize_eval.py 会按套件合并分片。
#
# 前置：
#   - 无头渲染用 OSMesa（见 docs/troubleshooting.md）
#   - 首次运行需联网：export HTTPS_PROXY=socks5h://127.0.0.1:<端口>
#   - 需要 HF_TOKEN（paligemma tokenizer 是 gated）；或者所有文件都缓存好后 export HF_HUB_OFFLINE=1
#
# 用法（推荐通过 Makefile）：
#   make eval                                        # 官方基线，四个套件，每任务 50 episode
#   make eval SUITES=libero_spatial TASK_IDS=[0] N_EPISODES=2   # 测时间
#   make eval POLICY_PATH=outputs/<run>/checkpoints/last/pretrained_model NAME=lora_r32
# ===================================================================

set -euo pipefail

POLICY_PATH="${POLICY_PATH:-lerobot/pi05_libero_finetuned}"
NAME="${NAME:-$(basename "${POLICY_PATH}")}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10:0-4 libero_10:5-9}"
N_EPISODES="${N_EPISODES:-50}"          # 每个任务；官方评测为 50
N_ENVS="${N_ENVS:-10}"                  # 每个进程里并行的仿真环境数（CPU 渲染，按核数调）
TASK_IDS="${TASK_IDS:-}"                # 如 "[0]"；作用于没有写分片范围的套件；留空 = 全部 10 个任务
GPUS="${GPUS:-0 1 2 3 4 5}"
SEED="${SEED:-1000}"
OUT_ROOT="${OUT_ROOT:-outputs/eval/${NAME}_$(date +%Y%m%d_%H%M%S)}"

export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

# lerobot-eval 会用 --rename_map 覆盖 checkpoint 里保存的映射（不传 = 覆盖成空），
# 所以从 pi05_base 训出来的 checkpoint（相机键是 base_0_rgb 等）评测时必须再传一次。
DEFAULT_RENAME='{"observation.images.image": "observation.images.base_0_rgb", "observation.images.image2": "observation.images.left_wrist_0_rgb"}'
if [[ -z "${RENAME_MAP:-}" && -f "${POLICY_PATH}/config.json" ]] && grep -q base_0_rgb "${POLICY_PATH}/config.json"; then
    RENAME_MAP="${DEFAULT_RENAME}"
fi
RENAME_ARG=()
[[ -n "${RENAME_MAP:-}" ]] && RENAME_ARG=(--rename_map="${RENAME_MAP}")

read -r -a SUITE_ARR <<< "${SUITES}"
read -r -a GPU_ARR <<< "${GPUS}"
if (( ${#SUITE_ARR[@]} > ${#GPU_ARR[@]} )); then
    echo "❌ 分片数 (${#SUITE_ARR[@]}) 多于 GPU 数 (${#GPU_ARR[@]})"; exit 1
fi

# 被占用的卡上再加载一份模型（fp32 约 12 GB 起）会 OOM，启动前先拦下
BUSY=()
for gpu in "${GPU_ARR[@]:0:${#SUITE_ARR[@]}}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}")
    (( used > 2000 )) && BUSY+=("GPU ${gpu}: ${used} MiB")
done
if (( ${#BUSY[@]} > 0 )); then
    echo "❌ 以下 GPU 已被占用："; printf '   %s\n' "${BUSY[@]}"
    echo "   用 nvidia-smi 查看占用进程，或用 GPUS=\"...\" 换卡"; exit 1
fi

# 评测进程是 & 启动的：kill 本脚本、关闭终端、或整条命令放后台时，它们会变成孤儿继续占卡。
# 收到 INT/TERM/HUP 时把整棵进程树（含仿真 worker）一起结束。
kill_tree() {
    local c
    for c in $(pgrep -P "$1"); do kill_tree "$c"; done
    kill -TERM "$1" 2>/dev/null || true
}
cleanup() {
    echo "中断：结束所有评测进程..."
    for p in "${PIDS[@]}"; do kill_tree "$p"; done
    exit 130
}
PIDS=()
trap cleanup INT TERM HUP

mkdir -p "${OUT_ROOT}"
echo "policy=${POLICY_PATH}  suites=${SUITES}  episodes/任务=${N_EPISODES}  n_envs=${N_ENVS}"
echo "rename：${RENAME_MAP:-无}"
echo "输出：${OUT_ROOT}"

for i in "${!SUITE_ARR[@]}"; do
    spec="${SUITE_ARR[$i]}"; gpu="${GPU_ARR[$i]}"
    suite="${spec%%:*}"; shard="${suite}"
    task_ids="${TASK_IDS}"
    if [[ "${spec}" == *:* ]]; then
        range="${spec#*:}"
        task_ids="[$(seq -s, "${range%-*}" "${range#*-}")]"
        shard="${suite}_t${range}"
    fi
    TASK_IDS_ARG=()
    [[ -n "${task_ids}" ]] && TASK_IDS_ARG=(--env.task_ids="${task_ids}")
    # lerobot-eval 要求 output_dir 不存在，所以只建日志、不建分片目录
    log="${OUT_ROOT}/${shard}.log"
    echo "  ${shard} (tasks ${task_ids:-all}) → GPU ${gpu}  (日志 ${log})"
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
        --output_dir="${OUT_ROOT}/${shard}" \
        > "${log}" 2>&1 &
    PIDS+=($!)
done

FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done

python scripts/summarize_eval.py "${OUT_ROOT}" || true
(( FAIL == 0 )) || { echo "❌ 有套件失败，查看 ${OUT_ROOT}/*.log"; exit 1; }
