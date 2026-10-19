# 用法：make <target> [VAR=value ...]
#   make check
#   make data EPISODES=[0,1,2,3,4,5,6,7,8,9]   # 只预下载这些 episode（约 300 MB）
#   make data                                  # 预下载全量 LIBERO（35 GB，隧道下约 7 小时）
#   make infer TASK="put the bowl on the plate"
#   make smoke                         # 6 卡 20 步，只用前 10 个 episode，上报 wandb
#   make smoke NUM_GPUS=1              # 出问题时退回单卡排查
#   make smoke STEPS=50 JOB_NAME=smoke_bs8 BATCH_SIZE=8
#   make train NUM_GPUS=6 STEPS=30000 JOB_NAME=lora_r32   # 全量数据
#   make eval                          # LIBERO 四套件评测（libero_10 拆两片，共 5 卡），默认官方基线 pi05_libero_finetuned
#   make eval SUITES=libero_spatial TASK_IDS=[0] N_EPISODES=2 N_ENVS=2   # 测耗时
#   make eval SUITES="libero_10:0-4 libero_10:5-9" N_EPISODES=10        # 只看最难的 libero_10，快速对比
#   make eval POLICY_PATH=outputs/<run>/checkpoints/last/pretrained_model NAME=lora_r32
#
# 实验序列（训练都从 pi05_base 出发；每次训完用 make eval 评测）：
#   0. 诊断   make norm-stats                           # pi05_libero_base + LIBERO 统计量
#             make norm-stats OPENPI_PROMPT=1 CHECK=200 # 再改成 openpi 的 prompt 格式 + 三者开环对比
#             make eval POLICY_PATH=lerobot/pi05_libero_base NAME=libero_base_raw N_EPISODES=10
#             make eval POLICY_PATH=outputs/ckpt/pi05_libero_base_with_stats NAME=libero_base_stats
#             make eval POLICY_PATH=outputs/ckpt/pi05_libero_base_openpi_prompt NAME=libero_base_openpi_prompt
#   1. 基线   make eval
#   2. 主实验 make train JOB_NAME=r32_default
#   3. rank   make train JOB_NAME=r8_default  LORA_R=8
#             make train JOB_NAME=r64_default LORA_R=64
#   4. 位置   make train JOB_NAME=r32_expert  LORA_TARGETS=expert
#             make train JOB_NAME=r32_all     LORA_TARGETS=all
#
# 命令行传入的变量通过 export 传给脚本；没传的保持为空，由脚本里的默认值兜底。

SHELL := /bin/bash

export STEPS JOB_NAME NUM_GPUS BATCH_SIZE LR LORA_R SAVE_FREQ LOG_FREQ OUTPUT_DIR EPISODES
export POLICY_PATH RENAME_MAP LORA_TARGETS
export WANDB WANDB_PROJECT
export HF_MODEL_ID LOCAL_CHECKPOINT DEVICE TASK
export NAME SUITES N_EPISODES N_ENVS TASK_IDS GPUS SEED OUT_ROOT

SMOKE_STEPS    := $(or $(STEPS),20)
SMOKE_EPISODES := [0,1,2,3,4,5,6,7,8,9]

.PHONY: help check data infer smoke train eval norm-stats

help:
	@grep -E '^#' Makefile | sed 's/^# \{0,1\}//'

check:
	python scripts/1_check_setup.py

# 多卡训练时其他进程会在屏障处等主进程下载数据集，下载过慢会触发 NCCL 超时，所以先单独下好。
# 下载可断点续传，中断后重跑即可。
data:
	python -c "import json, os; from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; \
	e = os.environ.get('EPISODES'); d = D('HuggingFaceVLA/libero', episodes=json.loads(e) if e else None); \
	print(d.num_episodes, 'episodes', d.num_frames, 'frames')"

infer:
	python scripts/2_inference.py

# 目的：用最少的步数把完整流程（DDP / LoRA / wandb / 保存 checkpoint）跑一遍，不看收敛
smoke:
	EPISODES='$(or $(EPISODES),$(SMOKE_EPISODES))' \
	$(MAKE) --no-print-directory data
	STEPS=$(SMOKE_STEPS) \
	EPISODES='$(or $(EPISODES),$(SMOKE_EPISODES))' \
	JOB_NAME=$(or $(JOB_NAME),pi05_smoke) \
	NUM_GPUS=$(or $(NUM_GPUS),6) \
	LOG_FREQ=$(or $(LOG_FREQ),2) \
	SAVE_FREQ=$(or $(SAVE_FREQ),$(SMOKE_STEPS)) \
	bash scripts/3_train_lora.sh

train:
	bash scripts/3_train_lora.sh

eval:
	bash scripts/4_eval_libero.sh

# 只构建：几十秒，不占 GPU；CHECK=N 时额外加载模型（1 张卡），在 N 个数据集帧上对比两套 processor
norm-stats:
	CUDA_VISIBLE_DEVICES=$(or $(firstword $(GPUS)),0) python scripts/attach_norm_stats.py --check $(or $(CHECK),0) \
	  $(if $(OPENPI_PROMPT),--openpi-prompt)
