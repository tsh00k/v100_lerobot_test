# 用法：make <target> [VAR=value ...]
#   make check
#   make data                          # 预下载 LIBERO 数据集（多卡训练前先跑）
#   make infer TASK="put the bowl on the plate"
#   make smoke                         # 6 卡 20 步，上报 wandb
#   make smoke NUM_GPUS=1              # 出问题时退回单卡排查
#   make smoke STEPS=50 JOB_NAME=smoke_bs8 BATCH_SIZE=8
#   make train NUM_GPUS=6 STEPS=30000 JOB_NAME=lora_r32
#
# 命令行传入的变量通过 export 传给脚本；没传的保持为空，由脚本里的默认值兜底。

SHELL := /bin/bash

export STEPS JOB_NAME NUM_GPUS BATCH_SIZE LR LORA_R SAVE_FREQ LOG_FREQ OUTPUT_DIR
export WANDB WANDB_PROJECT
export HF_MODEL_ID LOCAL_CHECKPOINT DEVICE TASK

SMOKE_STEPS := $(or $(STEPS),20)

.PHONY: help check data infer smoke train

help:
	@grep -E '^#' Makefile | sed 's/^# \{0,1\}//'

check:
	python scripts/1_check_setup.py

# 多卡训练时其他进程会在屏障处等主进程下载数据集，下载过慢会触发 NCCL 超时，所以先单独下好
data:
	python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; d = D('HuggingFaceVLA/libero'); print(d.num_episodes, 'episodes', d.num_frames, 'frames')"

infer:
	python scripts/2_inference.py

# 目的：用最少的步数把完整流程（DDP / LoRA / wandb / 保存 checkpoint）跑一遍，不看收敛
smoke:
	STEPS=$(SMOKE_STEPS) \
	JOB_NAME=$(or $(JOB_NAME),pi05_smoke) \
	NUM_GPUS=$(or $(NUM_GPUS),6) \
	LOG_FREQ=$(or $(LOG_FREQ),2) \
	SAVE_FREQ=$(or $(SAVE_FREQ),$(SMOKE_STEPS)) \
	bash scripts/3_train_lora.sh

train:
	bash scripts/3_train_lora.sh
