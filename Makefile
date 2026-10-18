# 用法：make <target> [VAR=value ...]
#   make check
#   make infer TASK="put the bowl on the plate"
#   make smoke                         # 1 卡 100 步，上报 wandb
#   make smoke STEPS=300 JOB_NAME=smoke_bs8 BATCH_SIZE=8
#   make train NUM_GPUS=6 STEPS=30000 JOB_NAME=lora_r32
#
# 命令行传入的变量通过 export 传给脚本；没传的保持为空，由脚本里的默认值兜底。

SHELL := /bin/bash

export STEPS JOB_NAME NUM_GPUS BATCH_SIZE LR LORA_R SAVE_FREQ LOG_FREQ OUTPUT_DIR
export WANDB WANDB_PROJECT
export HF_MODEL_ID LOCAL_CHECKPOINT DEVICE TASK

SMOKE_STEPS := $(or $(STEPS),100)

.PHONY: help check infer smoke train

help:
	@grep -E '^#' Makefile | sed 's/^# \{0,1\}//'

check:
	python scripts/1_check_setup.py

infer:
	python scripts/2_inference.py

smoke:
	STEPS=$(SMOKE_STEPS) \
	JOB_NAME=$(or $(JOB_NAME),pi05_smoke) \
	NUM_GPUS=$(or $(NUM_GPUS),1) \
	LOG_FREQ=$(or $(LOG_FREQ),10) \
	SAVE_FREQ=$(or $(SAVE_FREQ),$(SMOKE_STEPS)) \
	bash scripts/3_train_lora.sh

train:
	bash scripts/3_train_lora.sh
