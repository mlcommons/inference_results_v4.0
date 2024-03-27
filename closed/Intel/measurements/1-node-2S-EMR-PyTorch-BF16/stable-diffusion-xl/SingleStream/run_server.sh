#!/bin/bash

export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export num_physical_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
num_numa=$(numactl --hardware|grep available|awk -F' ' '{ print $2 }')
export MODEL_PATH=${PWD}/stable_diffusion_fp32/
TIMESTAMP=$(date +%m-%d-%H-%M)
OUTPUT_DIR=./logs

mkdir -p ${OUTPUT_DIR}

NUM_PROC=2
CPUS_PER_PROC=64
WORKERS_PER_PROC=1
TOTAL_SAMPLE_COUNT=5000
BATCH_SIZE=1

LOGNAME=server-performance-bs-${BATCH_SIZE}-proc-${NUM_PROC}-cpus-${CPUS_PER_PROC}-workers-${WORKERS_PER_PROC}

FD_MAX=$(ulimit -n -H)
ulimit -n $((FD_MAX - 1))

python -u main.py \
	--scenario Server \
	--num-proc ${NUM_PROC} \
	--cpus-per-proc ${CPUS_PER_PROC} \
    --batch-size ${BATCH_SIZE} \
	--dtype bfloat16 \
    --device "cpu" \
	--workers-per-proc ${WORKERS_PER_PROC} \
	--warmup \
        --model-path ${MODEL_PATH} \
	2>&1 | tee ${OUTPUT_DIR}/${LOGNAME}.log

