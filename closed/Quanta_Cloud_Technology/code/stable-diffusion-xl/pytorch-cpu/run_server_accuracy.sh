#!/bin/bash


export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export num_physical_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
num_numa=$(numactl --hardware|grep available|awk -F' ' '{ print $2 }')

TIMESTAMP=$(date +%m-%d-%H-%M)

OUTPUT_DIR=./logs


mkdir -p ${OUTPUT_DIR}

NUM_PROC=2
CPUS_PER_PROC=64
WORKERS_PER_PROC=1
TOTAL_SAMPLE_COUNT=5000
BATCH_SIZE=1

LOGNAME=server-accuracy-bs-${BATCH_SIZE}-proc-${NUM_PROC}-cpus-${CPUS_PER_PROC}-workers-${WORKERS_PER_PROC}

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
	--cores-offset 0 \
	--total-sample-count ${TOTAL_SAMPLE_COUNT} \
	--log-dir ${OUTPUT_DIR} \
	--accuracy \
  --warmup \
  --model-path ${MODEL_PATH} \
	2>&1 | tee ${OUTPUT_DIR}/${LOGNAME}.log

echo "OUTPUT_DIR" ${OUTPUT_DIR}
if [ -e ${OUTPUT_DIR}/mlperf_log_accuracy.json ]; then
	echo " ==================================="
	echo "         Evaluating Accuracy        "
	echo " ==================================="

	python tools/accuracy_coco.py --mlperf-accuracy-file ${OUTPUT_DIR}/mlperf_log_accuracy.json --scenario "server" \
		--dataset-dir ./coco2014 --statistics-path tools/val2014.npz 2>&1 | tee -a accuracy-server-${TIMESTAMP}.txt
	cat coco-results-server.json
fi
