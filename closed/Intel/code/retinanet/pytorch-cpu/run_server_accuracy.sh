
CPUS_PER_INSTANCE=8
BATCH_SIZE=1

number_threads=`nproc --all`
export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
number_sockets=`grep physical.id /proc/cpuinfo | sort -u | wc -l`
cpu_per_socket=$((number_cores/number_sockets))
number_instance=$((number_cores/CPUS_PER_INSTANCE))

if [ -z "${DATA_DIR}" ]; then
    echo "Path to dataset not set. Please set it:"
    echo "export DATA_DIR=</path/to/openimages>"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "Path to trained checkpoint not set. Please set it:"
    echo "export MODEL_PATH=</path/to/retinanet-int8-model.pth>"
    exit 1
fi

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export $KMP_SETTING

CUR_DIR=${PWD}
APP=${CUR_DIR}/build/bin/mlperf_runner

if [ -e mlperf_log_accuracy.json ]; then
    rm mlperf_log_accuracy.json
fi

if [ -e accuracy.txt ]; then
    rm accuracy.txt
fi

python ../../user_config.py
USER_CONF=user.conf

${APP} --scenario Server \
	--mode Accuracy \
	--mlperf_conf mlperf.conf \
	--user_conf ${USER_CONF} \
	--model_name retinanet \
    --model_path ${MODEL_PATH} \
	--data_path ${DATA_DIR} \
	--num_instance $number_instance \
	--warmup_iters 100 \
	--cpus_per_instance $CPUS_PER_INSTANCE \
	--total_sample_count 24781 \
    --batch_size $BATCH_SIZE


echo " ==================================="
echo "         Evaluating Accuracy        "
echo " ==================================="

if [ -e mlperf_log_accuracy.json ]; then
    python -u ${ENV_DEPS_DIR}/mlperf_inference/vision/classification_and_detection/tools/accuracy-openimages.py \
        --mlperf-accuracy-file mlperf_log_accuracy.json \
        --openimages-dir ${DATA_DIR} 2>&1 | tee "accuracy.txt"
fi
