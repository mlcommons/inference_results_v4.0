#!/bin/bash

set -ex
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,percpu_arena:percpu,metadata_thp:always,dirty_decay_ms:30000,muzzy_decay_ms:-1"

: ${PRO_BS:=4}
: ${BS:=128}
: ${LEN:=-1}
: ${RESPONSE:=-1}
: ${QOS:=-1}
: ${SCENARIO=${2:-"Offline"}}
: ${ACCURACY:=false}
: ${DEBUG:=false}
: ${MODE:=quant}
: ${WAV:=true}
: ${PROFILE:=-1}
: ${WARMUP:=-1}
: ${VERSION=${1:-"original"}}

SUT_DIR=$(pwd)
EXECUTABLE=${SUT_DIR}/build/rnnt_inference
WORK_DIR=${SUT_DIR}/mlperf-rnnt-librispeech
OUT_DIR="${SUT_DIR}/logs/${SCENARIO}"

num_threads=`nproc --all`
num_physical_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
NUMA=$(numactl --hardware|grep available|awk -F' ' '{ print $2 }')

if [[ ${SCENARIO} == "Offline" ]]; then
  INTRA=4
  INTER=$((num_physical_cores/INTRA))
  OUT_DIR_NANE="${SCENARIO}_${VERSION}_${WAV}_BS${BS}_${INTER}_${INTRA}_SL${LEN}"
elif [[ ${SCENARIO} == "Server" ]]; then
  INTRA=8
  PRO_INTRA=1
  PRO_intra_per_socket=4
  if [ "$((num_physical_cores / NUMA))" = "56" ] ;then
    PRO_intra_per_socket=8
  fi
  PRO_INTER=$((NUMA*PRO_intra_per_socket))
  INTER=$[(num_physical_cores-PRO_INTER*PRO_INTRA)/INTRA]
  OUT_DIR_NANE="${SCENARIO}_${VERSION}_${WAV}_PBS${PRO_BS}_${PRO_INTER}_${PRO_INTRA}_BS${BS}_${INTER}_${INTRA}_SL${LEN}_RSP${RESPONSE}_QOS${QOS}"
fi

if [[ ${ACCURACY} == true ]]; then
  # OUT_DIR="${OUT_DIR}/accuracy/${OUT_DIR_NANE}"
  OUT_DIR="${OUT_DIR}/accuracy"
else
  # OUT_DIR="${OUT_DIR}/performance/run_1/${OUT_DIR_NANE}"
  OUT_DIR="${OUT_DIR}/performance/run_1"
fi
mkdir -p ${OUT_DIR}

number_sockets=`grep physical.id /proc/cpuinfo | sort -u | wc -l`
export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`

SCRIPT_ARGS=" --test_scenario=${SCENARIO}"
SCRIPT_ARGS+=" --model_file=${WORK_DIR}/rnnt_${MODE}_jit.pt"
SCRIPT_ARGS+=" --mlperf_config=${SUT_DIR}/inference/mlperf.conf"

python ../../user_config.py
USER_CONF=user.conf
SCRIPT_ARGS+=" --user_config=${USER_CONF}"

SCRIPT_ARGS+=" --output_dir=${OUT_DIR}"
SCRIPT_ARGS+=" --inter_parallel=${INTER}"
SCRIPT_ARGS+=" --intra_parallel=${INTRA}"
SCRIPT_ARGS+=" --batch_size=${BS}"
SCRIPT_ARGS+=" --split_len=${LEN}"
SCRIPT_ARGS+=" --warmup_iter=${WARMUP}"

if [[ ${WAV} == true ]]; then
  SCRIPT_ARGS+=" --sample_file=${WORK_DIR}/dev-clean-npy.pt --processor_file=${WORK_DIR}/processor_jit.pt --processor"
else
  SCRIPT_ARGS+=" --sample_file=${WORK_DIR}/dev-clean-input.pt --processor_file=${WORK_DIR}/processor_jit.pt"
fi

if [[ ${SCENARIO} == "Server" ]]; then
  SCRIPT_ARGS+=" --pro_inter_parallel=${PRO_INTER} --pro_intra_parallel=${PRO_INTRA} --pro_batch_size=${PRO_BS} --response_size=${RESPONSE} --qos_len=${QOS}"
fi

SCRIPT_ARGS+=" -n ${NUMA}"

if [[ ${ACCURACY} == true ]]; then
  SCRIPT_ARGS+=" --accuracy"
else
  SCRIPT_ARGS+=" --profiler_iter=${PROFILE}"
fi

[ ${DEBUG} != false ]  && EXECUTABLE=${SUT_DIR}/build_dbg/rnnt_inference
[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args"
[ ${DEBUG} == "lldb" ] && EXEC_ARGS="lldb --"
[ ${DEBUG} == "memcheck" ] && EXEC_ARGS="valgrind --leak-check=yes --gen-suppressions=all"

${EXEC_ARGS} ${EXECUTABLE} ${SCRIPT_ARGS}

if [[ ${ACCURACY} == true ]]; then
  export PYTHONPATH=${PWD}:${PWD}/models/:${PYTHONPATH}
  python -u eval_accuracy.py \
    --log_path=${OUT_DIR}/mlperf_log_accuracy.json \
    --manifest_path=${WORK_DIR}/local_data/wav/dev-clean-wav.json | tee ${OUT_DIR}/accuracy.txt
fi

set +x
