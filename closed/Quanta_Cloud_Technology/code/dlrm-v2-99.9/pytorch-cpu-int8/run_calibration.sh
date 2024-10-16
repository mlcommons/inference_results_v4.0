#!/bin/bash
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
# calibration
python python/calibration.py \
        --max-batchsize=65536 \
        --model-path=${MODEL_DIR}/dlrm-multihot-pytorch.pt \
        --dataset-path=${DATA_DIR} \
        --use-int8 --calibration

