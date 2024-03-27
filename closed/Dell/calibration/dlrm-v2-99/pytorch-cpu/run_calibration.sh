#!/bin/bash
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
# calibration
numactl -C 56-111 -m 1 python python/calibration.py \
        --max-batchsize=65536 \
        --model-path=/data/dlrm_2_dataset/model/dlrm-multihot-pytorch.pt \
        --dataset-path=/data/dlrm_2_dataset/inference \
        --use-int8 --calibration

