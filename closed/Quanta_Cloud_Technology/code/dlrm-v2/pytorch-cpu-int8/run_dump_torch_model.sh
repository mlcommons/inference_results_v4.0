#!/bin/bash
# export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"

# transform
numactl -C 56-111 -m 1 python python/dump_torch_model.py --model-path=$MODEL_DIR/model_weights --dataset-path=$DATA_DIR
