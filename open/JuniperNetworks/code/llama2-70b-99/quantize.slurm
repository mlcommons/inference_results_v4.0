#!/bin/bash

# Quantize Model
srun --container-image {Path to Container Image} \
    --container-mounts { Container Mounts } \
    --container-workdir  { Working Directory } \
    python ../quantization/quantize.py --model_dir { Path to HF Model } \
                                --output_dir {Path to Save Quantized Checkpoints } \
                                --dtype { Datatype } \
                                --tp_size { Tensor Parallelism }  \
                                --pp_size { Pipeline Parallelism } 