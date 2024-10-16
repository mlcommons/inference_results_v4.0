#!/bin/bash

# Convert checkpoint
srun --container-image {Path to Container Image} \
    --container-mounts { Container Mounts } \
    --container-workdir  { Working Directory } \
    python convert_checkpoint.py --model_dir { Path to HF Model } \
            --output_dir {Path to Save Converted Checkpoints } \
            --dtype { Datatype } \
            --tp_size { Tensor Parallelism } \
            --pp_size { Pipeline Parallelism }

# Build TensorRT Engine
srun --container-image { Path to Container Image } \
    --container-mounts { Conatainer Mounts } \
    --container-workdir { Working Directory } \
    trtllm-build --checkpoint_dir { Path to Converted Checkpoint Directory } \
            --output_dir { Path to Engine Directory }