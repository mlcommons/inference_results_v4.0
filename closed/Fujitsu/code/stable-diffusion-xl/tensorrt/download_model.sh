#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

source code/common/file_downloads.sh

# Make sure the script is executed inside the container
if [ -e /work/code/stable-diffusion-xl/tensorrt/download_model.sh ]
then
    echo "Inside container, start downloading..."
else
    echo "WARNING: Please enter the MLPerf container (make prebuild) before downloading SDXL model."
    echo "WARNING: SDXL model is NOT downloaded! Exiting..."
    exit 1
fi

MODEL_DIR=/work/build/models
DATA_DIR=/work/build/data

# Download the fp16 raw weights of MLCommon hosted HF checkpoints
download_file models SDXL/official_pytorch/fp16 \
    https://cloud.mlcommons.org/index.php/s/LCdW5RM6wgGWbxC/download \
    stable_diffusion_fp16.zip

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16.zip | grep "69cc42e5fa40da7c2cd738c931731b7c"
if [ $? -ne 0 ]; then
    echo "SDXL fp16 model md5sum mismatch"
    exit -1
fi

unzip ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16.zip \
    -d ${MODEL_DIR}/SDXL/official_pytorch/fp16/


# Run onnx generation script
python3 -m code.stable-diffusion-xl.tensorrt.create_onnx_model
echo "Runing SDXL UNet quantization on 500 calibration captions. The process will take ~30 mins on DGX H100"
python3 -m code.stable-diffusion-xl.ammo.quantize_sdxl --pretrained-base ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/ --batch-size 1 \
    --calib-size 500 --calib-data /work/code/stable-diffusion-xl/ammo/captions.tsv --percentile 0.4 \
    --n_steps 20 --latent ${DATA_DIR}/coco/SDXL/latents.pt --alpha 0.9 --quant-level 2.5 \
    --int8-ckpt-path ${MODEL_DIR}/SDXL/ammo_models/unetxl.int8.pt
echo "Exporting SDXL fp16-int8 UNet onnx. The process will take ~60 mins on DGX H100"
python3 -m code.stable-diffusion-xl.ammo.export_onnx --pretrained-base ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/ --quantized-ckpt ${MODEL_DIR}/SDXL/ammo_models/unetxl.int8.pt --quant-level 2.5 --onnx-dir ${MODEL_DIR}/SDXL/ammo_models/unetxl.int8
echo "SDXL model download and generation complete!"
