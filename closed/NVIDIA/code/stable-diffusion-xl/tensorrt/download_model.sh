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
#if [ -e /code/stable-diffusion-xl/tensorrt/download_model.sh ]
#then
#    echo "Inside container, start downloading..."
#else
#    echo "WARNING: Please enter the MLPerf container (make prebuild) before downloading SDXL model."
#    echo "WARNING: SDXL model is NOT downloaded! Exiting..."
#    exit 1
#fi

MODEL_DIR=build/models
DATA_DIR=build/data

if [ -z "$SDXL_CHECKPOINT_PATH" ]; then
    echo "SDXL_CHECKPOINT_PATH is not set. Proceeding with download and extraction..."

    # Download the fp16 raw weights of MLCommon hosted HF checkpoints
    download_file models SDXL/official_pytorch/fp16 \
        https://cloud.mlcommons.org/index.php/s/LCdW5RM6wgGWbxC/download \
        stable_diffusion_fp16.zip

    unzip ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16.zip \
        -d ${MODEL_DIR}/SDXL/official_pytorch/fp16/
fi

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/text_encoder/model.safetensors | grep "81b87e641699a4cd5985f47e99e71eeb"
if [ $? -ne 0 ]; then
    echo "SDXL CLIP1 fp16 model md5sum mismatch"
    exit -1
fi

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/text_encoder_2/model.safetensors | grep "5e540a9d92f6f88d3736189fd28fa6cd"
if [ $? -ne 0 ]; then
    echo "SDXL CLIP2 fp16 model md5sum mismatch"
    exit -1
fi

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/unet/diffusion_pytorch_model.safetensors | grep "edfa956683fb6121f717d095bf647f53"
if [ $? -ne 0 ]; then
    echo "SDXL UNet fp16 model md5sum mismatch"
    exit -1
fi

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/vae/diffusion_pytorch_model.safetensors | grep "25fe90074af9a0fe36d4a713ad5a3a29"
if [ $? -ne 0 ]; then
    echo "SDXL VAE fp16 model md5sum mismatch"
    exit -1
fi

# Run onnx generation script
python3 -m code.stable-diffusion-xl.tensorrt.create_onnx_model
test $? -eq 0 || exit $?
echo "Runing SDXL UNet quantization on 500 calibration captions. The process will take ~30 mins on DGX H100"
python3 -m code.stable-diffusion-xl.ammo.quantize_sdxl --pretrained-base ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/ --batch-size 1 \
    --calib-size 500 --calib-data code/stable-diffusion-xl/ammo/captions.tsv --percentile 0.4 \
    --n_steps 20 --latent ${DATA_DIR}/coco/SDXL/latents.pt --alpha 0.9 --quant-level 2.5 \
    --int8-ckpt-path ${MODEL_DIR}/SDXL/ammo_models/unetxl.int8.pt
test $? -eq 0 || exit $?
echo "Exporting SDXL fp16-int8 UNet onnx. The process will take ~60 mins on DGX H100"
python3 -m code.stable-diffusion-xl.ammo.export_onnx --pretrained-base ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/ --quantized-ckpt ${MODEL_DIR}/SDXL/ammo_models/unetxl.int8.pt --quant-level 2.5 --onnx-dir ${MODEL_DIR}/SDXL/ammo_models/unetxl.int8
test $? -eq 0 || exit $?
echo "SDXL model download and generation complete!"
