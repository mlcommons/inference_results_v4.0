# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import re

import torch
from diffusers import DiffusionPipeline
from importlib import import_module

ammo_export_sd = import_module("code.stable-diffusion-xl.ammo.export").ammo_export_sd
replace_lora_layers = import_module("code.stable-diffusion-xl.ammo.utils").replace_lora_layers
quantize_lvl = import_module("code.stable-diffusion-xl.ammo.utils").quantize_lvl
filter_func = import_module("code.stable-diffusion-xl.ammo.utils").filter_func

import ammo.torch.quantization as atq


def main():
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument("--onnx-dir", default=None)
    parser.add_argument(
        "--pretrained-base",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument(
        "--quantized-ckpt",
        type=str,
        default="./base.unet.state_dict.fp8.0.25.384.percentile.all.pt",
    )
    parser.add_argument("--format", default="int8", choices=["int8"])  # Now only support int8
    parser.add_argument(
        "--quant-level",
        default=3.0,
        type=float,
        choices=[1.0, 2.0, 2.5, 3.0],
        help="Quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC",
    )
    parser.add_argument('--device', default='cpu', help='device')
    args = parser.parse_args()
    device = torch.device(args.device)

    base = DiffusionPipeline.from_pretrained(
        args.pretrained_base,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    replace_lora_layers(base.unet)
    atq.replace_quant_module(base.unet)
    quant_config = atq.INT8_DEFAULT_CFG if args.format == "int8" else atq.FP8_DEFAULT_CFG
    atq.set_quantizer_by_cfg(base.unet, quant_config["quant_cfg"])

    base.unet.load_state_dict(torch.load(args.quantized_ckpt), strict=True)
    quantize_lvl(base.unet, args.quant_level)
    atq.disable_quantizer(base.unet, filter_func)

    # QDQ needs to be in FP32
    base.unet.to(torch.float32).to(device)
    ammo_export_sd(base, f"{str(args.onnx_dir)}")


if __name__ == "__main__":
    main()
