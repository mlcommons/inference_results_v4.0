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

import os
import argparse

import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
from importlib import import_module
from pathlib import Path

# dash in stable-diffusion-xl breaks traditional way of module import
plugin_calib = import_module("code.stable-diffusion-xl.ammo.plugin_calib")
get_smoothquant_config = import_module("code.stable-diffusion-xl.ammo.utils").get_smoothquant_config
load_calib_prompts = import_module("code.stable-diffusion-xl.ammo.utils").load_calib_prompts
replace_lora_layers = import_module("code.stable-diffusion-xl.ammo.utils").replace_lora_layers

import ammo.torch.quantization as atq


def do_calibrate(base, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        base(
            prompt=prompts,
            num_inference_steps=kwargs["n_steps"],
            negative_prompt=[
                "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
            ]
            * len(prompts),
            latents=kwargs["latent"],
            guidance_scale=8.0,  # MLPerf requirements
        ).images


def main():
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument("--int8-ckpt-path", default="./sdxl_int8.pt")
    parser.add_argument(
        "--pretrained-base", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument("--n_steps", type=int, default=20)

    # Calibration and quantization parameters
    parser.add_argument("--percentile", type=float, default=None, required=False)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--calib-size", type=int, default=500)
    parser.add_argument("--calib-data", type=str, default="./captions.tsv")
    parser.add_argument("--latent", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="SmoothQuant Alpha")
    parser.add_argument(
        "--quant-level",
        default=3.0,
        type=float,
        choices=[1.0, 2.0, 2.5, 3.0],
        help="Quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC",
    )

    args = parser.parse_args()

    args.calib_size = args.calib_size // args.batch_size

    base = DiffusionPipeline.from_pretrained(
        args.pretrained_base,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    base.to("cuda")
    base.scheduler = EulerDiscreteScheduler.from_config(base.scheduler.config)
    replace_lora_layers(base.unet)

    # This is a list of prompts
    cali_prompts = load_calib_prompts(args.batch_size, args.calib_data)
    quant_config = get_smoothquant_config(base.unet, args.quant_level)

    if args.percentile is not None:
        quant_config["percentile"] = args.percentile
        quant_config["base-step"] = int(args.n_steps)

    init_latent = None
    if args.latent is not None:
        init_latent = torch.load(args.latent).to(torch.float16)

    def forward_loop():
        do_calibrate(
            base=base,
            calibration_prompts=cali_prompts,
            calib_size=args.calib_size,
            n_steps=args.n_steps,
            latent=init_latent,
        )

    atq.replace_quant_module(base.unet)
    atq.set_quantizer_by_cfg(base.unet, quant_config["quant_cfg"])
    if args.percentile is not None:
        plugin_calib.precentile_calib_mode(base_unet=base.unet, quant_config=quant_config)
    if args.alpha is not None:
        plugin_calib.reg_alpha(base_unet=base.unet, alpha=args.alpha)
    plugin_calib.calibrate(base.unet, quant_config["algorithm"], forward_loop=forward_loop)

    torch_export_path = Path(args.int8_ckpt_path)
    torch_export_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        base.unet.state_dict(),
        torch_export_path,
    )


if __name__ == "__main__":
    main()
