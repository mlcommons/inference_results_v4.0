#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to preprocess the data for SDXL."""

import argparse
import gc
import torch
import pprint

from pathlib import Path
from code.common import logging
from importlib import import_module

CLIP = import_module("code.stable-diffusion-xl.tensorrt.network").CLIP
CLIPWithProj = import_module("code.stable-diffusion-xl.tensorrt.network").CLIPWithProj
UNetXL = import_module("code.stable-diffusion-xl.tensorrt.network").UNetXL
VAE = import_module("code.stable-diffusion-xl.tensorrt.network").VAE


def convert_torch_to_onnx(pytorch_model_dir, output_dir, model_name):
    """
    Convert pytorch model under pytorch_model_dir/model to onnx under output_dir/model
    """
    if model_name == 'text_encoder':
        network = CLIP(max_batch_size=1, device='cuda', verbose=False)
        onnx_name = 'clip1'
    elif model_name == 'text_encoder_2':
        network = CLIPWithProj(max_batch_size=1, device='cuda', verbose=False)
        onnx_name = 'clip2'
    elif model_name == 'unet':
        network = UNetXL(max_batch_size=1, device='cuda', verbose=False)
        onnx_name = 'unetxl'
    elif model_name == 'vae':
        network = VAE(max_batch_size=1, device='cuda', verbose=False)
        onnx_name = 'vae'
    else:
        logging.error("Unrecognized model: {}".format(model_name))

    pytorch_model_path = Path(pytorch_model_dir, model_name)
    model = network.get_model(pytorch_model_path)
    inputs = network.get_sample_input(batch_size=1)

    output_path = Path(output_dir, onnx_name)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / 'model.onnx'

    if model_name == 'vae':
        logging.info(f"Exporting vae in fp32")
        with torch.inference_mode():
            torch.onnx.export(model,
                              inputs,
                              output_path.as_posix(),
                              export_params=True,
                              opset_version=17,
                              do_constant_folding=True,
                              input_names=network.get_input_names(),
                              output_names=network.get_onnx_output_names(),
                              dynamic_axes=network.get_dynamic_axes(),
                              )
    else:
        logging.info(f"Exporting {onnx_name} in fp16")
        with torch.inference_mode(), torch.autocast("cuda"):
            torch.onnx.export(model,
                              inputs,
                              output_path.as_posix(),
                              export_params=True,
                              opset_version=17,
                              do_constant_folding=True,
                              input_names=network.get_input_names(),
                              output_names=network.get_onnx_output_names(),
                              dynamic_axes=network.get_dynamic_axes(),
                              )

    logging.info("Dynamic axes:")
    pprint.pprint(network.get_dynamic_axes())

    del model
    torch.cuda.empty_cache()
    gc.collect()


def generate_onnx_sdxl(model_dir, output_dir):
    pytorch_model_dir = Path(model_dir, "SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe")
    output_dir = Path(output_dir, "SDXL", "onnx_models")

    models = ['text_encoder', 'text_encoder_2', 'unet', 'vae']
    for model in models:
        logging.info("Creating SDXL {} onnx...".format(model))
        convert_torch_to_onnx(pytorch_model_dir, output_dir, model)
        logging.info("Done creating {} onnx...".format(model))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", "-m",
        help="Directory containing the pytorch models.",
        default="build/models"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for the onnx models.",
        default="build/models"
    )
    args = parser.parse_args()
    model_dir = args.model_dir
    output_dir = args.output_dir

    generate_onnx_sdxl(model_dir, output_dir)

    logging.info("SDXL ONNX Model Generation Is Done!")


if __name__ == '__main__':
    main()
