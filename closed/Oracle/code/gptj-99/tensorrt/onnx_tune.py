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
import os
import torch
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file


def tune_scale_factors(fp8_scalers_path, scaler=1.01, index=15):
    """This function will replace the original fp8 quantization scalers in-place
    Args:
        fp8_scalers_path: safetensor path, for GPTJ, the fp8_scalers_path would be the path to the rank0.safetensors
        scaler: scale ratio of the amax
        index: targeting QKV layer index
    """
    def filter_layer(name, index=15):
        try:
            parts = name.split('transformer.layers.')[1].split('.')
            layer_index = int(parts[0])
            return layer_index > index
        except (IndexError, ValueError):
            return False

    gptj = {}
    with safe_open(fp8_scalers_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            gptj[key] = f.get_tensor(key)

    for key in gptj.keys():
        if filter_layer(name=key, index=index) and "qkv.activation_scaling_factor" in key:
            new_val = gptj[key] * scaler
            gptj[key] = new_val

    save_file(gptj, os.path.join(fp8_scalers_path))


def main():
    """
        Examples: python onnx_tune.py --fp8-scalers-path ./fp8-quantized-gptj/rank0.safetensors --scaler 1.01 --index 15
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8-scalers-path", default="./rank0.safetensors")
    parser.add_argument("--scaler", type=float, default=1.01)
    parser.add_argument("--index", type=int, default=15)
    args = parser.parse_args()

    tune_scale_factors(fp8_scalers_path=args.fp8_scalers_path, scaler=args.scaler, index=args.index)


if __name__ == "__main__":
    main()
