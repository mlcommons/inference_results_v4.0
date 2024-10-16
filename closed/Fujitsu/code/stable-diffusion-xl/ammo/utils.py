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

import re
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear


def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding).*"
    )
    return pattern.match(name) is not None


def quantize_lvl(unet, quant_level=2.5):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current ammo setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.input_quantizer.enable()
            module.weight_quantizer.enable()
        elif isinstance(module, torch.nn.Linear):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()


def lora_forward(self, x, scale=None):
    return self._torch_forward(x)


def replace_lora_layers(unet):
    for name, module in unet.named_modules():
        if isinstance(module, LoRACompatibleConv):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            bias = module.bias

            new_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias is not None,
            )

            new_conv.weight.data = module.weight.data.clone().to(module.weight.data.device)
            if bias is not None:
                new_conv.bias.data = module.bias.data.clone().to(module.bias.data.device)

            # Replace the LoRACompatibleConv layer with the Conv2d layer
            path = name.split(".")
            sub_module = unet
            for p in path[:-1]:
                sub_module = getattr(sub_module, p)
            setattr(sub_module, path[-1], new_conv)
            new_conv._torch_forward = new_conv.forward
            new_conv.forward = types.MethodType(lora_forward, new_conv)

        elif isinstance(module, LoRACompatibleLinear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias

            new_linear = nn.Linear(in_features, out_features, bias=bias is not None)

            new_linear.weight.data = module.weight.data.clone().to(module.weight.data.device)
            if bias is not None:
                new_linear.bias.data = module.bias.data.clone().to(module.bias.data.device)

            # Replace the LoRACompatibleLinear layer with the Linear layer
            path = name.split(".")
            sub_module = unet
            for p in path[:-1]:
                sub_module = getattr(sub_module, p)
            setattr(sub_module, path[-1], new_linear)
            new_linear._torch_forward = new_linear.forward
            new_linear.forward = types.MethodType(lora_forward, new_linear)


def get_smoothquant_config(model, quant_level=2.5):
    quant_config = {
        "quant_cfg": {},
        "algorithm": "smoothquant",
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if (
            w_name in quant_config["quant_cfg"].keys()  # type: ignore
            or i_name in quant_config["quant_cfg"].keys()  # type: ignore
        ):
            continue
        if filter_func(name):
            continue
        if isinstance(module, torch.nn.Linear):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}  # type: ignore
                quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}  # type: ignore
        elif isinstance(module, torch.nn.Conv2d):
            quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}  # type: ignore
            quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": None}  # type: ignore
    return quant_config


def load_calib_prompts(batch_size, calib_data_path):
    df = pd.read_csv(calib_data_path, sep="\t")
    lst = df["caption"].tolist()
    return [lst[i: i + batch_size] for i in range(0, len(lst), batch_size)]


class PercentileAmaxes:
    def __init__(self, total_step, percentile) -> None:
        self.data = {}
        self.total_step = total_step
        self.percentile = percentile
        self.i = 0

    def append(self, item):
        _cur_step = self.i % self.total_step
        if _cur_step not in self.data.keys():
            self.data[_cur_step] = item
        else:
            self.data[_cur_step] = np.minimum(self.data[_cur_step], item)
        self.i += 1
