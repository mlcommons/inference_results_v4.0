# Adapted https://github.com/huggingface/optimum/blob/15a162824d0c5d8aa7a3d14ab6e9bb07e5732fb6/optimum/exporters/onnx/convert.py#L573-L614

# Copyright 2022 The HuggingFace Team. All rights reserved.
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

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import os
from pathlib import Path

import onnx
import torch
from optimum.onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
from torch.onnx import export as onnx_export

AXES_NAME = {
    "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    "timestep": {0: "steps"},
    "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
    "text_embeds": {0: "batch_size"},
    "time_ids": {0: "batch_size"},
    "latent": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
}


def generate_dummy_inputs(device):
    dummy_input = {}
    dummy_input["sample"] = torch.ones(2, 4, 128, 128).to(device)
    dummy_input["timestep"] = torch.ones(1).to(device)
    dummy_input["encoder_hidden_states"] = torch.ones(2, 77, 2048).to(device)
    dummy_input["added_cond_kwargs"] = {}
    dummy_input["added_cond_kwargs"]["text_embeds"] = torch.ones(2, 1280).to(device)
    dummy_input["added_cond_kwargs"]["time_ids"] = torch.ones(2, 6).to(device)

    return dummy_input


def ammo_export_sd(base, exp_name):
    exp_path = Path(exp_name)
    os.makedirs(exp_path, exist_ok=True)
    dummy_inputs = generate_dummy_inputs(device=base.unet.device)

    output = Path(exp_path, "unet.onnx")

    input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
    output_names = ["latent"]

    dynamic_axes = AXES_NAME
    do_constant_folding = True
    opset_version = 14

    # Copied from Huggingface's Optimum
    onnx_export(
        base.unet,
        (dummy_inputs,),
        f=output.as_posix(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
        opset_version=opset_version,
    )

    onnx_model = onnx.load(str(output), load_external_data=False)
    model_uses_external_data = check_model_uses_external_data(onnx_model)

    if model_uses_external_data:
        tensors_paths = _get_onnx_external_data_tensors(onnx_model)
        onnx_model = onnx.load(str(output), load_external_data=True)
        onnx.save(
            onnx_model,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=output.name + "_data",
            size_threshold=1024,
        )
        for tensor in tensors_paths:
            os.remove(output.parent / tensor)
