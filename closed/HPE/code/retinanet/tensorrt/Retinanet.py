#!/usr/bin/env python3
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

from __future__ import annotations
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Optional

# The plugin .so file has to be loaded at global scope and before `import torch` to avoid cuda version mismatch.
from code.plugin import load_trt_plugin_by_network
load_trt_plugin_by_network("retinanet")

import tensorrt as trt
import onnx
import re

from nvmitten.constants import Precision
from nvmitten.nvidia.builder import (get_dyn_ranges,
                                     TRTBuilder,
                                     CalibratableTensorRTEngine,
                                     MLPerfInferenceEngine,
                                     LegacyBuilder)
from nvmitten.pipeline import Operation
from nvmitten.utils import logging

from code.common.fields import Fields
from code.common.mitten_compat import ArgDiscarder
from code.common.systems.system_list import DETECTED_SYSTEM

RetinanetEntropyCalibrator = import_module("code.retinanet.tensorrt.calibrator").RetinanetEntropyCalibrator
RetinanetGraphSurgeon = import_module("code.retinanet.tensorrt.retinanet_graphsurgeon").RetinanetGraphSurgeon

INPUT_SHAPE = (3, 800, 800)

retinanet_tactic_dict = {
    "res2a_branch2c_conv \+ res2a_branch2c_scale_value \+ res2a_branch2c_scale \+ res2a_branch2c_bias_value \+ res2a_branch2c_bias": "0xe7e31a63934a12f0",
    "res2a_branch1_conv \+ res2a_branch1_scale_value \+ res2a_branch1_scale \+ res2a_branch1_bias_value \+ res2a_branch1_bias \+ res2a_residual_add \+ res2a_relu": "0x5e4918ccf433630e",
    "res2b_branch2a_conv \+ res2b_branch2a_scale_value \+ res2b_branch2a_scale \+ res2b_branch2a_bias_value \+ res2b_branch2a_bias \+ res2b_branch2a_relu": "0xa41ea2e47a90dd1c",
    "res2c_branch2a_conv \+ res2c_branch2a_scale_value \+ res2c_branch2a_scale \+ res2c_branch2a_bias_value \+ res2c_branch2a_bias \+ res2c_branch2a_relu": "0xa41ea2e47a90dd1c",
    "res3a_branch2a_conv \+ res3a_branch2a_scale_value \+ res3a_branch2a_scale \+ res3a_branch2a_bias_value \+ res3a_branch2a_bias \+ res3a_branch2a_relu": "0x1e8fa12cff3e3bdb",
    "res3a_branch1_conv \+ res3a_branch1_scale_value \+ res3a_branch1_scale \+ res3a_branch1_bias_value \+ res3a_branch1_bias": "0xdf9672025c2e4e0b",
    "res3b_branch2a_conv \+ res3b_branch2a_scale_value \+ res3b_branch2a_scale \+ res3b_branch2a_bias_value \+ res3b_branch2a_bias \+ res3b_branch2a_relu": "0x5d63ba6b5cfeb06f",
    "res3c_branch2a_conv \+ res3c_branch2a_scale_value \+ res3c_branch2a_scale \+ res3c_branch2a_bias_value \+ res3c_branch2a_bias \+ res3c_branch2a_relu": "0x5d63ba6b5cfeb06f",
    "res3d_branch2a_conv \+ res3d_branch2a_scale_value \+ res3d_branch2a_scale \+ res3d_branch2a_bias_value \+ res3d_branch2a_bias \+ res3d_branch2a_relu": "0x5d63ba6b5cfeb06f",
    "res4a_branch2a_conv \+ res4a_branch2a_scale_value \+ res4a_branch2a_scale \+ res4a_branch2a_bias_value \+ res4a_branch2a_bias \+ res4a_branch2a_relu": "0x4c7f57d11ee08da7",
    "res4a_branch1_conv \+ res4a_branch1_scale_value \+ res4a_branch1_scale \+ res4a_branch1_bias_value \+ res4a_branch1_bias": "0x5353e9f102bbfecb",
    "res4b_branch2a_conv \+ res4b_branch2a_scale_value \+ res4b_branch2a_scale \+ res4b_branch2a_bias_value \+ res4b_branch2a_bias \+ res4b_branch2a_relu": "0x8d507197d88f65f7",
    "res4c_branch2a_conv \+ res4c_branch2a_scale_value \+ res4c_branch2a_scale \+ res4c_branch2a_bias_value \+ res4c_branch2a_bias \+ res4c_branch2a_relu": "0x8d507197d88f65f8",
    "res4d_branch2a_conv \+ res4d_branch2a_scale_value \+ res4d_branch2a_scale \+ res4d_branch2a_bias_value \+ res4d_branch2a_bias \+ res4d_branch2a_relu": "0x8d507197d88f65f9",
    "res4e_branch2a_conv \+ res4e_branch2a_scale_value \+ res4e_branch2a_scale \+ res4e_branch2a_bias_value \+ res4e_branch2a_bias \+ res4e_branch2a_relu": "0x8d507197d88f65f10",
    "res4f_branch2a_conv \+ res4f_branch2a_scale_value \+ res4f_branch2a_scale \+ res4f_branch2a_bias_value \+ res4f_branch2a_bias \+ res4f_branch2a_relu": "0x8d507197d88f65f11",
    "res5a_branch2a_conv \+ res5a_branch2a_scale_value \+ res5a_branch2a_scale \+ res5a_branch2a_bias_value \+ res5a_branch2a_bias \+ res5a_branch2a_relu": "0xf6c902b1b42bb2b8",
    "res5a_branch1_conv \+ res5a_branch1_scale_value \+ res5a_branch1_scale \+ res5a_branch1_bias_value \+ res5a_branch1_bias": "0x46549cb5b134b984",
    "res5b_branch2a_conv \+ res5b_branch2a_scale_value \+ res5b_branch2a_scale \+ res5b_branch2a_bias_value \+ res5b_branch2a_bias \+ res5b_branch2a_relu": "0x64405ad349837235",
    "res5c_branch2a_conv \+ res5c_branch2a_scale_value \+ res5c_branch2a_scale \+ res5c_branch2a_bias_value \+ res5c_branch2a_bias \+ res5c_branch2a_relu": "0x64405ad349837236",
    "res5b_branch2b_conv \+ res5b_branch2b_scale_value \+ res5b_branch2b_scale \+ res5b_branch2b_bias_value \+ res5b_branch2b_bias \+ res5b_branch2b_relu": "0x53554c607d072469",
    "res5c_branch2b_conv \+ res5c_branch2b_scale_value \+ res5c_branch2b_scale \+ res5c_branch2b_bias_value \+ res5c_branch2b_bias \+ res5c_branch2b_relu": "0x53554c607d072470",
    "\/backbone\/fpn\/inner_blocks.2\/Conv": "0x743ccad8b4fb4cdc",
    "\/backbone\/fpn\/layer_blocks.2\/Conv": "0x8d507197d88f65f7",
    "\/backbone\/fpn\/extra_blocks\/p6\/Conv": "0xbe4bcb63b11804b0",
    "\/head\/classification_head\/conv\/conv.0_2\/Conv \+ \/head\/classification_head\/conv\/conv.1_2\/Relu \|\| \/head\/regression_head\/conv\/conv.0_2\/Conv \+ \/head\/regression_head\/conv\/conv.1_2\/Relu": "0x64405ad349837235",
    "\/head\/classification_head\/conv\/conv.0_3\/Conv \+ \/head\/classification_head\/conv\/conv.1_3\/Relu \|\| \/head\/regression_head\/conv\/conv.0_3\/Conv \+ \/head\/regression_head\/conv\/conv.1_3\/Relu": "0x0fa5b9fed85f9b93",
    "\/backbone\/fpn\/extra_blocks\/p7\/Conv": "0xe927f99036cd867d",
    "\/head\/classification_head\/conv\/conv.2_2\/Conv \+ \/head\/classification_head\/conv\/conv.3_2\/Relu": "0x8c3f36379fc56733",
    "\/head\/regression_head\/conv\/conv.2_2\/Conv \+ \/head\/regression_head\/conv\/conv.3_2\/Relu": "0x8c3f36379fc56733",
    "\/head\/classification_head\/conv\/conv.0_4\/Conv \+ \/head\/classification_head\/conv\/conv.1_4\/Relu \|\| \/head\/regression_head\/conv\/conv.0_4\/Conv \+ \/head\/regression_head\/conv\/conv.1_4\/Relu": "0xbe4bcb63b11804b0",
    "\/head\/classification_head\/conv\/conv.2_3\/Conv \+ \/head\/classification_head\/conv\/conv.3_3\/Relu": "0x63a17049059f4b74",
    "\/head\/regression_head\/conv\/conv.2_3\/Conv \+ \/head\/regression_head\/conv\/conv.3_3\/Relu": "0x63a17049059f4b74",
    "\/backbone\/fpn\/inner_blocks.1\/Conv \+ \/backbone\/fpn\/Add": "0xdfdddae7a4bcc830",
    "\/head\/classification_head\/conv\/conv.4_2\/Conv \+ \/head\/classification_head\/conv\/conv.5_2\/Relu": "0x743ccad8b4fb4cdc",
    "\/head\/regression_head\/conv\/conv.4_2\/Conv \+ \/head\/regression_head\/conv\/conv.5_2\/Relu": "0x743ccad8b4fb4cdc",
    "\/backbone\/fpn\/layer_blocks.1\/Conv": "0xa40f0124308a9944",
    "\/head\/classification_head\/conv\/conv.2_4\/Conv \+ \/head\/classification_head\/conv\/conv.3_4\/Relu": "0xb2f0016f82a78152",
    "\/head\/regression_head\/conv\/conv.2_4\/Conv \+ \/head\/regression_head\/conv\/conv.3_4\/Relu": "0xb2f0016f82a78152",
    "\/head\/classification_head\/conv\/conv.4_3\/Conv \+ \/head\/classification_head\/conv\/conv.5_3\/Relu": "0xbe4bcb63b11804b0",
    "\/head\/regression_head\/conv\/conv.4_3\/Conv \+ \/head\/regression_head\/conv\/conv.5_3\/Relu": "0xbe4bcb63b11804b0",
    "\/head\/classification_head\/conv\/conv.0_1\/Conv \+ \/head\/classification_head\/conv\/conv.1_1\/Relu \|\| \/head\/regression_head\/conv\/conv.0_1\/Conv \+ \/head\/regression_head\/conv\/conv.1_1\/Relu": "0xe7d5ef0bf6358f70",
    "\/head\/classification_head\/conv\/conv.6_2\/Conv \+ \/head\/classification_head\/conv\/conv.7_2\/Relu": "0x743ccad8b4fb4cdc",
    "\/head\/regression_head\/conv\/conv.6_2\/Conv \+ \/head\/regression_head\/conv\/conv.7_2\/Relu": "0x743ccad8b4fb4cdc",
    "\/head\/classification_head\/conv\/conv.4_4\/Conv \+ \/head\/classification_head\/conv\/conv.5_4\/Relu": "0x3e5b5ece3eb421f6",
    "\/head\/regression_head\/conv\/conv.4_4\/Conv \+ \/head\/regression_head\/conv\/conv.5_4\/Relu": "0x3e5b5ece3eb421f6",
    "\/head\/classification_head\/conv\/conv.6_3\/Conv \+ \/head\/classification_head\/conv\/conv.7_3\/Relu": "0xbe4bcb63b11804b0",
    "\/head\/regression_head\/conv\/conv.6_3\/Conv \+ \/head\/regression_head\/conv\/conv.7_3\/Relu": "0xbe4bcb63b11804b0",
    "\/head\/classification_head\/conv\/conv.2_1\/Conv \+ \/head\/classification_head\/conv\/conv.3_1\/Relu": "0x95cf40af74f98aba",
    "\/head\/regression_head\/conv\/conv.2_1\/Conv \+ \/head\/regression_head\/conv\/conv.3_1\/Relu": "0x95cf40af74f98aba",
    "\/head\/classification_head\/cls_logits_2\/Conv": "0x44a81ac017da0445",
    "\/head\/regression_head\/bbox_reg_2\/Conv": "0x23b890da05937b9e",
    "\/head\/classification_head\/conv\/conv.6_4\/Conv \+ \/head\/classification_head\/conv\/conv.7_4\/Relu": "0x3e5b5ece3eb421f6",
    "\/head\/regression_head\/conv\/conv.6_4\/Conv \+ \/head\/regression_head\/conv\/conv.7_4\/Relu": "0x3e5b5ece3eb421f6",
    "\/head\/classification_head\/cls_logits_3\/Conv": "0x4ce968916c7f46ae",
    "\/head\/regression_head\/bbox_reg_3\/Conv": "0x85c1a5f7f239cf84",
    "\/head\/classification_head\/conv\/conv.4_1\/Conv \+ \/head\/classification_head\/conv\/conv.5_1\/Relu": "0xa40f0124308a9944",
    "\/head\/regression_head\/conv\/conv.4_1\/Conv \+ \/head\/regression_head\/conv\/conv.5_1\/Relu": "0xa40f0124308a9944",
    "\/backbone\/fpn\/layer_blocks.0\/Conv": "0x388a7ba5b9327e20",
    "\/head\/classification_head\/conv\/conv.0\/Conv \+ \/head\/classification_head\/conv\/conv.1\/Relu \|\| \/head\/regression_head\/conv\/conv.0\/Conv \+ \/head\/regression_head\/conv\/conv.1\/Relu": "0xe7d5ef0bf6358f70",
    "\/head\/classification_head\/cls_logits_4\/Conv": "0xa9eb6a7edadba006",
    "\/head\/regression_head\/bbox_reg_4\/Conv": "0x85c1a5f7f239cf84",
    "\/head\/classification_head\/conv\/conv.6_1\/Conv \+ \/head\/classification_head\/conv\/conv.7_1\/Relu": "0xa40f0124308a9944",
    "\/head\/regression_head\/conv\/conv.6_1\/Conv \+ \/head\/regression_head\/conv\/conv.7_1\/Relu": "0xa40f0124308a9944",
    "\/head\/classification_head\/conv\/conv.2\/Conv \+ \/head\/classification_head\/conv\/conv.3\/Relu": "0xd470dbb02604d85e",
    "\/head\/regression_head\/conv\/conv.2\/Conv \+ \/head\/regression_head\/conv\/conv.3\/Relu": "0xd470dbb02604d85e",
    "\/head\/classification_head\/cls_logits_1\/Conv": "0x86f8ad560d6764ad",
    "\/head\/regression_head\/bbox_reg_1\/Conv": "0xd1827e195c0f06b1",
    "\/head\/classification_head\/conv\/conv.4\/Conv \+ \/head\/classification_head\/conv\/conv.5\/Relu": "0x388a7ba5b9327e20",
    "\/head\/regression_head\/conv\/conv.4\/Conv \+ \/head\/regression_head\/conv\/conv.5\/Relu": "0x388a7ba5b9327e20",
    "\/head\/classification_head\/conv\/conv.6\/Conv \+ \/head\/classification_head\/conv\/conv.7\/Relu": "0x388a7ba5b9327e20",
    "\/head\/regression_head\/conv\/conv.6\/Conv \+ \/head\/regression_head\/conv\/conv.7\/Relu": "0x388a7ba5b9327e20",
    "\/head\/classification_head\/cls_logits\/Conv": "0x86f8ad560d6764ad"
}


class HopperTacticSelector(trt.IAlgorithmSelector):
    def select_algorithms(self, ctx, choices):
        print("\nselect algorithms: " + ctx.name)
        retinanet_layer_pattern = re.compile('|'.join(retinanet_tactic_dict))
        if re.search(retinanet_layer_pattern, ctx.name):
            print("Matched")
            for layer_regex, tactic_id in retinanet_tactic_dict.items():
                if re.match(layer_regex, ctx.name):
                    filtered_idxs = [idx for idx, choice in enumerate(choices) if choice.algorithm_variant.tactic == int(tactic_id, 16)]
                    to_ret = filtered_idxs
                    print("Filtered id: {} for regex {} ".format(tactic_id, layer_regex))
        else:
            to_ret = [idx for idx, _ in enumerate(choices)]
        return to_ret

    def report_algorithms(self, ctx, choices):
        pass


class RetinanetEngineBuilderOp(CalibratableTensorRTEngine,
                               TRTBuilder,
                               MLPerfInferenceEngine,
                               Operation,
                               ArgDiscarder):
    @classmethod
    def immediate_dependencies(cls):
        # TODO: Integrate dataset scripts as deps
        return None

    def __init__(self,
                 # TODO: Legacy value - Remove after refactor is done.
                 config_ver: str = "default",
                 # TODO: This should be a relative path within the ScratchSpace.
                 model_path: str = "build/models/retinanet-resnext50-32x4d/retinanet-fpn-torch2.1-postprocessed.onnx",
                 # Override the normal default values
                 workspace_size: int = 8 << 30,
                 calib_batch_size: int = 10,
                 calib_max_batches: int = 50,
                 calib_data_map: PathLike = Path("data_maps/open-images-v6-mlperf/cal_map.txt"),
                 cache_file: PathLike = Path("code/retinanet/tensorrt/calibrator.cache"),
                 # Benchmark specific values
                 batch_size: int = 1,
                 nms_type: str = "nmsopt",
                 disable_beta1_smallk: bool = False,
                 energy_aware_kernels: bool = False,
                 **kwargs):
        super().__init__(workspace_size=workspace_size,
                         calib_batch_size=calib_batch_size,
                         calib_max_batches=calib_max_batches,
                         calib_data_map=calib_data_map,
                         cache_file=cache_file,
                         **kwargs)

        self.config_ver = config_ver
        self.model_path = model_path
        self.batch_size = batch_size
        assert nms_type in {"efficientnms", "nmsopt", "nmspva"}, "Unrecognized nms_type, please select from efficientnms, nmsopt, nmspva"
        self.nms_type = nms_type
        self.disable_beta1_smallk = disable_beta1_smallk
        self.energy_aware_kernels = energy_aware_kernels

        self.device_type = "dla" if self.dla_enabled else "gpu"
        if self.device_type == "gpu" and self.nms_type != "efficientnms":
            self.nms_type = "nmsopt"  # for jetson submission sets nms_type to nmspva, reset nms_type to nmsopt for GPU engine

        # Determine required subnetworks to generate engines
        self.subnetworks = None

    def set_calibrator(self, image_dir):
        if self.precision != Precision.INT8:
            return

        self.calibrator = RetinanetEntropyCalibrator(image_dir,
                                                     self.cache_file,
                                                     self.calib_batch_size,
                                                     self.calib_max_batches,
                                                     self.force_calibration,
                                                     self.calib_data_map)

    def create_network(self, builder: trt.Builder, subnetwork_name: str = None):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        compute_sm = DETECTED_SYSTEM.get_compute_sm()

        # Parse from ONNX file
        parser = trt.OnnxParser(network, self.logger)
        retinanet_gs = RetinanetGraphSurgeon(self.model_path,
                                             self.precision,
                                             self.cache_file,
                                             compute_sm,
                                             self.device_type,
                                             self.need_calibration,
                                             disable_beta1_smallk=self.disable_beta1_smallk,
                                             nms_type=self.nms_type)
        if subnetwork_name:
            subnet = RetinanetGraphSurgeon.subnetwork_map[subnetwork_name]
        else:
            subnet = None
        model = retinanet_gs.create_onnx_model(subnetwork=subnet)
        success = parser.parse(onnx._serialize(model))
        if not success:
            err_desc = parser.get_error(0).desc()
            raise RuntimeError(f"Retinanet onnx model processing failed! Error: {err_desc}")

        # Check input and output tensor names
        if subnet:
            if subnet.inputs:
                actual = {network.get_input(i).name for i in range(network.num_inputs)}
                expected = {tens.name for tens in subnet.inputs}
                assert actual == expected, f"Subnetwork input mismatch: Got: {actual}, Expected: {expected}"
            if subnet.outputs:
                actual = {network.get_output(i).name for i in range(network.num_outputs)}
                expected = {tens.name for tens in subnet.outputs}
                assert actual == expected, f"Subnetwork output mismatch: Got: {actual}, Expected: {expected}"

        self.apply_subnetwork_io_types(network, subnetwork_name)
        return network

    def apply_network_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for network inputs and outputs to the tensorrt.INetworkDefinition.


        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        # Set input dtype and format
        input_tensor = network.get_input(0)
        if self.input_dtype == "int8":
            input_tensor.dtype = trt.int8
            dynamic_range_dict = dict()
            if self.cache_file.exists():
                dynamic_range_dict = get_dyn_ranges(self.cache_file)
                input_dr = dynamic_range_dict.get("images", -1)
                if input_dr == -1:
                    raise RuntimeError(f"Cannot find 'images' in the calibration cache. Exiting...")
                input_tensor.set_dynamic_range(-input_dr, input_dr)
            else:
                print("WARNING: Calibration cache file not found! Calibration is required")
        if self.input_format == "linear":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        elif self.input_format == "chw4":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

    def apply_subnetwork_io_types(self, network: trt.INetworkDefinition, subnetwork_name: str):
        """Applies I/O dtype and formats for subnetwork inputs and outputs to the tensorrt.INetworkDefinition.

        Note: Currently in Mitten, ONNXNetwork.Subnetwork does not support tensor data format or dynamic range
        specification, as well as expected devices. Hence, we take the subnetwork name instead of the subnetwork
        description itself as an argument.

        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
            subnetwork_name (str): The name or ID of the subnetwork
        """
        dynamic_range_dict = dict()
        dynamic_range_dict = get_dyn_ranges(self.cache_file)
        num_tensor_in = network.num_inputs
        num_tensor_out = network.num_outputs

        if not subnetwork_name:
            for i in range(num_tensor_in):
                tensor_in = network.get_input(i)
                input_dr = dynamic_range_dict.get(tensor_in.name, -1)
                if input_dr == -1:
                    raise RuntimeError(f"Cannot find {tensor_in.name} in the calibration cache. Exiting...")
                self._set_tensor_format(tensor_in, use_dla=self.dla_enabled)
        elif subnetwork_name == "dla":
            for i in range(num_tensor_in):
                tensor_in = network.get_input(i)
                input_dr = dynamic_range_dict.get(tensor_in.name, -1)
                if input_dr == -1:
                    raise RuntimeError(f"Cannot find {tensor_in.name} in the calibration cache. Exiting...")

                logging.debug(f"tensor_in: {tensor_in.name} | previous dynamic range: {tensor_in.dynamic_range} | to set dynamic range: {input_dr}")
                # enforce IO format to avoid reformats for the DLA node
                self._set_tensor_format(tensor_in, use_dla=self.dla_enabled, tensor_format=trt.TensorFormat.DLA_HWC4, dynamic_range=input_dr)
            for i in range(num_tensor_out):
                tensor_out = network.get_output(i)
                output_dr = dynamic_range_dict.get(tensor_out.name, -1)
                if output_dr == -1:
                    raise RuntimeError(f"Cannot find {tensor_out.name} in the calibration cache. Exiting...")
                # enforce IO format to avoid reformats for the DLA node
                logging.debug(f"tensor_out: {tensor_out.name} | previous dynamic range: {tensor_out.dynamic_range} | to set dynamic range: {output_dr}")
                self._set_tensor_format(tensor_out, use_dla=self.dla_enabled, tensor_format=trt.TensorFormat.DLA_LINEAR, dynamic_range=output_dr)
        elif subnetwork_name == "nmspva":
            for i in range(num_tensor_in):
                tensor_in = network.get_input(i)
                # PVA plugin has to read non DLA tensor format so we trick TRT to think the tensors from DLA are in linear layour
                # However, PVA plugin reads the DLA_LINEAR tensor layout during runtime. So this is fine
                input_dr = dynamic_range_dict.get(tensor_in.name, -1)
                if input_dr == -1:
                    raise RuntimeError(f"Cannot find {tensor_in.name} in the calibration cache. Exiting...")
                logging.debug(f"tensor_in: {tensor_in.name} | previous dynamic range: {tensor_in.dynamic_range} | to set dynamic range: {input_dr}")
                self._set_tensor_format(tensor_in, use_dla=self.dla_enabled, tensor_format=trt.TensorFormat.LINEAR, dynamic_range=input_dr)
        else:
            raise RuntimeError(f"RetinaNet initialize failed! Invalid subnetwork: {subnetwork_name}")

    def _set_tensor_format(self,
                           tensor: trt.ITensor,
                           use_dla: bool = False,
                           tensor_format: Optional[trt.TensorFormat] = None,
                           dynamic_range: Optional[Tuple[int, int]] = None):
        """Set input tensor dtype and format.

        Args:
            input_tensor_name (str): The tensor to modify
            use_dla (bool): If True, uses DLA input formats if applicable. (Default: False)
            tensor_format (trt.TensorFormat): Overrides the tensor format to set the input to. If not set, uses
                                              self.input_format. (Default: None)
            dynamic_range (Tuple[int, int]): A tuple of length 2 in the format [min_value (inclusive), max_value
                                             (inclusive)]. This argument is ignored if the input tensor is not in INT8
                                             precision. (Default: None)
        """
        # Apply dynamic ranges for INT8 inputs
        if self.input_dtype == "int8":
            tensor.dtype = trt.int8
            if dynamic_range is not None:
                tensor.set_dynamic_range(-dynamic_range, dynamic_range)

        if not tensor_format:
            # Set the same format as the input data if not specified
            if self.input_format == "linear":
                tensor_format = trt.TensorFormat.LINEAR
            elif self.input_format == "chw4":
                # WAR for DLA reformat bug in https://nvbugs/3713387:
                # For resnet50, inputs dims are [3, 224, 224].
                # For those particular dims, CHW4 == DLA_HWC4, so can use same CHW4 data for both GPU and DLA engine.
                # By lying to TRT and saying input is DLA_HWC4, we elide the pre-DLA reformat layer.
                if use_dla:
                    tensor_format = trt.TensorFormat.DLA_HWC4
                else:
                    tensor_format = trt.TensorFormat.CHW4
        tensor.allowed_formats = 1 << int(tensor_format)

    def run(self, scratch_space, dependency_outputs):
        # Set up INT8 calibration
        subdir = Path("open-images-v6-mlperf/calibration/Retinanet/fp32")
        self.set_calibrator(scratch_space.path / "preprocessed_data" / subdir)

        # Create builder config
        builder_config = self.create_builder_config()
        builder_config.int8_calibrator = self.calibrator

        if self.energy_aware_kernels:
            tactic_selector = HopperTacticSelector()
            builder_config.algorithm_selector = tactic_selector

        # If no subnetworks are needed, use default parameters
        if not self.subnetworks:
            self.subnetworks = [(None, self.batch_size)]

        for subnet_name, batch_size in self.subnetworks:
            # Enforce Direct IO for PVA plugin
            if self.device_type == "dla" and self.nms_type == "nmspva":
                builder_config.set_flag(trt.BuilderFlag.DIRECT_IO)
            network = self.create_network(self.builder, subnetwork_name=subnet_name)
            engine_dir = self.engine_dir(scratch_space)
            engine_name = self.engine_name(self.device_type,
                                           batch_size,
                                           self.precision,
                                           subnetwork_name=subnet_name,
                                           tag=self.config_ver)
            engine_fpath = engine_dir / engine_name
            self.build_engine(network, builder_config, batch_size, engine_fpath)


class Retinanet(LegacyBuilder):
    """Temporary spoofing class to wrap around Mitten to adhere to the old API.
    """

    def __init__(self, args):
        super().__init__(RetinanetEngineBuilderOp(**args))
