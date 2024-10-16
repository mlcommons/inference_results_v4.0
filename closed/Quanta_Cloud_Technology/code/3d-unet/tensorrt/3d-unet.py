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
from typing import Optional, Tuple

from code.plugin import load_trt_plugin_by_network
load_trt_plugin_by_network("3d-unet-kits")

import onnx
import onnx_graphsurgeon as gs
import numpy as np
import tensorrt as trt

from nvmitten.constants import Precision
from nvmitten.nvidia.builder import (get_dyn_ranges,
                                     TRTBuilder,
                                     CalibratableTensorRTEngine,
                                     MLPerfInferenceEngine,
                                     ONNXNetwork,
                                     LegacyBuilder)
from nvmitten.pipeline import Operation
from nvmitten.utils import logging

from code.common.fields import Fields
from code.common.mitten_compat import ArgDiscarder
from code.common.systems.system_list import DETECTED_SYSTEM

UNet3DKiTS19MinMaxCalibrator = import_module("code.3d-unet.tensorrt.calibrator").UNet3DKiTS19MinMaxCalibrator


class UnetGraphSurgeon(ONNXNetwork):

    def __init__(self,
                 onnx_path: PathLike,
                 precision: Precision,
                 calib_cache_path: os.PathLike,
                 enable_instnorm3d_plugin: bool = True,
                 pixel_shuffle_cdwh: bool = True):
        super().__init__(onnx_path, precision, calib_cache_path=calib_cache_path)

        self.enable_instnorm3d_plugin = enable_instnorm3d_plugin
        self.pixel_shuffle_cdwh = pixel_shuffle_cdwh

        self.enable_plugin_optimizations = precision in [Precision.INT8, Precision.FP32]

    def fuse_ops(self):
        if self.enable_instnorm3d_plugin:
            self.instnorm3d_plugin()

        # Previous implementation had individual knobs for all of these optimizations, but all were set to the value of
        # `self.enable_plugin_optimizations`. If necessary, `disable_*` params can be added to explicitly disable these
        # plugins even if the precision is correct.
        if self.enable_plugin_optimizations:
            self.conv3d1x1x1k4_plugin()
            self.conv3d3x3x3c1k32_plugin()
            self.conv_for_deconv(pixel_shuffle_cdwh=True,
                                 enable_pixelshuffle3d_plugin=True,
                                 enable_pixelshuffle3d_plugin_concat_fuse=True)

    def instnorm3d_plugin(self):
        for node in self.graph.nodes:
            # Replace InstanceNormalization with InstanceNormalization_TRT plugin node
            if node.op == "InstanceNormalization":
                node.op = "InstanceNormalization_TRT"
                node.attrs["scales"] = node.inputs[1]
                node.attrs["bias"] = node.inputs[2]
                node.attrs["plugin_version"] = "2"
                node.attrs["plugin_namespace"] = ""
                node.attrs["relu"] = 0
                node.attrs["alpha"] = 0.0
                scales = node.attrs["scales"].values
                biases = node.attrs["bias"].values
                assert len(scales) == len(biases), "Scales and biases do not have the same length!"
                del node.inputs[2]
                del node.inputs[1]

        # Set relu node attributes to INSTNORM3D plugin and remove relu nodes.
        nodes = [node for node in self.graph.nodes if node.op == "InstanceNormalization_TRT"]
        for node in nodes:
            relu_node = node.o()
            node.attrs["relu"] = 1
            node.attrs["alpha"] = 0.0
            node.outputs = relu_node.outputs
            relu_node.outputs.clear()

    def conv3d1x1x1k4_plugin(self):
        nodes = [node for node in self.graph.nodes if node.op == "InstanceNormalization_TRT"]
        last_layer_node = nodes[-1].o()
        last_layer_node.op = "CONV3D1X1X1K4_TRT"
        assert len(last_layer_node.inputs) == 3, "Weight and Bias needed"
        weights = last_layer_node.inputs[1]
        weights_shape = weights.values.shape
        weights_c = weights_shape[1]
        weights_k = weights_shape[0]
        assert weights_shape == (3, 32, 1, 1, 1), "Expecting c == 32 and k == 3"
        bias = last_layer_node.inputs[2]
        bias_shape = bias.values.shape
        assert bias_shape[0] == weights_k, "Unexpected Bias shape"
        last_layer_node.attrs["inputChannels"] = weights_c
        last_layer_node.attrs["weights"] = weights
        last_layer_node.attrs["bias"] = bias
        last_layer_node.attrs["plugin_version"] = "1"
        last_layer_node.attrs["plugin_namespace"] = ""
        del last_layer_node.inputs[2]
        del last_layer_node.inputs[1]

        # add the identity layer, since the last layer is quantized
        identity_out = gs.Variable("output", dtype=np.float32)
        identity = gs.Node(op="Identity", inputs=last_layer_node.outputs, outputs=[identity_out])
        self.graph.nodes.append(identity)
        self.graph.outputs.append(identity_out)
        last_layer_node.outputs[0].name = "conv3d1x1x1k4_out"

    def conv3d3x3x3c1k32_plugin(self):
        nodes = [node for node in self.graph.nodes if node.op == "Conv"]
        first_layer_node = nodes[0]
        first_layer_node.op = "CONV3D3X3X3C1K32_TRT"
        assert len(first_layer_node.inputs) == 2, "Weight needed"
        weights = first_layer_node.inputs[1]
        weights_shape = weights.values.shape
        weights_c = weights_shape[1]
        weights_k = weights_shape[0]
        assert weights_shape == (32, 1, 3, 3, 3), "Expecting c == 1 and k == 32"
        first_layer_node.attrs["inputChannels"] = weights_c
        first_layer_node.attrs["weights"] = weights
        first_layer_node.attrs["plugin_version"] = "1"
        first_layer_node.attrs["plugin_namespace"] = ""
        del first_layer_node.inputs[1]

    def conv_for_deconv(self,
                        pixel_shuffle_cdwh: bool = True,
                        enable_pixelshuffle3d_plugin: bool = True,
                        enable_pixelshuffle3d_plugin_concat_fuse: bool = True):
        added_nodes = []
        input_d = self.graph.inputs[0].shape[2]
        input_h = self.graph.inputs[0].shape[3]
        input_w = self.graph.inputs[0].shape[4]

        # We start the conversion from the lowest dimension
        current_d = input_d // 32
        current_h = input_h // 32
        current_w = input_w // 32

        for (node_idx, node) in enumerate(self.graph.nodes):
            if node.op == "ConvTranspose":
                name = node.name
                node.op = "Conv"
                assert node.attrs["kernel_shape"] == [2, 2, 2], "The conversion only makes sense for 2x2x2 deconv"
                node.attrs["kernel_shape"] = [1, 1, 1]
                assert node.attrs["strides"] == [2, 2, 2], "The conversion only makes sense for stride=2x2x2 deconv"
                node.attrs["strides"] = [1, 1, 1]

                # Transpose weights from cktrs to (ktrs)c111 or (trsk)c111
                assert len(node.inputs) >= 2, "Weights needed"
                weights = node.inputs[1]
                weights_shape = weights.values.shape
                weights_c = weights_shape[0]
                weights_k = weights_shape[1]
                assert weights_shape[2:] == (2, 2, 2), "The conversion only makes sense for 2x2x2 deconv"
                weights_transpose_axes = (1, 2, 3, 4, 0) if pixel_shuffle_cdwh else (2, 3, 4, 1, 0)
                weights.values = weights.values.transpose(weights_transpose_axes).reshape(weights_k * 8, weights_c, 1, 1, 1)

                # Check bias sanity
                assert len(node.inputs) == 3, "Bias needed"
                bias = node.inputs[2]
                bias_shape = bias.values.shape
                assert bias_shape[0] == weights_k, "Unexpected Bias shape"
                bias.values = bias.values.repeat(8).reshape([1, weights_k, 2, 2, 2]).transpose(weights_transpose_axes).reshape(weights_k * 8)

                deconv_output = node.outputs[0]
                concat_node = self.graph.nodes[node_idx + 1]
                assert concat_node.op == "Concat", "Cannot find the right Concat node"
                if enable_pixelshuffle3d_plugin:
                    # Insert PixelShuffle
                    pixel_shuffle_output = gs.Variable(name + "_pixelshuffle_plugin_out")
                    pixel_shuffle_node = gs.Node(
                        "PIXELSHUFFLE3D_TRT", name + "_pixelshuffle_plugin",
                        {}, [deconv_output], [pixel_shuffle_output])
                    pixel_shuffle_node.op = "PIXELSHUFFLE3D_TRT"
                    pixel_shuffle_node.attrs["R"] = 2
                    pixel_shuffle_node.attrs["S"] = 2
                    pixel_shuffle_node.attrs["T"] = 2
                    pixel_shuffle_node.attrs["plugin_version"] = "1"
                    pixel_shuffle_node.attrs["plugin_namespace"] = ""
                    assert concat_node.inputs[0] is deconv_output, "Wrong concat order"
                    if enable_pixelshuffle3d_plugin_concat_fuse:
                        pixel_shuffle_node.outputs = concat_node.outputs
                        pixel_shuffle_node.inputs.append(concat_node.inputs[1])
                        concat_node.outputs.clear()
                    else:
                        concat_node.inputs[0] = pixel_shuffle_output
                    added_nodes.extend([pixel_shuffle_node])
                else:
                    reshape1_shape = [0, weights_k, 2, 2, 2, current_d, current_h, current_w] if pixel_shuffle_cdwh else\
                                     [0, 2, 2, 2, weights_k, current_d, current_h, current_w]
                    shuffle_axes = [0, 1, 5, 2, 6, 3, 7, 4] if pixel_shuffle_cdwh else [0, 4, 5, 1, 6, 2, 7, 3]
                    current_d *= 2
                    current_h *= 2
                    current_w *= 2
                    reshape2_shape = [0, weights_k, current_d, current_h, current_w]
                    reshape1_shape_const = gs.Constant(name + "_pixelshuffle_reshape1_shape", np.array(reshape1_shape, dtype=np.int32))
                    reshape2_shape_const = gs.Constant(name + "_pixelshuffle_reshape2_shape", np.array(reshape2_shape, dtype=np.int32))
                    reshape1_output = gs.Variable(name + "_pixelshuffle_reshape1_out")
                    shuffle_output = gs.Variable(name + "_pixelshuffle_shuffle_out")
                    reshape2_output = gs.Variable(name + "_pixelshuffle_reshape2_out")
                    reshape1_node = gs.Node(
                        "Reshape", name + "_pixelshuffle_reshape1",
                        {}, [deconv_output, reshape1_shape_const], [reshape1_output])
                    shuffle_node = gs.Node(
                        "Transpose", name + "_pixelshuffle_transpose",
                        {"perm": shuffle_axes}, [reshape1_output], [shuffle_output])
                    reshape2_node = gs.Node(
                        "Reshape", name + "_pixelshuffle_reshape2",
                        {}, [shuffle_output, reshape2_shape_const], [reshape2_output])
                    assert concat_node.inputs[0] is deconv_output, "Wrong concat order"
                    concat_node.inputs[0] = reshape2_output
                    added_nodes.extend([reshape1_node, shuffle_node, reshape2_node])
        self.graph.nodes.extend(added_nodes)

    def cleanup_graph(self):
        # Remove the four unnecessary outputs.
        self.graph.outputs = [
            output
            for output in self.graph.outputs
            if output.name == "output"
        ]
        self.graph.cleanup().toposort()

        # Add names to the layer after the graph is toposorted.
        uniq_num = 0
        for node in self.graph.nodes:
            if not node.name or node.name.isdigit():
                op_name = str(node.op)
                node.name = f'gs_{op_name}_{uniq_num}'
                node.attrs['name'] = node.name
                uniq_num += 1
            for out_idx, out_tensor in enumerate(node.outputs):
                postfix = "_" + out_idx if len(node.outputs) > 1 else ""
                if not out_tensor.name or out_tensor.name.isdigit():
                    out_tensor.name = node.name + "__output" + postfix


class UnetEngineBuilderOp(CalibratableTensorRTEngine,
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
                 model_path: str = "build/models/3d-unet-kits19/3dUNetKiTS19.onnx",
                 export_graphsurgeoned_model: bool = False,
                 # Override the normal default value
                 workspace_size: int = 8 << 30,
                 calib_max_batches: int = 20,
                 calib_data_map: PathLike = Path("data_maps/kits19/cal_map.txt"),
                 cache_file: PathLike = Path("code/3d-unet/tensorrt/calibrator.cache"),
                 # Benchmark specific values
                 batch_size: int = 1,
                 force_calibration: bool = False,
                 **kwargs):
        super().__init__(workspace_size=workspace_size,
                         calib_max_batches=calib_max_batches,
                         calib_data_map=calib_data_map,
                         cache_file=cache_file,
                         **kwargs)

        self.config_ver = config_ver
        self.model_path = model_path
        self.export_graphsurgeoned_model = export_graphsurgeoned_model
        self.batch_size = batch_size

        assert not self.dla_enabled, "3D UNET does not support DLA execution"
        self.device_type = "gpu"

        # Migrated from the old builder implementation with the API. These configuration options were not configured via
        # the config dict and are therefore not exposed as arguments in the Mitten implementation.
        self.channel_idx = 1
        self.num_input_channel = 1

        self.input_volume_dim = [128, 128, 128]
        self.input_tensor_dim = [-1] + self.input_volume_dim
        self.input_tensor_dim.insert(self.channel_idx, self.num_input_channel)

    def set_calibrator(self, image_dir):
        if self.precision != Precision.INT8:
            return

        self.calibrator = UNet3DKiTS19MinMaxCalibrator(image_dir,
                                                       self.cache_file,
                                                       self.calib_batch_size,
                                                       self.calib_max_batches,
                                                       self.force_calibration,
                                                       self.calib_data_map)
        assert self.calibrator, "3D UNET Calibrator failed to init"
        assert self.calibrator.get_algorithm() == trt.CalibrationAlgoType.MINMAX_CALIBRATION

    def create_network(self, builder: trt.Builder, onnx_export_path: os.PathLike):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        compute_sm = DETECTED_SYSTEM.get_compute_sm()

        # Parse from ONNX file
        parser = trt.OnnxParser(network, self.logger)
        unet_gs = UnetGraphSurgeon(self.model_path, self.precision, self.cache_file)
        model = unet_gs.create_onnx_model()

        if self.export_graphsurgeoned_model:
            onnx.save(model, onnx_export_path)
        success = parser.parse(onnx._serialize(model))
        if not success:
            err_desc = parser.get_error(0).desc()
            raise RuntimeError(f"3D-UNET onnx model processing failed! Error: {err_desc}")

        self.apply_network_io_types(network)
        return network

    def apply_network_io_types(self, network: trt.INetworkDefinition):
        # Use Linear input if INT8 / FP32 always Linear
        assert self.input_dtype == "int8" and self.input_format == "linear",\
            "3D-UNet has to be built with INT8 LINEAR input"

        # Set input/output tensor dtype and formats
        input_tensor = network.get_input(0)
        input_tensor.shape = self.input_tensor_dim
        input_tensor.dtype = trt.int8
        input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        if not self.need_calibration:
            int8_calib_ranges = get_dyn_ranges(self.cache_file)
            if "input" not in int8_calib_ranges:
                raise RuntimeError("'input' not in calibration cache")
            input_dyn_range = int8_calib_ranges["input"]
            input_tensor.set_dynamic_range(-input_dyn_range, input_dyn_range)

        # Always use FP16 LINEAR output
        # workaround for calibration not working with the identity layer properly, with last layer plugin
        output_tensor = network.get_output(0)
        output_tensor.dtype = trt.float32 if self.force_calibration else trt.float16
        output_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)

    def run(self, scratch_space, dependency_outputs):
        # Set up INT8 calibration
        subdir = Path("KiTS19/calibration/fp32")
        self.set_calibrator(scratch_space.path / "preprocessed_data" / subdir)

        # Create builder config
        builder_config = self.create_builder_config()
        builder_config.int8_calibrator = self.calibrator
        # 3D-UNET is a pretty intensive workload. Explicitly enable FP16 fallback for INT8 precision
        if self.precision == Precision.INT8:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        network = self.create_network(self.builder,
                                      scratch_space.path / "post_gs_3dunet_kits19.onnx")
        engine_dir = self.engine_dir(scratch_space)
        engine_name = self.engine_name(self.device_type,
                                       self.batch_size,
                                       self.precision,
                                       tag=self.config_ver)
        engine_fpath = engine_dir / engine_name
        self.build_engine(network, builder_config, self.batch_size, engine_fpath)


class UnetBuilder(LegacyBuilder):
    """Temporary spoofing class to wrap around Mitten to adhere to the old API.
    """

    def __init__(self, args):
        super().__init__(UnetEngineBuilderOp(**args))
