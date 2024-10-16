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
from typing import Any, Dict, Optional

import numpy as np
import tensorrt as trt
import torch

from nvmitten.constants import Precision
from nvmitten.nvidia.builder import (TRTBuilder,
                                     CalibratableTensorRTEngine,
                                     MLPerfInferenceEngine,
                                     ONNXNetwork,
                                     LegacyBuilder)
from nvmitten.pipeline import Operation
from nvmitten.utils import dict_get, logging

from code.plugin import load_trt_plugin_by_network
load_trt_plugin_by_network("rnnt")

from code.common.fields import Fields
from code.common.mitten_compat import ArgDiscarder
from code.common.systems.system_list import SystemClassifications
from code.rnnt.dali.pipeline import DALIInferencePipeline

from importlib import import_module
RNNTCalibrator = import_module("code.rnnt.tensorrt.calibrator").RNNTCalibrator


def set_tensor_dtype(tensor, dtype, dformat):
    # dtype string to trt dtype
    t_dtype = {"int8": trt.int8,
               "int32": trt.int32,
               "fp16": trt.float16,
               "fp32": trt.float32}[dtype]

    tensor.dtype = t_dtype
    if t_dtype == trt.int8:
        tensor.dynamic_range = (-128, 127)

    t_dformat = {"linear": trt.TensorFormat.LINEAR,
                 "chw4": trt.TensorFormat.CHW4,
                 "hwc8": trt.TensorFormat.HWC8}[dformat]
    tensor.allowed_formats = 1 << int(t_dformat)


class RNNHyperParam:
    # alphabet
    labels_size = 29   # alphabet

    # encoder
    encoder_input_size = 240
    encoder_hidden_size = 1024
    enc_pre_rnn_layers = 2
    enc_post_rnn_layers = 3

    # encoder
    decoder_input_size = 320
    decoder_hidden_size = 320
    joint_hidden_size = 512
    dec_rnn_layers = 2


class BaseRNNTBuilder(TRTBuilder):

    state_dict = None
    """dict: PyTorch state dict. Cached as a class variable to prevent making unnecessary copies."""

    @classmethod
    def _load_state_dict(cls,
                         model_path: PathLike = "build/models/rnn-t/DistributedDataParallel_1576581068.9962234-epoch-100.pt",
                         force_reload: bool = False):
        if not cls.state_dict or force_reload:
            ckpt = torch.load(model_path, map_location="cpu")
            cls.state_dict = ckpt["state_dict"]
        return cls.state_dict

    def __init__(self,
                 *args,
                 batch_size: int = 1,
                 workspace_size: int = 4 << 30,
                 max_seq_length: int = 128,
                 num_profiles: int = 1,  # Unused. Forcibly overridden to 1.
                 opt: str = "greedy",
                 **kwargs):
        # Old RNNT builder forced num_profiles to 1, do the same here
        super().__init__(*args, num_profiles=1, workspace_size=workspace_size, **kwargs)

        self.batch_size = batch_size
        self.state_dict = BaseRNNTBuilder._load_state_dict()
        self.max_seq_length = max_seq_length
        self.opt = opt

    def create_builder_config(self, *args, **kwargs) -> trt.IBuilderConfig:
        builder_config = super().create_builder_config(*args, **kwargs)

        # https://nvbugs/4038825
        # Enable legacy backend for RNNT builder to bypass Myelin for encoder
        if not SystemClassifications.is_orin():
            logging.info("Enabling cuDNN/cuBLAS/cuBLASLt tactics for TRT 8.6 on datacenter systems")
            builder_config.set_preview_feature(trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, False)
        return builder_config


class RNNTEncoder(CalibratableTensorRTEngine,
                  BaseRNNTBuilder,
                  ArgDiscarder):
    """Represents the RNNT Encoder network.
    """

    def __init__(self,
                 *args,
                 # Legacy behavior: capture batch size here, and override with enc_batch_size if provided.
                 batch_size: int = 2048,
                 enc_batch_size: int = None,
                 expose_state: bool = True,
                 calib_batch_size: int = 10,
                 calib_max_batches: int = 500,
                 force_calibration: bool = False,
                 calib_data_map: PathLike = "build/preprocessed_data/rnnt_train_clean_512_fp32/val_map_512.txt",
                 cache_file: PathLike = Path("code/rnnt/tensorrt/calibrator.cache"),
                 disable_encoder_plugin: bool = False,
                 **kwargs):
        if disable_encoder_plugin:
            logging.info(f"Encoder plugin is disabled. Disabling calibrator.")
            force_calibration = False

        if enc_batch_size is not None:
            batch_size = enc_batch_size

        super().__init__(batch_size=batch_size,
                         calib_batch_size=calib_batch_size,
                         calib_max_batches=calib_max_batches,
                         calib_data_map=calib_data_map,
                         cache_file=cache_file,
                         **kwargs)
        self.expose_state = expose_state
        self.unroll = self.force_calibration
        self.disable_encoder_plugin = disable_encoder_plugin

        if self.need_calibration:
            if self.expose_state:
                raise RuntimeError("Cannot expose state during calibration.")

            if self.input_dtype != "fp32":
                logging.warning("Using non-fp32 input may result in accuracy degradation and bad calibration scales.")

            self.precision = Precision.INT8

            if calib_batch_size < self.batch_size:
                # MLPINF-437
                raise RuntimeError("Calibration batch size must be at least network batch size.")

    @property
    def need_calibration(self):
        # Force calibration off if encoder plugin is disabled.
        return (not self.disable_encoder_plugin) and (self.force_calibration or not self.cache_file.exists())

    def get_calibrator(self, data_dir):
        if self.precision != Precision.INT8:
            return None

        self.calibrator = RNNTCalibrator(self.calib_batch_size,
                                         self.calib_max_batches,
                                         self.need_calibration,
                                         self.cache_file,
                                         self.calib_data_map,
                                         data_dir,
                                         self.input_dtype)
        return self.calibrator

    def parse_calibration(self):
        """Parse calibration file to get dynamic range of all network tensors.
        Returns the tensor:range dict.
        """

        if not self.cache_file.exists():
            return None

        with self.cache_file.open(mode="rb") as f:
            lines = f.read().decode('ascii').splitlines()

        calibration_dict = dict()
        for line in lines:
            split = line.split(':')
            if len(split) != 2:
                continue
            tensor = split[0]
            calibration_dict[tensor] = np.uint32(int(split[1], 16)).view(np.dtype('float32')).item()
        return calibration_dict

    def _init_weights_per_layer(self, layer, idx, is_unrolled=False):
        name = layer.name
        # initialization of the gate weights
        weight_ih = self.state_dict[name + '.weight_ih_l' + str(idx)]
        weight_ih = weight_ih.chunk(4, 0)

        weight_hh = self.state_dict[name + '.weight_hh_l' + str(idx)]
        weight_hh = weight_hh.chunk(4, 0)

        bias_ih = self.state_dict[name + '.bias_ih_l' + str(idx)]
        bias_ih = bias_ih.chunk(4, 0)

        bias_hh = self.state_dict[name + '.bias_hh_l' + str(idx)]
        bias_hh = bias_hh.chunk(4, 0)

        for gate_type in [trt.RNNGateType.INPUT, trt.RNNGateType.CELL, trt.RNNGateType.FORGET, trt.RNNGateType.OUTPUT]:
            for is_w in [True, False]:
                if is_w:
                    if (gate_type == trt.RNNGateType.INPUT):
                        weights = trt.Weights(weight_ih[0].numpy().astype(np.float32))
                        bias = trt.Weights(bias_ih[0].numpy().astype(np.float32))
                    elif (gate_type == trt.RNNGateType.FORGET):
                        weights = trt.Weights(weight_ih[1].numpy().astype(np.float32))
                        bias = trt.Weights(bias_ih[1].numpy().astype(np.float32))
                    elif (gate_type == trt.RNNGateType.CELL):
                        weights = trt.Weights(weight_ih[2].numpy().astype(np.float32))
                        bias = trt.Weights(bias_ih[2].numpy().astype(np.float32))
                    elif (gate_type == trt.RNNGateType.OUTPUT):
                        weights = trt.Weights(weight_ih[3].numpy().astype(np.float32))
                        bias = trt.Weights(bias_ih[3].numpy().astype(np.float32))
                else:
                    if (gate_type == trt.RNNGateType.INPUT):
                        weights = trt.Weights(weight_hh[0].numpy().astype(np.float32))
                        bias = trt.Weights(bias_hh[0].numpy().astype(np.float32))
                    elif (gate_type == trt.RNNGateType.FORGET):
                        weights = trt.Weights(weight_hh[1].numpy().astype(np.float32))
                        bias = trt.Weights(bias_hh[1].numpy().astype(np.float32))
                    elif (gate_type == trt.RNNGateType.CELL):
                        weights = trt.Weights(weight_hh[2].numpy().astype(np.float32))
                        bias = trt.Weights(bias_hh[2].numpy().astype(np.float32))
                    elif (gate_type == trt.RNNGateType.OUTPUT):
                        weights = trt.Weights(weight_hh[3].numpy().astype(np.float32))
                        bias = trt.Weights(bias_hh[3].numpy().astype(np.float32))

                layer_idx = idx if not is_unrolled else 0
                layer.set_weights_for_gate(layer_idx, gate_type, is_w, weights)
                layer.set_bias_for_gate(layer_idx, gate_type, is_w, bias)

    def add_unrolled_rnns(self,
                          network,
                          num_layers,
                          max_seq_length,
                          input_tensor,
                          length_tensor,
                          length_tensor_host,
                          input_size,
                          hidden_size,
                          hidden_state_tensor,
                          cell_state_tensor,
                          name):
        past_layer = None
        for i in range(num_layers):
            if past_layer is None:
                # For the first layer, set-up inputs
                rnn_layer = network.add_rnn_v2(input_tensor, 1, hidden_size, max_seq_length, trt.RNNOperation.LSTM)
                rnn_layer.seq_lengths = length_tensor
                # Note that we don't hook-up argument-state-tensors because
                # calib_unroll can only be called with --seq_splitting_off
            else:
                # Hook-up the past layer
                rnn_layer = network.add_rnn_v2(past_layer.get_output(0), 1, hidden_size, max_seq_length, trt.RNNOperation.LSTM)
                rnn_layer.seq_lengths = length_tensor
            rnn_layer.get_output(0).name = f"{name}{i}_output"
            rnn_layer.get_output(1).name = f"{name}{i}_hidden"
            rnn_layer.get_output(2).name = f"{name}{i}_cell"
            # Set the name as expected for weight finding
            rnn_layer.name = name
            self._init_weights_per_layer(rnn_layer, i, True)
            # Now rename the layer for readability
            rnn_layer.name = f"{name}{i}"
            # Move on to the next layer
            past_layer = rnn_layer
        return rnn_layer

    def add_rolled_rnns(self,
                        network,
                        num_layers,
                        max_seq_length,
                        input_tensor,
                        length_tensor,
                        length_tensor_host,
                        input_size,
                        hidden_size,
                        hidden_state_tensor,
                        cell_state_tensor,
                        name):

        if self.disable_encoder_plugin:
            rnn_layer = network.add_rnn_v2(input_tensor, num_layers, hidden_size, max_seq_length, trt.RNNOperation.LSTM)
            rnn_layer.seq_lengths = length_tensor
            rnn_layer.name = name
            # connect the initial hidden/cell state tensors (if they exist)
            if hidden_state_tensor:
                rnn_layer.hidden_state = hidden_state_tensor
            if cell_state_tensor:
                rnn_layer.cell_state = cell_state_tensor

            for i in range(rnn_layer.num_layers):
                self._init_weights_per_layer(rnn_layer, idx=i)

            return rnn_layer
        else:
            layer = None
            plugin = None
            plugin_name = "RNNTEncoderPlugin"
            calibration_dict = self.parse_calibration()

            for plugin_creator in trt.get_plugin_registry().plugin_creator_list:
                if plugin_creator.name == plugin_name:
                    logging.info("RNNTEncoderPlugin Plugin found")

                    fields = []

                    fields.append(trt.PluginField("numLayers", np.array([num_layers], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("hiddenSize", np.array([hidden_size], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("inputSize", np.array([input_size], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("max_seq_length", np.array([max_seq_length], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("max_batch_size", np.array([self.batch_size], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("dataType", np.array([trt.DataType.INT8], dtype=np.int32), trt.PluginFieldType.INT32))

                    for layer in range(num_layers):
                        weightsI = self.state_dict[name + '.weight_ih_l' + str(layer)]
                        weightsH = self.state_dict[name + '.weight_hh_l' + str(layer)]

                        if layer == 0:
                            assert(weightsI.numpy().astype(np.float16).size == 4 * hidden_size * input_size)
                        else:
                            assert(weightsI.numpy().astype(np.float16).size == 4 * hidden_size * hidden_size)
                        assert(weightsH.numpy().astype(np.float16).size == 4 * hidden_size * hidden_size)

                        fields.append(trt.PluginField("weightsI", weightsI.numpy().astype(np.float16), trt.PluginFieldType.FLOAT16))
                        fields.append(trt.PluginField("weightsH", weightsH.numpy().astype(np.float16), trt.PluginFieldType.FLOAT16))

                    for layer in range(num_layers):
                        biases = torch.cat((self.state_dict[name + '.bias_ih_l' + str(layer)], self.state_dict[name + '.bias_hh_l' + str(layer)]), 0)

                        fields.append(trt.PluginField("bias", biases.numpy().astype(np.float16), trt.PluginFieldType.FLOAT16))

                    scaleFactors = []

                    if name == "encoder.pre_rnn.lstm":
                        scaleFactors.append(1 / calibration_dict["input"])
                    elif name == "encoder.post_rnn.lstm":
                        scaleFactors.append(1 / calibration_dict["encoder_reshape"])

                    else:
                        logging.error("Unrecognised name in add_rnns")

                    fields.append(trt.PluginField("scaleFactors", np.array(scaleFactors, dtype=np.float32), trt.PluginFieldType.FLOAT32))

                    field_collection = trt.PluginFieldCollection(fields)

                    plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)

                    inputs = []
                    inputs.append(input_tensor)
                    inputs.append(hidden_state_tensor)
                    inputs.append(cell_state_tensor)
                    inputs.append(length_tensor)
                    assert(length_tensor_host)  # None if disable_encoder_plugin=True
                    inputs.append(length_tensor_host)

                    layer = network.add_plugin_v2(inputs, plugin)
                    layer.name = name

                    break

            if not plugin:
                logging.error("RNNTEncoderPlugin not found")
            if not layer:
                logging.error(f"Layer {name} not set")

            return layer

    @property
    def add_rnns(self):
        if self.unroll:
            return self.add_unrolled_rnns
        else:
            return self.add_rolled_rnns

    def create_network(self, builder: trt.Builder):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        input_tensor = network.add_input("input",
                                         trt.DataType.FLOAT,
                                         (-1, self.max_seq_length, RNNHyperParam.encoder_input_size))
        set_tensor_dtype(input_tensor, self.input_dtype, self.input_format)

        length_tensor = network.add_input("length", trt.DataType.INT32, (-1,))
        length_tensor_host = None
        if not self.disable_encoder_plugin:
            length_tensor_host = network.add_input("length_host", trt.DataType.INT32, (-1,))

        # compute (seq_length + 1) // 2
        one_constant = network.add_constant((1,), np.array([1]).astype(np.int32))
        one_constant.get_output(0).name = "one_constant"
        length_add_one = network.add_elementwise(length_tensor, one_constant.get_output(0), trt.ElementWiseOperation.SUM)
        length_add_one.get_output(0).name = "length_add_one"
        two_constant = network.add_constant((1,), np.array([2]).astype(np.int32))
        two_constant.get_output(0).name = "two_constant"
        length_half = network.add_elementwise(length_add_one.get_output(0), two_constant.get_output(0), trt.ElementWiseOperation.FLOOR_DIV)
        length_half.get_output(0).name = "length_half"

        # state handling
        enc_tensor_dict = {'lower': dict(), 'upper': dict()}
        for tensor_name in ['hidden', 'cell']:
            if self.expose_state:
                enc_tensor_dict['lower'][tensor_name] = network.add_input("lower_" + tensor_name, trt.DataType.FLOAT, (-1, RNNHyperParam.enc_pre_rnn_layers, RNNHyperParam.encoder_hidden_size))
                enc_tensor_dict['upper'][tensor_name] = network.add_input("upper_" + tensor_name, trt.DataType.FLOAT, (-1, RNNHyperParam.enc_post_rnn_layers, RNNHyperParam.encoder_hidden_size))
                set_tensor_dtype(enc_tensor_dict['lower'][tensor_name], self.input_dtype, self.input_format)
                set_tensor_dtype(enc_tensor_dict['upper'][tensor_name], self.input_dtype, self.input_format)
            else:
                enc_tensor_dict['lower'][tensor_name] = None
                enc_tensor_dict['upper'][tensor_name] = None

        # pre_rnn
        encoder_lower = self.add_rnns(network,
                                      RNNHyperParam.enc_pre_rnn_layers,
                                      self.max_seq_length,
                                      input_tensor,
                                      length_tensor,
                                      length_tensor_host,
                                      RNNHyperParam.encoder_input_size,
                                      RNNHyperParam.encoder_hidden_size,
                                      enc_tensor_dict['lower']['hidden'],
                                      enc_tensor_dict['lower']['cell'],
                                      'encoder.pre_rnn.lstm')
        # reshape (stack time x 2)
        reshape_layer = network.add_shuffle(encoder_lower.get_output(0))
        reshape_layer.reshape_dims = trt.Dims((0, self.max_seq_length // 2, RNNHyperParam.encoder_hidden_size * 2))
        reshape_layer.name = 'encoder_reshape'
        reshape_layer.get_output(0).name = 'encoder_reshape'

        # post_rnn
        encoder_upper = self.add_rnns(network,
                                      RNNHyperParam.enc_post_rnn_layers,
                                      self.max_seq_length // 2,
                                      reshape_layer.get_output(0),
                                      length_half.get_output(0),
                                      length_tensor_host,
                                      RNNHyperParam.encoder_hidden_size * 2,
                                      RNNHyperParam.encoder_hidden_size,
                                      enc_tensor_dict['upper']['hidden'],
                                      enc_tensor_dict['upper']['cell'],
                                      'encoder.post_rnn.lstm')

        # Add expected names for "regular" LSTM layers.
        if not self.unroll:
            encoder_lower.name = 'encoder_pre_rnn'
            encoder_lower.get_output(0).name = "encoder_pre_rnn_output"
            encoder_lower.get_output(1).name = "encoder_pre_rnn_hidden"
            encoder_lower.get_output(2).name = "encoder_pre_rnn_cell"

            encoder_upper.name = 'encoder_post_rnn'
            encoder_upper.get_output(0).name = "encoder_post_rnn_output"
            encoder_upper.get_output(1).name = "encoder_post_rnn_hidden"
            encoder_upper.get_output(2).name = "encoder_post_rnn_cell"

        # mark outputs
        network.mark_output(encoder_upper.get_output(0))
        set_tensor_dtype(encoder_upper.get_output(0), self.input_dtype, self.input_format)
        if self.expose_state:
            # lower_hidden
            network.mark_output(encoder_lower.get_output(1))
            set_tensor_dtype(encoder_lower.get_output(1), self.input_dtype, self.input_format)
            # upper_hidden
            network.mark_output(encoder_upper.get_output(1))
            set_tensor_dtype(encoder_upper.get_output(1), self.input_dtype, self.input_format)
            # lower_cell
            network.mark_output(encoder_lower.get_output(2))
            set_tensor_dtype(encoder_lower.get_output(2), self.input_dtype, self.input_format)
            # upper_cell
            network.mark_output(encoder_upper.get_output(2))
            set_tensor_dtype(encoder_upper.get_output(2), self.input_dtype, self.input_format)

        return network


class RNNTDecoder(BaseRNNTBuilder,
                  ArgDiscarder):
    """Represents the RNNT Decoder network.
    """

    def __init__(self,
                 *args,
                 disable_decoder_plugin: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.disable_decoder_plugin = disable_decoder_plugin

    def add_decoder_rnns(self,
                         network,
                         num_layers,
                         input_tensor,
                         hidden_size,
                         hidden_state_tensor,
                         cell_state_tensor,
                         name):
        max_seq_length = 1   # processed single step
        if self.disable_decoder_plugin:
            rnn_layer = network.add_rnn_v2(input_tensor, num_layers, hidden_size, max_seq_length, trt.RNNOperation.LSTM)

            # connect the initial hidden/cell state tensors
            rnn_layer.hidden_state = hidden_state_tensor
            rnn_layer.cell_state = cell_state_tensor

            # initialization of the gate weights
            for i in range(num_layers):
                weight_ih = self.state_dict[name + '.weight_ih_l' + str(i)]
                weight_ih = weight_ih.chunk(4, 0)

                weight_hh = self.state_dict[name + '.weight_hh_l' + str(i)]
                weight_hh = weight_hh.chunk(4, 0)

                bias_ih = self.state_dict[name + '.bias_ih_l' + str(i)]
                bias_ih = bias_ih.chunk(4, 0)

                bias_hh = self.state_dict[name + '.bias_hh_l' + str(i)]
                bias_hh = bias_hh.chunk(4, 0)

                for gate_type in [trt.RNNGateType.INPUT, trt.RNNGateType.CELL, trt.RNNGateType.FORGET, trt.RNNGateType.OUTPUT]:
                    for is_w in [True, False]:
                        if is_w:
                            if (gate_type == trt.RNNGateType.INPUT):
                                weights = trt.Weights(weight_ih[0].numpy().astype(np.float32))
                                bias = trt.Weights(bias_ih[0].numpy().astype(np.float32))
                            elif (gate_type == trt.RNNGateType.FORGET):
                                weights = trt.Weights(weight_ih[1].numpy().astype(np.float32))
                                bias = trt.Weights(bias_ih[1].numpy().astype(np.float32))
                            elif (gate_type == trt.RNNGateType.CELL):
                                weights = trt.Weights(weight_ih[2].numpy().astype(np.float32))
                                bias = trt.Weights(bias_ih[2].numpy().astype(np.float32))
                            elif (gate_type == trt.RNNGateType.OUTPUT):
                                weights = trt.Weights(weight_ih[3].numpy().astype(np.float32))
                                bias = trt.Weights(bias_ih[3].numpy().astype(np.float32))
                        else:
                            if (gate_type == trt.RNNGateType.INPUT):
                                weights = trt.Weights(weight_hh[0].numpy().astype(np.float32))
                                bias = trt.Weights(bias_hh[0].numpy().astype(np.float32))
                            elif (gate_type == trt.RNNGateType.FORGET):
                                weights = trt.Weights(weight_hh[1].numpy().astype(np.float32))
                                bias = trt.Weights(bias_hh[1].numpy().astype(np.float32))
                            elif (gate_type == trt.RNNGateType.CELL):
                                weights = trt.Weights(weight_hh[2].numpy().astype(np.float32))
                                bias = trt.Weights(bias_hh[2].numpy().astype(np.float32))
                            elif (gate_type == trt.RNNGateType.OUTPUT):
                                weights = trt.Weights(weight_hh[3].numpy().astype(np.float32))
                                bias = trt.Weights(bias_hh[3].numpy().astype(np.float32))

                        rnn_layer.set_weights_for_gate(i, gate_type, is_w, weights)
                        rnn_layer.set_bias_for_gate(i, gate_type, is_w, bias)

            return rnn_layer
        else:
            layer = None
            plugin = None
            plugin_name = "RNNTDecoderPlugin"

            # logging.info(trt.get_plugin_registry().plugin_creator_list)

            for plugin_creator in trt.get_plugin_registry().plugin_creator_list:
                if plugin_creator.name == plugin_name:
                    logging.info("Decoder Plugin found")

                    fields = []

                    fields.append(trt.PluginField("numLayers", np.array([num_layers], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("hiddenSize", np.array([hidden_size], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("inputSize", np.array([hidden_size], dtype=np.int32), trt.PluginFieldType.INT32))
                    fields.append(trt.PluginField("dataType", np.array([trt.DataType.HALF], dtype=np.int32), trt.PluginFieldType.INT32))

                    for layer in range(num_layers):
                        weights = torch.cat((self.state_dict[name + '.weight_ih_l' + str(layer)], self.state_dict[name + '.weight_hh_l' + str(layer)]), 0)

                        assert(weights.numpy().astype(np.float16).size == 8 * hidden_size * hidden_size)

                        fields.append(trt.PluginField("weights", weights.numpy().astype(np.float16), trt.PluginFieldType.FLOAT16))

                    for layer in range(num_layers):
                        biases = torch.cat((self.state_dict[name + '.bias_ih_l' + str(layer)], self.state_dict[name + '.bias_hh_l' + str(layer)]), 0)

                        fields.append(trt.PluginField("bias", biases.numpy().astype(np.float16), trt.PluginFieldType.FLOAT16))

                    field_collection = trt.PluginFieldCollection(fields)

                    plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)

                    inputs = []
                    inputs.append(input_tensor)
                    inputs.append(hidden_state_tensor)
                    inputs.append(cell_state_tensor)

                    layer = network.add_plugin_v2(inputs, plugin)

                    break

            if not plugin:
                logging.error("Plugin not found")

            return layer

    def create_network(self, builder: trt.Builder):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Decoder
        #    Embedding layer : (29 => 320)
        #    Rnn             : LSTM layers=2, h=320

        # Embedding layer
        #   gather layer with LUT of RNNHyperParam.labels_size=29 entries with RNNHyperParam.decoder_input_size=320 size per entry
        #   blank token does not need to be looked up, whereas the SOS (start-of-sequence) requires all zeros for embed vector
        dec_embedding_input = network.add_input("dec_embedding_input", trt.DataType.INT32, (-1, 1))
        dec_embedding_orig = self.state_dict["prediction.embed.weight"].numpy().astype(np.float32)
        dec_embedding_sos = np.zeros((1, RNNHyperParam.decoder_input_size), dtype=np.float32)
        dec_embedding_weights = trt.Weights(np.concatenate((dec_embedding_orig, dec_embedding_sos), axis=0))
        dec_embedding_lut = network.add_constant((RNNHyperParam.labels_size, RNNHyperParam.decoder_input_size), dec_embedding_weights)
        self.dec_embedding = network.add_gather(dec_embedding_lut.get_output(0), dec_embedding_input, axis=0)
        self.dec_embedding.name = 'decoder_embedding'

        # Rnn layer
        dec_rnn_layers = RNNHyperParam.dec_rnn_layers

        # Create tensors  [ batch, seq=1, input ]
        dec_tensor_dict = dict()
        # dec_tensor_dict['input']  = network.add_input("dec_input", trt.DataType.FLOAT, (-1, 1, RNNHyperParam.decoder_input_size))
        dec_tensor_dict['input'] = self.dec_embedding.get_output(0)
        dec_tensor_dict['hidden'] = network.add_input("hidden", trt.DataType.FLOAT, (-1, dec_rnn_layers, RNNHyperParam.decoder_hidden_size))
        dec_tensor_dict['cell'] = network.add_input("cell", trt.DataType.FLOAT, (-1, dec_rnn_layers, RNNHyperParam.decoder_hidden_size))
        for dec_tensor_name, dec_tensor_val in dec_tensor_dict.items():
            # RNN input is an internal layer whose type we should let TRT determine
            if dec_tensor_name != 'input':
                set_tensor_dtype(dec_tensor_val, self.input_dtype, self.input_format)

        # Instantiate RNN
        # logging.info("dec_input_size = {:}".format(dec_input_size))
        logging.info("dec_embed_lut OUT tensor shape = {:}".format(dec_embedding_lut.get_output(0).shape))
        logging.info("dec_embedding OUT tensor shape = {:}".format(self.dec_embedding.get_output(0).shape))
        decoder = self.add_decoder_rnns(network,
                                        dec_rnn_layers,
                                        dec_tensor_dict['input'],
                                        RNNHyperParam.decoder_hidden_size,
                                        dec_tensor_dict['hidden'],
                                        dec_tensor_dict['cell'],
                                        'prediction.dec_rnn.lstm')
        decoder.name = 'decoder_rnn'

        # Determine outputs (and override size)
        #   output
        #   hidden
        #   cell
        for output_idx in range(3):
            output_tensor = decoder.get_output(output_idx)
            network.mark_output(output_tensor)
            set_tensor_dtype(output_tensor, self.input_dtype, self.input_format)

        return network


class RNNTJointFC1(BaseRNNTBuilder,
                   ArgDiscarder):
    """Represents the FC1 layer of the RNNT network. FC1_a corresponds to the FC1 component attached to the Encoder, and
    FC1_b corresponds to the FC1 component attached to the Decoder.
    """

    def __init__(self,
                 *args,
                 mode: str = "encoder",
                 **kwargs):
        super().__init__(*args, **kwargs)

        if mode == "encoder":
            self.fc_in = "enc_input"
            self.fc_layer_name = "joint_fc1_a"
            self.hidden_size = RNNHyperParam.encoder_hidden_size
            self.weight_offset = 0
            self.add_bias = True
        elif mode == "decoder":
            self.fc_in = "dec_input"
            self.fc_layer_name = "joint_fc1_b"
            self.hidden_size = RNNHyperParam.decoder_hidden_size
            self.weight_offset = RNNHyperParam.encoder_hidden_size
            self.add_bias = False
        else:
            raise RuntimeError(f"`mode` must be 'encoder' or 'decoder'. Got '{mode}'.")

    def create_split_fc1_layer(self,
                               layer_name,
                               network,
                               input_tensor,
                               input_size,
                               output_size,
                               weight_offset,
                               joint_fc1_weight_ckpt,
                               joint_fc1_bias_ckpt,
                               add_bias=False):

        # detach weight (using weight_offset)
        joint_fc1_kernel_np = np.zeros((output_size, input_size))
        for i in range(output_size):
            for j in range(input_size):
                joint_fc1_kernel_np[i][j] = joint_fc1_weight_ckpt.numpy()[i][j + weight_offset]
        joint_fc1_kernel = joint_fc1_kernel_np.astype(np.float32)

        # detach bias (if available)
        joint_fc1_bias_np = np.zeros((output_size))
        if add_bias:
            for i in range(output_size):
                joint_fc1_bias_np[i] = joint_fc1_bias_ckpt.numpy()[i]
            joint_fc1_bias = joint_fc1_bias_np.astype(np.float32)

        # instantiate FC layer
        if add_bias:
            joint_fc1 = network.add_fully_connected(
                input_tensor,
                output_size,
                joint_fc1_kernel,
                joint_fc1_bias)
        else:
            joint_fc1 = network.add_fully_connected(
                input_tensor,
                output_size,
                joint_fc1_kernel)

        # epilogue
        joint_fc1.name = layer_name
        return joint_fc1

    def create_network(self, builder: trt.Builder):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Create tensors [ batch, seq=1, input ]
        input_tensor = network.add_input(self.fc_in, trt.DataType.FLOAT, (-1, self.hidden_size, 1, 1))
        set_tensor_dtype(input_tensor, self.input_dtype, "hwc8")  # hwc8 to avoid reformatting

        # FC1 + bias :
        joint_fc1_output_size = RNNHyperParam.joint_hidden_size
        joint_fc1_weight_ckpt = self.state_dict['joint_net.0.weight']
        joint_fc1_bias_ckpt = self.state_dict['joint_net.0.bias']

        # Instantiate split FC1 : one for the encoder and one for the decoder
        joint_fc1 = self.create_split_fc1_layer(self.fc_layer_name,
                                                network,
                                                input_tensor,
                                                self.hidden_size,
                                                joint_fc1_output_size,
                                                self.weight_offset,
                                                joint_fc1_weight_ckpt,
                                                joint_fc1_bias_ckpt,
                                                add_bias=self.add_bias)
        final_output = joint_fc1.get_output(0)

        # Set output properties
        network.mark_output(final_output)
        set_tensor_dtype(final_output, self.input_dtype, "hwc8")  # hwc8 to avoid reformatting

        return network


class RNNTJointBackend(BaseRNNTBuilder,
                       ArgDiscarder):
    """Represents the backend of the joint network after fc1_a and fc1_b.
    """

    def __init__(self,
                 engine_dir,
                 *args,
                 dump_joint_fc2_weights: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.engine_dir = Path(engine_dir)
        self.dump_joint_fc2_weights = dump_joint_fc2_weights

    def create_network(self, builder: trt.Builder):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Create tensors  [ batch, seq=1, input ]
        joint_fc1_output_size = RNNHyperParam.joint_hidden_size
        fc1_a_output = network.add_input("joint_fc1_a_output", trt.DataType.FLOAT, (-1, 1, joint_fc1_output_size))
        set_tensor_dtype(fc1_a_output, self.input_dtype, self.input_format)

        fc1_b_output = network.add_input("joint_fc1_b_output", trt.DataType.FLOAT, (-1, 1, joint_fc1_output_size))
        set_tensor_dtype(fc1_b_output, self.input_dtype, self.input_format)

        # element_wise SUM
        joint_fc1_sum = network.add_elementwise(fc1_a_output, fc1_b_output, trt.ElementWiseOperation.SUM)
        joint_fc1_sum.name = 'joint_fc1_sum'

        # reLU
        joint_relu = network.add_activation(joint_fc1_sum.get_output(0), trt.ActivationType.RELU)
        joint_relu.name = 'joint_relu'

        # FC2 + bias :
        joint_fc2_weight_ckpt = self.state_dict['joint_net.3.weight']
        joint_fc2_kernel = trt.Weights(joint_fc2_weight_ckpt.numpy().astype(np.float32))

        joint_fc2_bias_ckpt = self.state_dict['joint_net.3.bias']
        joint_fc2_bias = trt.Weights(joint_fc2_bias_ckpt.numpy().astype(np.float32))

        joint_fc2_shuffle = network.add_shuffle(joint_relu.get_output(0))   # Add an extra dimension for FC processing
        joint_fc2_shuffle.reshape_dims = (-1, joint_fc1_output_size, 1, 1)
        joint_fc2_shuffle.name = 'joint_fc2_shuffle'

        joint_fc2 = network.add_fully_connected(joint_fc2_shuffle.get_output(0),
                                                RNNHyperParam.labels_size,
                                                joint_fc2_kernel,
                                                joint_fc2_bias)
        joint_fc2.name = 'joint_fc2'

        # opt = GREEDY
        # -------------
        #    - Do not use softmax layer
        #    - Use TopK (K=1) GPU sorting

        # TopK (k=1)
        joint_top1 = network.add_topk(joint_fc2.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)
        joint_top1.name = 'joint_top1'

        # Final_output = joint_fc2.get_output(0)
        final_output = joint_top1.get_output(1)
        network.mark_output(final_output)

        # epilogue: dump fc2 weights and bias if required
        if self.dump_joint_fc2_weights:
            joint_fc2_weight_ckpt.numpy().astype(np.float16).tofile(self.engine_dir / "joint_fc2_weight_ckpt.fp16.dat")
            joint_fc2_bias_ckpt.numpy().astype(np.float16).tofile(self.engine_dir / "joint_fc2_bias_ckpt.fp16.dat")
            joint_fc2_weight_ckpt.numpy().astype(np.float32).tofile(self.engine_dir / "joint_fc2_weight_ckpt.fp32.dat")
            joint_fc2_bias_ckpt.numpy().astype(np.float32).tofile(self.engine_dir / "joint_fc2_bias_ckpt.fp32.dat")

        return network


class RNNTIsel(BaseRNNTBuilder,
               ArgDiscarder):
    """Represents the select operation for RNNT.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def create_network(self, builder: trt.Builder):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Isel:
        #    output_hidden: [ BS, layers=2, decoder_hidden_size=320 ]
        #    output_cell  : [ BS, layers=2, decoder_hidden_size=320 ]
        #
        #    input_select : [ BS, 1, 1 ]
        #    input0_hidden: [ BS, layers=2, decoder_hidden_size=320 ]
        #    input0_cell  : [ BS, layers=2, decoder_hidden_size=320 ]
        #    input1_hidden: [ BS, layers=2, decoder_hidden_size=320 ]
        #    input1_cell  : [ BS, layers=2, decoder_hidden_size=320 ]

        # Declare input tensors: port 0
        input0_hidden = network.add_input("input0_hidden", trt.DataType.FLOAT, (-1, RNNHyperParam.dec_rnn_layers, RNNHyperParam.decoder_hidden_size))
        input0_cell = network.add_input("input0_cell", trt.DataType.FLOAT, (-1, RNNHyperParam.dec_rnn_layers, RNNHyperParam.decoder_hidden_size))
        if self.opt == "greedy":
            input0_winner = network.add_input("input0_winner", trt.DataType.INT32, (-1, 1, 1))

        # Declare input tensors: port 1
        input1_hidden = network.add_input("input1_hidden", trt.DataType.FLOAT, (-1, RNNHyperParam.dec_rnn_layers, RNNHyperParam.decoder_hidden_size))
        input1_cell = network.add_input("input1_cell", trt.DataType.FLOAT, (-1, RNNHyperParam.dec_rnn_layers, RNNHyperParam.decoder_hidden_size))
        if self.opt == "greedy":
            input1_winner = network.add_input("input1_winner", trt.DataType.INT32, (-1, 1, 1))

        # Reformat tensors
        for input_tensor in (input0_hidden, input0_cell, input1_hidden, input1_cell):
            set_tensor_dtype(input_tensor, self.input_dtype, self.input_format)

        # One Iselect layer per component
        if self.input_dtype != "fp16" or self.opt != "greedy":
            # Select tensor
            input_select = network.add_input("input_select", trt.DataType.BOOL, (-1, 1, 1))

            isel_hidden = network.add_select(input_select, input0_hidden, input1_hidden)
            isel_cell = network.add_select(input_select, input0_cell, input1_cell)
            isel_hidden.name = 'Iselect Dec hidden'
            isel_cell.name = 'Iselect Dec cell'
            if self.opt == "greedy":
                isel_winner = network.add_select(input_select, input0_winner, input1_winner)
                isel_winner.name = 'Iselect Dec winner'

            # Declare outputs
            output_hidden = isel_hidden.get_output(0)
            output_cell = isel_cell.get_output(0)
            network.mark_output(output_hidden)
            network.mark_output(output_cell)
            set_tensor_dtype(output_hidden, self.input_dtype, self.input_format)
            set_tensor_dtype(output_cell, self.input_dtype, self.input_format)

            if self.opt == "greedy":
                output_winner = isel_winner.get_output(0)
                network.mark_output(output_winner)
        else:
            sel3Layer = None
            plugin = None
            plugin_name = "RNNTSelectPlugin"

            # Select tensor
            input_select = network.add_input("input_select", trt.DataType.INT32, (-1, 1, 1))

            for plugin_creator in trt.get_plugin_registry().plugin_creator_list:
                if plugin_creator.name == plugin_name:
                    logging.info("Select Plugin found")

                    fields = []

                    field_collection = trt.PluginFieldCollection(fields)

                    plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)

                    inputs = []
                    inputs.append(input_select)
                    inputs.append(input0_hidden)
                    inputs.append(input1_hidden)
                    inputs.append(input0_cell)
                    inputs.append(input1_cell)
                    inputs.append(input0_winner)
                    inputs.append(input1_winner)

                    sel3Layer = network.add_plugin_v2(inputs, plugin)

                    sel3Layer.name = 'Select3'

                    break

            if not plugin:
                logging.error("Select plugin not found")

            # Declare outputs
            output_hidden = sel3Layer.get_output(0)
            output_cell = sel3Layer.get_output(1)
            network.mark_output(output_hidden)
            network.mark_output(output_cell)
            set_tensor_dtype(output_hidden, self.input_dtype, self.input_format)
            set_tensor_dtype(output_cell, self.input_dtype, self.input_format)

            output_winner = sel3Layer.get_output(2)
            network.mark_output(output_winner)
            set_tensor_dtype(output_winner, "int32", self.input_format)
        return network


class RNNTIgather(BaseRNNTBuilder,
                  ArgDiscarder):
    """Represents the select operation for RNNT.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def create_network(self, builder: trt.Builder):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Igather:
        #    encoder_input  : [ BS, SEQ//2=1152//2, encoder_hidden_size=1024 ]  native
        #    t_coordinate   : [ BS, 1, ]                                        int32
        #
        #    igather_output : [ BS, 1,  encoder_hidden_size=1024 ]              native

        # Declare input tensors
        encoder_input = network.add_input("encoder_input", trt.DataType.FLOAT, (-1, self.max_seq_length // 2, RNNHyperParam.encoder_hidden_size))
        t_coordinate = network.add_input("t_coordinate", trt.DataType.INT32, trt.Dims([-1]))
        set_tensor_dtype(encoder_input, self.input_dtype, self.input_format)

        igather_layer = network.add_gather(encoder_input, t_coordinate, axis=1)
        igather_layer.name = "Igather joint cell"
        igather_layer.num_elementwise_dims = 1

        # Declare outputs
        igather_output = igather_layer.get_output(0)
        network.mark_output(igather_output)
        set_tensor_dtype(igather_output, self.input_dtype, self.input_format)

        return network


class RNNTDaliPipelineOp(Operation):
    @classmethod
    def immediate_dependencies(cls):
        return None

    def __init__(self, audio_fp16_input: bool = True):
        self.audio_fp16_input = audio_fp16_input
        _audio_input_precision_str = "fp16" if audio_fp16_input else "fp32"
        self.dali_fpath = f"build/bin/dali/dali_pipeline_gpu_{_audio_input_precision_str}.pth"

    def run(self, scratch_space, dependency_outputs):
        dali_dir = scratch_space.working_dir(namespace="bin/dali")

        dali_pipeline = DALIInferencePipeline.from_config(
            device="gpu",
            config=dict(),    # Default case
            device_id=0,
            batch_size=16,
            total_samples=16,  # Unused, can be set arbitrarily
            num_threads=2,
            audio_fp16_input=self.audio_fp16_input
        )
        dali_pipeline.serialize(filename=self.dali_fpath)


class RNNTEngineBuilderOp(MLPerfInferenceEngine, Operation, ArgDiscarder):
    @classmethod
    def immediate_dependencies(cls):
        return {RNNTDaliPipelineOp}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Save args to forward to builder
        self.args = args
        self.kwargs = kwargs

    def run(self, scratch_space, dependency_outputs):
        engine_dir = self.engine_dir(scratch_space)

        # Build each engine separately
        # For now, just forward args and kwargs as-is. This means it is relying on ArgDiscarder to work correctly.

        logging.info("Building encoder...")
        encoder = RNNTEncoder(*self.args, **self.kwargs)
        builder_config = encoder.create_builder_config()
        builder_config.int8_calibrator = encoder.get_calibrator(scratch_space.path / "preprocessed_data" / "rnnt_train_clean_512_fp32" / "fp32")
        encoder.build_engine(encoder.create_network(encoder.builder),
                             builder_config,
                             encoder.batch_size,
                             engine_dir / "encoder.plan")

        logging.info("Building decoder...")
        decoder = RNNTDecoder(*self.args, **self.kwargs)
        decoder.build_engine(decoder.create_network(decoder.builder),
                             decoder.create_builder_config(),
                             decoder.batch_size,
                             engine_dir / "decoder.plan")

        logging.info("Building fc1_a...")
        fc1_a = RNNTJointFC1(*self.args, mode="encoder", **self.kwargs)
        fc1_a.build_engine(fc1_a.create_network(fc1_a.builder),
                           fc1_a.create_builder_config(),
                           fc1_a.batch_size,
                           engine_dir / "fc1_a.plan")

        logging.info("Building fc1_b...")
        fc1_b = RNNTJointFC1(*self.args, mode="decoder", **self.kwargs)
        fc1_b.build_engine(fc1_b.create_network(fc1_b.builder),
                           fc1_b.create_builder_config(),
                           fc1_b.batch_size,
                           engine_dir / "fc1_b.plan")

        logging.info("Building joint backend...")
        joint_backend = RNNTJointBackend(engine_dir, *self.args, **self.kwargs)
        joint_backend.build_engine(joint_backend.create_network(joint_backend.builder),
                                   joint_backend.create_builder_config(),
                                   joint_backend.batch_size,
                                   engine_dir / "joint_backend.plan")

        logging.info("Building Iselect...")
        isel = RNNTIsel(*self.args, **self.kwargs)
        isel.build_engine(isel.create_network(isel.builder),
                          isel.create_builder_config(),
                          isel.batch_size,
                          engine_dir / "isel.plan")

        logging.info("Building Igather...")
        igather = RNNTIgather(*self.args, **self.kwargs)
        igather.build_engine(igather.create_network(igather.builder),
                             igather.create_builder_config(),
                             igather.batch_size,
                             engine_dir / "igather.plan")


class RNNT(LegacyBuilder):

    def __init__(self, args):
        super().__init__(RNNTEngineBuilderOp(**args))
        self.audio_fp16_input = dict_get(args, "audio_fp16_input", default=True)

    def build_engines(self):
        dali_pipeline = RNNTDaliPipelineOp(audio_fp16_input=self.audio_fp16_input)
        dali_pipeline.run(self.legacy_scratch, None)

        super().build_engines()

    def calibrate(self):
        # Only the encoder has calibration
        calib_data_map = dict_get(self.mitten_builder.kwargs,
                                  "calib_data_map",
                                  default="data_maps/rnnt_train_clean_512/val_map.txt")
        calib_args = dict(self.mitten_builder.kwargs)
        calib_args.update({"expose_state": False,
                           "input_dtype": "fp32",
                           "precision": "int8",
                           "max_seq_length": 512,
                           "force_calibration": True,
                           "calib_max_batches": 30,
                           "batch_size": 100,
                           "enc_batch_size": 100,
                           "calib_batch_size": 100,
                           "calib_data_map": calib_data_map})

        encoder = RNNTEncoder(**calib_args)
        encoder.create_profiles = encoder.calibration_profiles
        builder_config = encoder.create_builder_config()
        builder_config.int8_calibrator = encoder.get_calibrator(self.legacy_scratch.path / "preprocessed_data" / "rnnt_train_clean_512_fp32" / "fp32")
        engine_dir = self.mitten_builder.engine_dir(self.legacy_scratch)
        encoder.build_engine(encoder.create_network(encoder.builder),
                             builder_config,
                             encoder.batch_size,
                             engine_dir / "encoder_calib_run.plan")
