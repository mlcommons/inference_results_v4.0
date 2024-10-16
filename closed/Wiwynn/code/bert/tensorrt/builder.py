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
from nvmitten.pipeline import Operation, ScratchSpace
from nvmitten.utils import dict_get, logging

from code.common.fields import Fields
from code.common.mitten_compat import ArgDiscarder
from code.common.systems.system_list import SystemClassifications
from code.plugin import load_trt_plugin_by_network

from .builder_utils import BertConfig
from .network import (BERTVarSeqLenFP16,
                      BERTVarSeqLenINT8,
                      BERTFP8FasterTransformer)


class OrinFCMidTacticSelector(trt.IAlgorithmSelector):
    def select_algorithms(self, ctx, choices):
        if "fc_mid + PWN(PWN(PWN(PWN(PWN(PWN" in ctx.name:  # Apply to fc_mid + gelu
            # MLPINF-1828
            # Force TRT to select CASK conv kernels with better performance running in the harness on Orin
            forbidden_set = {
                5175522159945819109,  # 0x47d325fbbebba3e5
            }
            filtered_idxs = [idx
                             for idx, choice in enumerate(choices)
                             if choice.algorithm_variant.tactic not in forbidden_set]
            to_ret = filtered_idxs
        else:
            # By default, say that all tactics are acceptable:
            to_ret = [idx for idx, _ in enumerate(choices)]
        return to_ret

    def report_algorithms(self, ctx, choices):
        pass


class L4FcOutTacticSelector(trt.IAlgorithmSelector):
    def select_algorithms(self, ctx, choices):
        regex_to_tactic = {
            "_fc_out": int(-1177958131132575679),  # we need to force the negative value
        }
        for regex in regex_to_tactic:
            if regex in ctx.name:
                override_tactic = regex_to_tactic[regex]
                filtered_idxs = [idx
                                 for idx, choice in enumerate(choices)
                                 if choice.algorithm_variant.tactic == override_tactic]
                if len(filtered_idxs) == 0:
                    pass
                else:
                    return filtered_idxs

        # By default, say that all tactics are acceptable:
        to_ret = [idx for idx, _ in enumerate(choices)]
        return to_ret

    def report_algorithms(self, ctx, choices):
        pass


class BertEngineBuilderOp(TRTBuilder,
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
                 model_path: str = None,
                 # Override the normal default values
                 workspace_size: int = 5 << 30,
                 # Benchmark specific values
                 batch_size: int = 1,
                 seq_len: int = 384,
                 bert_opt_seqlen: int = 384,
                 energy_aware_kernels: bool = False,
                 use_small_tile_gemm_plugin: bool = False,
                 use_fp8: bool = False,
                 **kwargs):
        """Creates a BertEngineBuilderOp.

        Args:
            config_ver (str): Legacy field. Identifier for the benchmark configuration. (Default: "default")
            model_path (str): Path to the model file. (Default: None)
            workspace_size (int): Size of the TRT workspace allocation in bytes. If the batch size is greater than 512,
                                  this value is set to (7 << 30), or 7GB. (Default: 5 << 30, or 5 GB)
            batch_size (int): Batch size to build the engine for. (Default: 1)
            seq_len (int): Sequence length for BERT. This is pre-determined and set in preprocessing. To run with a
                           different sequence length, re-run preprocessing again. (Default: 384)
            bert_opt_seqlen (int): opt_shape provided to TRT builder in the optimization profile. (Default: 384)
            energy_aware_kernels (bool): Only used for Ada systems. Enables SM89 Energy Aware kernels. (Default: False)
            use_small_tile_gemm_plugin (bool): Enable Small Tile GEMM plugin. (Default: False)
            use_fp8 (bool): Use FP8 precision. Only supported on Hopper and Ada. (Default: False)
        """
        self.batch_size = batch_size
        if self.batch_size > 512:
            # Tactics selection is limited at large batch sizes
            workspace_size = 7 << 30
            logging.info(f"Large batch size detected for BERT. Overriding TensorRT workspace size to {workspace_size}")

        super().__init__(workspace_size=workspace_size,
                         **kwargs)

        self.config_ver = config_ver
        self.bert_config = BertConfig()

        self.seq_len = seq_len
        self.bert_opt_seqlen = bert_opt_seqlen
        self.energy_aware_kernels = energy_aware_kernels

        self.is_int8 = (self.precision == Precision.INT8)
        self.model_path = model_path
        if model_path is None:
            if self.is_int8:
                self.model_path = "build/models/bert/bert_large_v1_1_fake_quant.onnx"
            else:
                self.model_path = "build/models/bert/bert_large_v1_1.onnx"

        self.use_small_tile_gemm_plugin = use_small_tile_gemm_plugin
        self.use_fp8 = use_fp8
        if self.use_fp8:
            self.create_profiles = self.gpu_profiles_fp8

    def gpu_profiles(self, network: trt.INetworkDefinition, builder_config: trt.IBuilderConfig, batch_size: int):
        if not hasattr(self, "profiles"):
            self.profiles = []

        # The harness expects i -> S -> B. This should be fine, since now there is only one S per engine
        for i in range(self.num_profiles):
            profile = self.builder.create_optimization_profile()
            assert network.num_inputs == 4, "Unexpected number of inputs"
            assert network.get_input(0).name == 'input_ids'
            assert network.get_input(1).name == 'segment_ids'
            assert network.get_input(2).name == 'cu_seqlens'
            assert network.get_input(3).name == 'max_seqlen'

            B = self.batch_size
            S = self.seq_len

            # TODO Like this, we can only control granularity using multiples of max_seqlen (B*S)
            # Investigate if this can be improved otherwise
            min_shape = (1,)  # TODO is it an issue to cover such a wide range?
            max_shape = (B * S,)
            opt_shape = (B * self.bert_opt_seqlen,)
            profile.set_shape('input_ids', min_shape, opt_shape, max_shape)
            profile.set_shape('segment_ids', min_shape, opt_shape, max_shape)
            profile.set_shape('cu_seqlens', (1 + 1,), (B + 1,), (B + 1,))
            profile.set_shape('max_seqlen', (1,), (S,), (S,))
            if not profile:
                raise RuntimeError("Invalid optimization profile!")
            builder_config.add_optimization_profile(profile)
            self.profiles.append(profile)

    def gpu_profiles_fp8(self, network: trt.INetworkDefinition, builder_config: trt.IBuilderConfig, batch_size: int):
        if not hasattr(self, "profiles"):
            self.profiles = []

        for i in range(self.num_profiles):
            profile = self.builder.create_optimization_profile()
            assert network.num_inputs == 3, "Unexpected number of inputs"
            assert network.get_input(0).name == 'input_ids'
            assert network.get_input(1).name == 'token_type_ids'
            assert network.get_input(2).name == 'sequence_lengths'

            B = self.batch_size
            S = self.seq_len
            min_shape = (1, 1)  # TODO is it an issue to cover such a wide range?
            max_shape = (B, S)
            opt_shape = (B, S)
            profile.set_shape('input_ids', min_shape, opt_shape, max_shape)
            profile.set_shape('token_type_ids', min_shape, opt_shape, max_shape)
            profile.set_shape('sequence_lengths', (1,), (B,), (B,))
            if not profile:
                raise RuntimeError("Invalid optimization profile!")
            builder_config.add_optimization_profile(profile)
            self.profiles.append(profile)

    def engine_name(self,
                    device_type: str,
                    batch_size: int,
                    precision: str,
                    subnetwork_name: str = None,
                    tag: str = "default") -> str:
        # Override implementation of engine_name to match legacy behavior to include BERT metadata
        if not precision:
            if hasattr(self, "precision"):
                # TODO: self.precision is currently a string, but in the case it is an AliasedNameEnum member, add support
                # to use .valstr()
                precision = self.precision
            else:
                raise ValueError("precision cannot be None if self.precision is not set.")

        if subnetwork_name:
            device_type = f"{device_type}-{subnetwork_name}"

        name = self.benchmark.valstr()
        scenario = self.scenario.valstr()
        base_name = f"{name}-{scenario}-{device_type}-{precision}"
        metadata = f"S_{self.seq_len}_B_{self.batch_size}_P_{self.num_profiles}_vs"  # 'vs' denotes variable seqlen
        return f"{base_name}_{metadata}.{tag}.plan"

    def create_network(self, builder: trt.Builder, scratch_space: ScratchSpace):
        load_trt_plugin_by_network("bert", args={"use_fp8": self.use_fp8})

        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        if self.is_int8:
            logging.info("Using INT8 network")
            BERTVarSeqLenINT8(network,
                              self.bert_config,
                              use_small_tile_gemm_plugin=self.use_small_tile_gemm_plugin)
        elif self.use_fp8:
            logging.info("Using FP8 network")
            ft_weights_dir = "/opt/fp8/faster-transformer-bert-fp8-weights-scales"
            BERTFP8FasterTransformer(network, self.bert_config, self.seq_len, ft_weights_dir)
        else:
            logging.info("Using FP16 network")
            BERTVarSeqLenFP16(network, self.bert_config)
        return network

    def create_builder_config(self, *args, **kwargs):
        builder_config = super().create_builder_config(*args, **kwargs)

        # Always allow FP16 fallback
        builder_config.set_flag(trt.BuilderFlag.FP16)

        # https://nvbugs/3902152
        # Apply tactic selector/filter to enforce kernels on Orin:
        if SystemClassifications.is_orin():
            if self.batch_size == 1:
                logging.info(f"Enforcing conv kernels for Orin BERT singlestream")
                builder_config.algorithm_selector = OrinFCMidTacticSelector()
        # Apply tactic selector/filter to enforce sm89 energy aware kernels on L4:
        elif SystemClassifications.is_ada() and self.energy_aware_kernels and not use_fp8:
            logging.info(f"Enforcing L4 FcOut tactics selection")
            builder_config.algorithm_selector = L4FcOutTacticSelector()
        return builder_config

    def build_engine(self,
                     network: trt.INetworkDefinition,
                     builder_config: trt.IBuilderConfig,
                     batch_size: int,
                     save_to: PathLike):
        # Unsure if we need to override this method at all. The only difference is that the old legacy implementation
        # used trt.Builder.build_serialized_network instead of trt.Builder.build_engine.

        save_to = Path(save_to)
        if save_to.is_file():
            logging.warning(f"{save_to} already exists. This file will be overwritten")
        save_to.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Building engine to {save_to}")

        if network.has_implicit_batch_dimension:
            logging.info(f"Network uses implicit batch size. Setting max_batch_size to {batch_size}.")
            self.builder.max_batch_size = batch_size
        else:
            logging.info(f"Building optimization profiles.")
            self.create_profiles(network, builder_config, batch_size)

        engine_bytes = self.builder.build_serialized_network(network, builder_config)
        assert engine_bytes is not None, "Engine Build Failed!"
        with save_to.open(mode='wb') as f:
            f.write(engine_bytes)

    def run(self, scratch_space, dependency_outputs):
        builder_config = self.create_builder_config()
        network = self.create_network(self.builder, scratch_space)
        engine_dir = self.engine_dir(scratch_space)
        engine_name = self.engine_name("gpu",
                                       self.batch_size,
                                       self.precision,
                                       tag=self.config_ver)
        engine_fpath = engine_dir / engine_name
        self.build_engine(network, builder_config, self.batch_size, engine_fpath)


class BERT(LegacyBuilder):
    """Temporary spoofing class to wrap around Mitten to adhere to the old API.
    """

    def __init__(self, args):
        # Legacy behavior - num_profiles should be set by gpu_inference_streams instead of gpu_copy_streams.
        # In BERT harness, multiple profiles are created for various sequence length bins, which are dispatched to
        # dedicated inference streams.
        args["num_profiles"] = args.get("gpu_inference_streams", 4)
        super().__init__(BertEngineBuilderOp(**args))
