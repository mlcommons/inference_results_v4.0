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
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt

import torch
from torch import distributed as torch_distrib

from nvmitten.nvidia.builder import (CalibratableTensorRTEngine, TRTBuilder, MLPerfInferenceEngine, LegacyBuilder)
from nvmitten.pipeline import Operation

from code.common import logging
from code.common.constants import TRT_LOGGER
from code.common.mitten_compat import ArgDiscarder
from code.plugin import load_trt_plugin_by_network
load_trt_plugin_by_network("dlrmv2")

from .criteo import CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE, CRITEO_SYNTH_MULTIHOT_SIZES, CriteoDay23Dataset
from .calibrator import DLRMv2Calibrator


# Distributed Torch libraries to import DLRMv2's sharded checkpoint
from code.common.systems.system_list import SystemClassifications
if not SystemClassifications.is_soc():
    from torchrec import EmbeddingBagCollection
    from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
    from torchrec.datasets.criteo import INT_FEATURE_COUNT, CAT_FEATURE_COUNT
    from torchrec.distributed.comm import get_local_size
    from torchrec.distributed.model_parallel import DistributedModelParallel, get_default_sharders
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.planner.storage_reservations import HeuristicalStorageReservation
    from torchrec.models.dlrm import DLRM_DCN, DLRMTrain
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchsnapshot import Snapshot


class DLRMv2_Model:
    def __init__(self,
                 model_path: str = "/home/mlperf_inf_dlrmv2/model/model_weights",
                 num_embeddings_per_feature: int = CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE,
                 embedding_dim: int = 128,
                 dcn_num_layers: int = 3,
                 dcn_low_rank_dim: int = 512,
                 dense_arch_layer_sizes: List[int] = (512, 256, 128),
                 over_arch_layer_sizes: List[int] = (1024, 1024, 512, 256, 1),
                 load_ckpt_on_gpu: bool = False):

        self.model_path = model_path
        self.num_embeddings_per_feature = list(num_embeddings_per_feature)
        self.embedding_dim = embedding_dim
        self.dcn_num_layers = dcn_num_layers
        self.dcn_low_rank_dim = dcn_low_rank_dim
        self.dense_arch_layer_sizes = list(dense_arch_layer_sizes)
        self.over_arch_layer_sizes = list(over_arch_layer_sizes)

        self.state_dict_path = Path(Path(self.model_path).parent, 'mini_state_dict.pt')

        if load_ckpt_on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.distrib_backend = "nccl"

        else:
            self.device = torch.device("cpu")
            self.distrib_backend = "gloo"

        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        torch_distrib.init_process_group(backend=self.distrib_backend, rank=0, world_size=1)

        # cache model to avoid re-loading
        self.model = None

    def load_state_dict(self):
        if self.state_dict_path.exists():
            # if possible dont load full pytorch model, only state dict. this is faster
            logging.info(f'Loading State Dict from: {self.state_dict_path}')
            return torch.load(str(self.state_dict_path))

        else:
            # load model from sharded files using pytorch & cache state dict for subsequent runs
            self.model = self.load_model()
            return self.model.state_dict()

    def load_model(self, return_snapshot: bool = False):
        logging.info('Loading Model...')
        self.embedding_bag_configs = [
            EmbeddingBagConfig(name=f"t_{feature_name}",
                               embedding_dim=self.embedding_dim,
                               num_embeddings=self.num_embeddings_per_feature[feature_idx],
                               feature_names=[feature_name])
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        ]

        # create model
        self.embedding_bag_collection = EmbeddingBagCollection(tables=self.embedding_bag_configs, device=torch.device("meta"))
        torchrec_dlrm_config = DLRM_DCN(embedding_bag_collection=self.embedding_bag_collection, dense_in_features=len(DEFAULT_INT_NAMES),
                                        dense_arch_layer_sizes=self.dense_arch_layer_sizes,
                                        over_arch_layer_sizes=self.over_arch_layer_sizes,
                                        dcn_num_layers=self.dcn_num_layers,
                                        dcn_low_rank_dim=self.dcn_low_rank_dim,
                                        dense_device=self.device)
        torchrec_dlrm_model = DLRMTrain(torchrec_dlrm_config)

        # distribute the model
        planner = EmbeddingShardingPlanner(
            topology=Topology(local_world_size=get_local_size(), world_size=torch_distrib.get_world_size(), compute_device=self.device.type),
            storage_reservation=HeuristicalStorageReservation(percentage=0.05)
        )
        plan = planner.collective_plan(torchrec_dlrm_model, get_default_sharders(), torch_distrib.GroupMember.WORLD)
        model = DistributedModelParallel(module=torchrec_dlrm_model,
                                         device=self.device,
                                         plan=plan)

        # load weights
        snapshot = Snapshot(path=self.model_path)
        snapshot.restore(app_state={"model": model})
        model.eval()

        # remove embeddings from state dict
        minified_sd = model.state_dict().copy()
        for key in [key for key in minified_sd.keys() if 'embedding_bags' in key]:
            del minified_sd[key]

        # save a stripped state dict for easier loading
        torch.save(minified_sd, str(self.state_dict_path))

        if return_snapshot:
            return model, snapshot

        else:
            return model

    def get_embedding_weight(self, cat_feature_idx: int):
        assert cat_feature_idx < len(DEFAULT_CAT_NAMES)

        # load model if not already loaded
        if not self.model:
            self.model = self.load_model()

        embedding_bag_state = self.model.module.model.sparse_arch.embedding_bag_collection.state_dict()
        key = f"embedding_bags.t_cat_{cat_feature_idx}.weight"
        out = torch.zeros(embedding_bag_state[key].metadata().size, device=self.device)
        embedding_bag_state[key].gather(0, out=out)
        return out

    def dump_embedding_weights(self, save_dir: os.PathLike):
        def int8_quantize(mega_table):
            # compute scales
            mults = np.ndarray(shape=(CAT_FEATURE_COUNT))
            scales = np.ndarray(shape=(CAT_FEATURE_COUNT))
            for id, table in enumerate(mega_table):
                maxAbsVal = abs(max(table.max(), table.min(), key=abs))
                scales[id] = maxAbsVal / 127.0
                mults[id] = 1.0 / scales[id]

            # multiply scales, symmetric quantization
            mega_table_int8 = []
            for id, table in enumerate(mega_table):
                mega_table_int8.append(np.minimum(np.maximum(np.rint(table * mults[id]), -127), 127).astype(np.int8))

            return (np.vstack(mega_table_int8).reshape(-1).astype(np.int8), scales.astype(np.float32))

        # collect mega table
        mega_table = []
        for i in range(len(DEFAULT_CAT_NAMES)):
            weight = self.get_embedding_weight(i).cpu()
            mega_table.append(weight.numpy())

        # compute mega_table and scales for all support precisions
        precision_to_tensor = {
            'fp32': (np.vstack(mega_table).reshape(-1).astype(np.float32), None),
            'fp16': (np.vstack(mega_table).reshape(-1).astype(np.float16), None),
            'int8': int8_quantize(mega_table)
        }

        # save all mega_tables and scales
        for precision, (table, scales) in precision_to_tensor.items():
            table_path = save_dir / f"mega_table_{precision}.npy"
            logging.info(f'Saving mega_table [{precision}]: {table_path}')

            with open(table_path, 'wb') as table_file:
                np.save(table_file, table)

            if scales is not None:
                scales_path = save_dir / f'mega_table_scales.npy'
                logging.info(f'Saving mega_table_scales [{precision}]: {scales_path}')

                with open(scales_path, 'wb') as scales_file:
                    np.save(scales_file, scales)

    def load_embeddings(self, from_dir: os.PathLike):
        embedding_bag_configs = [
            EmbeddingBagConfig(name=f"t_{feature_name}",
                               embedding_dim=self.embedding_dim,
                               num_embeddings=self.num_embeddings_per_feature[feature_idx],
                               feature_names=[feature_name])
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        ]

        embedding_bag_collection = EmbeddingBagCollection(tables=embedding_bag_configs,
                                                          device=self.device)

        # torchrec 0.3.2 does not support init_fn as a EmbeddingBagConfig parameter.
        # Manually implement it here.
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES):
            with open(Path(from_dir) / f"embed_feature_{feature_idx}.weight.pt", 'rb') as f:
                dat = torch.load(f, map_location=self.device)
            with torch.no_grad():
                embedding_bag_collection.embedding_bags[f"t_{feature_name}"].weight.copy_(dat)
        return embedding_bag_collection


class DLRMv2Arch:
    """Loose representation of the DLRMv2 architecture based on TorchRec source:
    https://github.com/pytorch/torchrec/blob/main/torchrec/models/dlrm.py

    The components of the model are as follows:

    1. SparseArch (Embedding table, isolated into DLRMv2_Model)
    2. DenseArch (Bottom MLP)
    3. InteractionDCNArch (DCNv2, or sometimes referred to as interactions network / layer)
    4. OverArch (Top MLP + final linear layer)
    """

    def __init__(self,
                 state_dict,
                 bot_mlp_depth: int = 3,
                 crossnet_depth: int = 3,
                 top_mlp_depth: int = 4):
        self.bot_mlp_depth = bot_mlp_depth
        self.bottom_mlp = self.create_bot_mlp(state_dict)

        self.crossnet_depth = crossnet_depth
        self.crossnet = self.create_crossnet(state_dict)

        self.top_mlp_depth = top_mlp_depth
        self.top_mlp = self.create_top_mlp(state_dict)

        self.final_linear = self.create_final_linear(state_dict)

    def create_bot_mlp(self, state_dict):
        """ Bottom MLP keys
        model.dense_arch.model._mlp.0._linear.bias
        model.dense_arch.model._mlp.0._linear.weight
        model.dense_arch.model._mlp.1._linear.bias
        model.dense_arch.model._mlp.1._linear.weight
        model.dense_arch.model._mlp.2._linear.bias
        model.dense_arch.model._mlp.2._linear.weight
        """
        conf = defaultdict(dict)
        for i in range(self.bot_mlp_depth):
            key_prefix = f"model.dense_arch.model._mlp.{i}._linear."
            conf[i]["weight"] = state_dict[key_prefix + "weight"]
            conf[i]["bias"] = state_dict[key_prefix + "bias"]
        return conf

    def create_crossnet(self, state_dict):
        """ DCNv2 crossnet is based on torchrec.modules.crossnet.LowRankCrossNet:
            - https://pytorch.org/torchrec/torchrec.modules.html#torchrec.modules.crossnet.LowRankCrossNet
            - https://github.com/pytorch/torchrec/blob/42c55844d29343c644521e810597fd67017eac8f/torchrec/modules/crossnet.py#L90

        Keys:
        model.inter_arch.crossnet.V_kernels.0
        model.inter_arch.crossnet.V_kernels.1
        model.inter_arch.crossnet.V_kernels.2
        model.inter_arch.crossnet.W_kernels.0
        model.inter_arch.crossnet.W_kernels.1
        model.inter_arch.crossnet.W_kernels.2
        model.inter_arch.crossnet.bias.0
        model.inter_arch.crossnet.bias.1
        model.inter_arch.crossnet.bias.2
        """
        conf = defaultdict(dict)
        for i in range(self.crossnet_depth):
            V = f"model.inter_arch.crossnet.V_kernels.{i}"
            W = f"model.inter_arch.crossnet.W_kernels.{i}"
            bias = f"model.inter_arch.crossnet.bias.{i}"
            conf[i]['V'] = state_dict[V]
            conf[i]['W'] = state_dict[W]
            conf[i]["bias"] = state_dict[bias]
        return conf

    def create_top_mlp(self, state_dict):
        """ Top MLP keys
        model.over_arch.model.0._mlp.0._linear.bias
        model.over_arch.model.0._mlp.0._linear.weight
        model.over_arch.model.0._mlp.1._linear.bias
        model.over_arch.model.0._mlp.1._linear.weight
        model.over_arch.model.0._mlp.2._linear.bias
        model.over_arch.model.0._mlp.2._linear.weight
        model.over_arch.model.0._mlp.3._linear.bias
        model.over_arch.model.0._mlp.3._linear.weight
        """
        conf = defaultdict(dict)
        for i in range(self.top_mlp_depth):
            key_prefix = f"model.over_arch.model.0._mlp.{i}._linear."
            conf[i]["weight"] = state_dict[key_prefix + "weight"]
            conf[i]["bias"] = state_dict[key_prefix + "bias"]
        return conf

    def create_final_linear(self, state_dict):
        """ Probability reduction linear layer keys
        model.over_arch.model.1.bias
        model.over_arch.model.1.weight
        """
        conf = {
            "weight": state_dict["model.over_arch.model.1.weight"],
            "bias": state_dict["model.over_arch.model.1.bias"],
        }
        return conf


class DLRMv2TRTNetwork:
    def __init__(self,
                 batch_size: int,
                 network: trt.INetworkDefinition,
                 use_embedding_lookup_plugin: bool = True,
                 embedding_weights_on_gpu_part: float = 1.0,
                 model_path: str = "/home/mlperf_inf_dlrmv2/model/model_weights",
                 mega_table_npy_file: str = '/home/mlperf_inf_dlrmv2/model/embedding_weights/mega_table_int8.npy',
                 row_frequencies_npy_filepath: str = '/home/mlperf_inf_dlrmv2/criteo/day23/row_frequencies.npy',
                 mega_table_scales_npy_file: str = '/home/mlperf_inf_dlrmv2/model/embedding_weights/mega_table_scales.npy',
                 reduced_precision_io: int = 2,  # 0: fp32, 1: fp16, 2: int8, -1: calibration mode
                 high_accuracy_mode: bool = False):

        self.verbose = True
        self.logger = TRT_LOGGER
        self.logger.min_severity = trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO

        self.batch_size = batch_size
        self.network = network
        self.model_path = model_path
        assert self.network is not None, "TRT Network cannot be None"

        # DLRMv2EmbeddingLookupPlugin
        self.use_embedding_lookup_plugin = use_embedding_lookup_plugin
        self.embedding_weights_on_gpu_part = embedding_weights_on_gpu_part
        self.reduced_precision_io = reduced_precision_io
        self.high_accuracy_mode = high_accuracy_mode
        self.mega_table_npy_file = mega_table_npy_file
        self.row_frequencies_npy_filepath = row_frequencies_npy_filepath
        self.mega_table_scales_npy_file = mega_table_scales_npy_file

        # network precision configs
        if self.reduced_precision_io == 0 or self.reduced_precision_io == -1:
            # fp32 or calibration mode
            self.dense_dtype = trt.DataType.FLOAT
            self.bot_mlp_precision = trt.float32
            self.interaction_op_precision = trt.float32
            self.top_mlp_precision = trt.float32
            self.final_linear_precision = trt.float32

        elif self.reduced_precision_io == 1:
            # fp16 mode
            self.dense_dtype = trt.DataType.HALF
            self.bot_mlp_precision = trt.float16
            self.interaction_op_precision = trt.float16
            self.top_mlp_precision = trt.float16
            self.final_linear_precision = trt.float16

        elif self.reduced_precision_io == 2:
            # int8 mode
            self.dense_dtype = trt.DataType.HALF
            self.bot_mlp_precision = trt.int8
            self.interaction_op_precision = trt.int8
            self.top_mlp_precision = trt.int8
            self.final_linear_precision = trt.int8

            # NOTE(vir): WAR for high accuracy mode
            if self.high_accuracy_mode:
                self.interaction_op_precision = trt.float16

        else:
            assert False, "ERROR: invalid value for reduced_precision_io"

        # print out network precision configs
        self.pprint_network_precision_configs()

        # initialize trt network
        self.initialize()

    def pprint_network_precision_configs(self):
        precision_to_str = {
            trt.DataType.FLOAT: 'fp32',
            trt.DataType.HALF: 'fp16',
            trt.DataType.INT8: 'int8',
            trt.float32: 'fp32',
            trt.float16: 'fp16',
            trt.int8: 'int8'
        }

        precision_to_embedding_str = {
            -1: 'fp32',
            0: 'fp32',
            1: 'fp16',
            2: 'int8'
        }

        logging.info(f'Network Config:'
                     f'\n\thigh_accuracy_mode: {self.high_accuracy_mode}'
                     f'\n\tdense_dtype: {precision_to_str[self.dense_dtype]}'
                     f'\n\tembeddings: {precision_to_embedding_str[self.reduced_precision_io]}'
                     f'\n\tbot_mlp_precision: {precision_to_str[self.bot_mlp_precision]}'
                     f'\n\tinteraction_op_precision: {precision_to_str[self.interaction_op_precision]}'
                     f'\n\ttop_mlp_precision: {precision_to_str[self.top_mlp_precision]}'
                     f'\n\tfinal_linear_precision: {precision_to_str[self.final_linear_precision]}')

    def parse_calibration(self, cache_path: str = 'code/dlrm-v2/tensorrt/calibrator.cache'):
        if not os.path.exists(cache_path):
            if self.reduced_precision_io == 2:
                assert False, "ERROR: calibration cache missing, int8 engine cannot be built"

            return

        with open(cache_path, 'rb') as f:
            lines = f.read().decode('ascii').splitlines()

        calibration_dict = {}
        for line in lines:
            split = line.split(':')
            if len(split) != 2:
                continue

            tensor = split[0]
            drange = np.uint32(int(split[1], 16)).view(np.float32).item() * np.float32(127.0)
            calibration_dict[tensor] = drange

        return calibration_dict

    def initialize(self):
        dlrm_model = DLRMv2_Model(model_path=self.model_path)
        state_dict = dlrm_model.load_state_dict()

        # create mega_table and scales file if needed
        mega_table_npy_file_path = Path(self.mega_table_npy_file)
        mega_table_scales_npy_file_path = Path(self.mega_table_scales_npy_file)
        if not mega_table_npy_file_path.exists() or not mega_table_scales_npy_file_path.exists():
            logging.info("Generating missing embedding files...")
            mega_table_npy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dlrm_model.dump_embedding_weights(mega_table_npy_file_path.parent)
        else:
            logging.info("Found embedding mega_table and scales file.")

        # create TRT Network
        self.arch = DLRMv2Arch(state_dict)
        self.embedding_size = dlrm_model.embedding_dim

        # create numerical input
        numerical_input = self.network.add_input('numerical_input',
                                                 self.dense_dtype,
                                                 (-1 if self.reduced_precision_io != -1 else self.batch_size, INT_FEATURE_COUNT, 1, 1))

        # create bottom mlp
        # input for bot_mlp:                 [-1, 13, 1, 1]
        self.bottom_mlp = self._build_mlp(self.arch.bottom_mlp,
                                          numerical_input,
                                          INT_FEATURE_COUNT,
                                          'bot_mlp',
                                          precision=self.bot_mlp_precision)

        # flatten dense inputs:              [-1, 128, 1, 1] -> [-1, 128]
        unsqueezed_dense_input = self.bottom_mlp.get_output(0)
        squeeze_dense = self.network.add_shuffle(unsqueezed_dense_input)
        squeeze_dense.reshape_dims = (-1, self.embedding_size)
        squeeze_dense.name = 'squeeze_dense'
        squeeze_dense.get_output(0).name = 'squeeze_dense.output'
        dense_input = squeeze_dense.get_output(0)

        # create embedding lookup
        if self.use_embedding_lookup_plugin:
            # create sparse input:           [-1, total_hotness]
            sparse_input = self.network.add_input('sparse_input',
                                                  trt.DataType.INT32,
                                                  (-1 if self.reduced_precision_io != -1 else self.batch_size, sum(CRITEO_SYNTH_MULTIHOT_SIZES)))

            # dense input from bot_mlp:      [-1, 128]
            # sparse input from harness:     [-1, total_hotness]
            dlrm_embedding_lookup_plugin = self.get_dlrmv2_embedding_lookup_plugin()
            embedding_op = self.network.add_plugin_v2([dense_input, sparse_input], dlrm_embedding_lookup_plugin)
            embedding_op.name = "dlrmv2_embedding_lookup"
            embedding_op.get_output(0).name = "dlrmv2_embedding_lookup.output"

            # input for interaction layer:   [-1, 3456]
            interaction_input = embedding_op.get_output(0)

        else:
            # use pytorch for embedding lookups

            # NOTE(vir): only tested with fp32 precision, disable other modes
            assert self.reduced_precision_io == 0, 'DLRMv2 only supports fp32 when using pytorch for embedding lookup.'

            # pytorch embedding lookup:      [-1, 26, 128]
            embedding_lookup = self.network.add_input('embedding_lookup',
                                                      trt.DataType.FLOAT,
                                                      (-1 if self.reduced_precision_io != -1 else self.batch_size, CAT_FEATURE_COUNT, self.embedding_size))

            # flatten:                       [-1, 26, 128] -> [-1, 3328]
            squeeze_sparse = self.network.add_shuffle(embedding_lookup)
            squeeze_sparse.reshape_dims = (-1, CAT_FEATURE_COUNT * self.embedding_size)
            squeeze_sparse.name = "sqeeze_sparse"
            squeeze_sparse.get_output(0).name = "sqeeze_sparse.output"
            sparse_input = squeeze_sparse.get_output(0)

            # concatenate:                   [-1, 128, 1, 1] : [-1, 26, 128] -> [-1, 3456]
            embedding_concat = self.network.add_concatenation([dense_input, sparse_input])
            embedding_concat.axis = 1
            embedding_concat.name = "embedding_concat"
            embedding_concat.get_output(0).name = "embedding_concat.output"

            # input for interaction layer:   [-1, 3456]
            interaction_input = embedding_concat.get_output(0)

        # create interaction op
        # input from plugin/pytorch:         [-1, 3456]
        self.interaction_op = self._build_interaction_op(self.arch.crossnet,
                                                         interaction_input,
                                                         precision=self.interaction_op_precision)

        # create top mlp
        # input from interaction op:         [-1, 3456]
        top_mlp_input = self.interaction_op.get_output(0)
        self.top_mlp = self._build_mlp(self.arch.top_mlp,
                                       top_mlp_input,
                                       top_mlp_input.shape[1],
                                       'top_mlp',
                                       precision=self.top_mlp_precision)

        # create final linear layer
        # input from top mlp:                [-1, 3456]
        final_linear_input = self.top_mlp.get_output(0)
        self.final_linear = self._build_linear(self.arch.final_linear,
                                               final_linear_input,
                                               final_linear_input.shape[1],
                                               'final_linear',
                                               add_relu=False,
                                               precision=self.final_linear_precision)

        # create sigmoid output layer
        # input from final linear:           [-1, 3456]
        sigmoid_input = self.final_linear.get_output(0)
        self.sigmoid_layer = self.network.add_activation(sigmoid_input, trt.ActivationType.SIGMOID)
        self.sigmoid_layer.name = "sigmoid"
        self.sigmoid_layer.get_output(0).name = "sigmoid.output"

        # mark output
        sigmoid_output = self.sigmoid_layer.get_output(0)
        sigmoid_output.dtype = trt.float32
        self.network.mark_output(sigmoid_output)

    def _build_mlp(self,
                   config,
                   in_tensor,
                   in_channels,
                   name_prefix,
                   use_conv_for_fc=False,
                   precision=trt.float32):

        for index, state in config.items():
            layer = self._build_linear(state,
                                       in_tensor,
                                       in_channels,
                                       f'{name_prefix}_{index}',
                                       use_conv_for_fc=use_conv_for_fc,
                                       precision=precision)

            in_channels = state['weight'].shape[::-1][-1]
            in_tensor = layer.get_output(0)

        return layer

    def _build_linear(self,
                      state,
                      in_tensor,
                      in_channels,
                      name,
                      add_relu=True,
                      use_conv_for_fc=False,
                      precision=trt.float32):

        weights = state['weight'].numpy()
        bias = state['bias'].numpy()

        shape = weights.shape[::-1]
        out_channels = shape[-1]

        if use_conv_for_fc:
            layer = self.network.add_convolution(in_tensor, out_channels, (1, 1), weights, bias)
        else:
            layer = self.network.add_fully_connected(in_tensor, out_channels, weights, bias)

        layer.precision = precision
        layer.name = name
        layer.get_output(0).name = name + ".output"

        if add_relu:
            layer = self.network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
            layer.precision = precision
            layer.name = name + ".relu"
            layer.get_output(0).name = name + ".relu.output"

        return layer

    def _build_interaction_op(self, config, x, precision=trt.float32, use_conv=True, use_explicit_qdq=False):
        # From LowRankCrossNet docs:
        # https://pytorch.org/torchrec/torchrec.modules.html#torchrec.modules.crossnet.LowRankCrossNet
        # x_next = x_0 * (matmul(W_curr, matmul(V_curr, x_curr)) + bias_curr) + x_curr

        # enable explicit qdq on tensor
        def insert_qdq(input, scale=1.0):
            if not use_explicit_qdq:
                return input

            scale = self.network.add_constant([1], np.array([scale], np.float32)).get_output(0)
            quant = self.network.add_quantize(input, scale).get_output(0)
            dequant = self.network.add_dequantize(quant, scale).get_output(0)
            return dequant

        if use_conv:
            # unsqueeze input [-1, 3456] -> [-1, 3456, 1, 1]
            unsqueeze = self.network.add_shuffle(x)
            unsqueeze.reshape_dims = (-1, (CAT_FEATURE_COUNT * self.embedding_size) + self.embedding_size, 1, 1)
            unsqueeze.precision = unsqueeze.precision if use_explicit_qdq else precision
            unsqueeze.name = 'interaction_in'
            unsqueeze.get_output(0).name = 'interaction_in.output'
            x = unsqueeze.get_output(0)

        x = insert_qdq(x)
        x0 = x

        for index, state in config.items():
            V = state['V'].numpy()  # 512 x 3456
            W = state['W'].numpy()  # 3456 x 512
            b = state['bias'].numpy()  # 1 x 3456

            # set weights
            if use_conv:
                V = V.reshape(*V.shape, 1, 1)
                W = W.reshape(*W.shape, 1, 1)
                b = b.reshape(1, b.shape[0], 1, 1)

            else:
                V_tens = self.network.add_constant(V.shape, V)
                V_tens.precision = V_tens.precision if use_explicit_qdq else precision
                V_tens.name = f'interaction.V_{index}'
                V_tens.get_output(0).name = f'interaction.V_{index}.output'
                V_tens = V_tens.get_output(0)
                V_tens = insert_qdq(V_tens)

                W_tens = self.network.add_constant(W.shape, W)
                W_tens.precision = W_tens.precision if use_explicit_qdq else precision
                W_tens.name = f'interaction.W_{index}'
                W_tens.get_output(0).name = f'interaction.W_{index}.output'
                W_tens = W_tens.get_output(0)
                W_tens = insert_qdq(W_tens)

                b_tens = self.network.add_constant([1, b.shape[0]], b)
                b_tens.precision = b_tens.precision if use_explicit_qdq else precision
                b_tens.name = f'interaction.b_{index}'
                b_tens.get_output(0).name = f'interaction.b_{index}.output'
                b_tens = b_tens.get_output(0)
                b_tens = insert_qdq(b_tens)

            # set operations
            vx = self.network.add_convolution(x, V.shape[0], (1, 1), V)                                              \
                if use_conv else                                                                                     \
                self.network.add_matrix_multiply(x, trt.MatrixOperation.NONE, V_tens, trt.MatrixOperation.TRANSPOSE)
            vx.precision = vx.precision if use_explicit_qdq else precision
            vx.name = f'interaction.vx_{index}'
            vx.get_output(0).name = f'interaction.vx_{index}.output'
            vx = vx.get_output(0)
            vx = insert_qdq(vx)

            wvx = self.network.add_convolution(vx, W.shape[0], (1, 1), W, b)                                          \
                if use_conv else                                                                                      \
                self.network.add_matrix_multiply(vx, trt.MatrixOperation.NONE, W_tens, trt.MatrixOperation.TRANSPOSE)
            wvx.precision = wvx.precision if use_explicit_qdq else precision
            wvx.name = f'interaction.wvx_{index}'
            wvx.get_output(0).name = f'interaction.wvx_{index}.output'
            wvx = wvx.get_output(0)
            wvx = insert_qdq(wvx)

            if use_conv:
                inner = wvx  # bias integrated in conv layer

            else:
                inner = self.network.add_elementwise(wvx, b_tens, trt.ElementWiseOperation.SUM)
                inner.precision = inner.precision if use_explicit_qdq else precision
                inner.name = f'interaction.inner_{index}'
                inner.get_output(0).name = f'interaction.inner_{index}.output'
                inner = inner.get_output(0)
                inner = insert_qdq(inner)

            left_term = self.network.add_elementwise(inner, x0, trt.ElementWiseOperation.PROD)
            left_term.precision = left_term.precision if use_explicit_qdq else precision
            left_term.name = f'interaction.left_term_{index}'
            left_term.get_output(0).name = f'interaction.left_term_{index}.output'
            left_term = left_term.get_output(0)
            left_term = insert_qdq(left_term)

            x_ = self.network.add_elementwise(left_term, x, trt.ElementWiseOperation.SUM)
            x_.precision = x.precision if use_explicit_qdq else precision
            x_.name = f'interaction.out_{index}'
            x_.get_output(0).name = f'interaction.out_{index}.output'

            # port for next layer
            x = x_.get_output(0)
            x = insert_qdq(x)

        if use_conv:
            output = x_

        else:
            # unsqueeze output [-1, 3456] -> [-1, 3456, 1, 1]
            unsqueeze = self.network.add_shuffle(x)
            unsqueeze.reshape_dims = (-1, (CAT_FEATURE_COUNT * self.embedding_size) + self.embedding_size, 1, 1)
            unsqueeze.precision = unsqueeze.precision if use_explicit_qdq else precision
            unsqueeze.name = 'interaction'
            unsqueeze.get_output(0).name = 'interaction.output'
            output = unsqueeze

        return output

    def get_dlrmv2_embedding_lookup_plugin(self):
        """Create a plugin layer for the DLRMv2 Embedding Lookup plugin and return it. """

        pluginName = "DLRMv2_EMBEDDING_LOOKUP_TRT"
        embeddingRows = sum(CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE)
        tableOffsets = np.concatenate(([0], np.cumsum(CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE).astype(np.int32)[:-1])).astype(np.int32)
        tableHotness = np.array(CRITEO_SYNTH_MULTIHOT_SIZES).astype(np.int32)
        totalHotness = sum(CRITEO_SYNTH_MULTIHOT_SIZES)
        reducedPrecisionIO = self.reduced_precision_io

        plugin = None
        for plugin_creator in trt.get_plugin_registry().plugin_creator_list:
            if plugin_creator.name == pluginName:
                embeddingSize_field = trt.PluginField("embeddingSize", np.array([self.embedding_size], dtype=np.int32), trt.PluginFieldType.INT32)
                embeddingRows_field = trt.PluginField("embeddingRows", np.array([embeddingRows], dtype=np.int32), trt.PluginFieldType.INT32)
                embeddingWeightsOnGpuPart_field = trt.PluginField("embeddingWeightsOnGpuPart", np.array([self.embedding_weights_on_gpu_part], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                tableHotness_field = trt.PluginField("tableHotness", tableHotness, trt.PluginFieldType.INT32)
                tableOffsets_field = trt.PluginField("tableOffsets", tableOffsets, trt.PluginFieldType.INT32)
                batchSize_field = trt.PluginField("batchSize", np.array([self.batch_size], dtype=np.int32), trt.PluginFieldType.INT32)
                embedHotnessTotal_field = trt.PluginField("embedHotnessTotal", np.array([totalHotness], dtype=np.int32), trt.PluginFieldType.INT32)
                embeddingWeightsFilepath_field = trt.PluginField("embeddingWeightsFilepath", np.array(list(self.mega_table_npy_file.encode()), dtype=np.int8), trt.PluginFieldType.CHAR)
                rowFrequenciesFilepath_field = trt.PluginField("rowFrequenciesFilepath", np.array(list(self.row_frequencies_npy_filepath.encode()), dtype=np.int8), trt.PluginFieldType.CHAR)
                embeddingScalesFilepath_field = trt.PluginField("embeddingScalesFilepath", np.array(list(self.mega_table_scales_npy_file.encode()), dtype=np.int8), trt.PluginFieldType.CHAR)
                reducedPrecisionIO_field = trt.PluginField("reducedPrecisionIO", np.array([reducedPrecisionIO], dtype=np.int32), trt.PluginFieldType.INT32)

                field_collection = trt.PluginFieldCollection([
                    embeddingSize_field,
                    embeddingRows_field,
                    embeddingWeightsOnGpuPart_field,
                    tableHotness_field,
                    tableOffsets_field,
                    batchSize_field,
                    embedHotnessTotal_field,
                    embeddingWeightsFilepath_field,
                    rowFrequenciesFilepath_field,
                    embeddingScalesFilepath_field,
                    reducedPrecisionIO_field
                ])
                plugin = plugin_creator.create_plugin(name=pluginName, field_collection=field_collection)

        return plugin


class DLRMv2EngineBuilderOp(CalibratableTensorRTEngine,
                            TRTBuilder,
                            MLPerfInferenceEngine,
                            Operation,
                            ArgDiscarder):
    @classmethod
    def immediate_dependencies(cls):
        return None

    def __init__(self,
                 workspace_size: int = 4 << 30,

                 # TODO: Legacy value - Remove after refactor is done.
                 config_ver: str = "default",

                 # TODO: This should be a relative path within the ScratchSpace.
                 model_path: str = "/home/mlperf_inf_dlrmv2/model/model_weights",

                 # Override the default values
                 calib_batch_size: int = 256,
                 calib_max_batches: int = 500,
                 calib_data_map: os.PathLike = Path("data_maps/criteo/cal_map.txt"),
                 cache_file: os.PathLike = Path("code/dlrm-v2/tensorrt/calibrator.cache"),

                 # Benchmark specific values
                 batch_size: int = 8192,
                 use_embedding_lookup_plugin: bool = True,
                 embedding_weights_on_gpu_part: float = 1.0,
                 mega_table_npy_file: str = '/home/mlperf_inf_dlrmv2/model/embedding_weights/mega_table_int8.npy',
                 mega_table_scales_npy_file: str = '/home/mlperf_inf_dlrmv2/model/embedding_weights/mega_table_scales.npy',
                 row_frequencies_npy_filepath: str = '/home/mlperf_inf_dlrmv2/criteo/day23/row_frequencies.npy',
                 reduced_precision_io: int = 2,
                 accuracy_level: str = "99%",
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
        self.use_embedding_lookup_plugin = use_embedding_lookup_plugin
        self.embedding_weights_on_gpu_part = embedding_weights_on_gpu_part
        self.mega_table_npy_file = mega_table_npy_file
        self.mega_table_scales_npy_file = mega_table_scales_npy_file
        self.row_frequencies_npy_filepath = row_frequencies_npy_filepath
        self.reduced_precision_io = reduced_precision_io
        self.high_accuracy_mode = accuracy_level == '99.9%'

        self.use_timing_cache = True
        self.timing_cache_file = f"./build/cache/dlrm_build_cache_{self.precision}.cache"

        if self.force_calibration:
            logging.info('Building Engine in Calibration Mode')

            # TODO(vir): find a better palce to set these
            self.mega_table_npy_file = '/home/mlperf_inf_dlrmv2/model/embedding_weights/mega_table.npy'
            self.embedding_weights_on_gpu_part = 0.3
            self.reduced_precision_io = -1

    # TODO(vir): put this default path in a better place
    # DLRM has it's own scratch space right now, so it's hard coded. We can fix this in the future.
    def get_calibrator(self, data_dir: str = "/home/mlperf_inf_dlrmv2/criteo/day23/fp32"):
        return DLRMv2Calibrator(
            data_dir=data_dir,
            calib_batch_size=self.calib_batch_size,
            calib_max_batches=self.calib_max_batches,
            force_calibration=self.force_calibration,
            cache_file=self.cache_file
        )

    def create_network(self, builder: trt.Builder):
        dlrm_network = DLRMv2TRTNetwork(batch_size=self.batch_size,
                                        network=builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)),
                                        use_embedding_lookup_plugin=self.use_embedding_lookup_plugin,
                                        embedding_weights_on_gpu_part=self.embedding_weights_on_gpu_part,
                                        model_path=self.model_path,
                                        mega_table_npy_file=self.mega_table_npy_file,
                                        mega_table_scales_npy_file=self.mega_table_scales_npy_file,
                                        row_frequencies_npy_filepath=self.row_frequencies_npy_filepath,
                                        reduced_precision_io=self.reduced_precision_io,
                                        high_accuracy_mode=self.high_accuracy_mode)

        return dlrm_network.network

    def run(self, scratch_space, dependency_outputs):
        builder_config = self.create_builder_config(self.builder)
        builder_config.int8_calibrator = self.get_calibrator()
        builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        builder_config.builder_optimization_level = 4  # Needed for ConvMulAdd fusion from Myelin

        if self.use_timing_cache:
            # load existing cache if found, else create a new one
            timing_cache = b""

            if os.path.exists(self.timing_cache_file):
                with open(self.timing_cache_file, 'rb') as f:
                    timing_cache = f.read()

            trt_timing_cache = builder_config.create_timing_cache(timing_cache)
            builder_config.set_timing_cache(trt_timing_cache, False)
            logging.info(f'Using Timing Cache: {self.timing_cache_file}')

        builder_config.set_flag(trt.BuilderFlag.FP16)
        builder_config.set_flag(trt.BuilderFlag.INT8)
        builder_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        # int8 mode
        # if self.reduced_precision_io == 2:
        #     builder_config.set_flag(trt.BuilderFlag.DIRECT_IO)

        network = self.create_network(self.builder)
        engine_dir = self.engine_dir(scratch_space)
        engine_name = self.engine_name("gpu", self.batch_size, self.precision, tag=self.config_ver)
        engine_fpath = engine_dir / engine_name

        self.build_engine(network, builder_config, self.batch_size, engine_fpath)

        if self.use_timing_cache:
            Path(self.timing_cache_file).parent.mkdir(parents=True, exist_ok=True)

            # save latest timing cache
            with open(self.timing_cache_file, 'wb') as f:
                f.write(trt_timing_cache.serialize())

            logging.info(f'Timing Cache Updated: {self.timing_cache_file}')


class DLRMv2(LegacyBuilder):
    """Temporary spoofing class to wrap around Mitten to adhere to the old API.
    """

    def __init__(self, args):
        self.mitten_builder = DLRMv2EngineBuilderOp(**args)
        super().__init__(self.mitten_builder)

    def calibrate(self):
        # NOTE(VIR): WAR: override mitten calibration with polygraphy, which gives better scales
        from polygraphy.backend.trt import EngineFromNetwork, TrtRunner, CreateConfig, Calibrator, CreateNetwork
        from polygraphy.logger import G_LOGGER
        from polygraphy import func

        # calibration data params
        lower_bound = 89137319
        upper_bound = 89265318
        batch_size = 256
        num_batches = 1 + ((upper_bound - lower_bound) // batch_size)

        # calibration model builder params
        cache_file = self.mitten_builder.cache_file
        model_path = self.mitten_builder.model_path
        mega_table_npy_file = self.mitten_builder.mega_table_npy_file
        mega_table_scales_npy_file = self.mitten_builder.mega_table_scales_npy_file
        row_frequencies_npy_filepath = self.mitten_builder.row_frequencies_npy_filepath

        # load dataset
        dataset = CriteoDay23Dataset('/home/mlperf_inf_dlrmv2/criteo/day23/fp32')

        # create network
        @func.extend(CreateNetwork())
        def create_network(_, network):
            DLRMv2TRTNetwork(batch_size=batch_size,
                             network=network,
                             use_embedding_lookup_plugin=True,
                             embedding_weights_on_gpu_part=0.3,
                             model_path=model_path,
                             mega_table_npy_file=mega_table_npy_file,
                             mega_table_scales_npy_file=mega_table_scales_npy_file,
                             row_frequencies_npy_filepath=row_frequencies_npy_filepath,
                             reduced_precision_io=-1)

        # calibration dataset generator
        def data_loader():
            for idx in range(num_batches):
                s = lower_bound + (idx + 0) * batch_size
                e = lower_bound + (idx + 1) * batch_size
                assert s < upper_bound and e <= upper_bound + 1

                batch = dataset.get_batch(indices=np.arange(s, e))
                dense_input = np.ascontiguousarray(batch["dense"], dtype=np.float32).reshape(batch_size, -1, 1, 1)
                sparse_input = np.ascontiguousarray(np.hstack(batch["sparse"]), dtype=np.int32).reshape(batch_size, -1)

                yield {
                    'numerical_input': dense_input,
                    'sparse_input': sparse_input
                }

        calibrator = Calibrator(data_loader=data_loader(), cache=cache_file, batch_size=batch_size)
        load_engine = EngineFromNetwork(create_network, config=CreateConfig(int8=True, calibrator=calibrator))

        # run calibration
        with G_LOGGER.verbosity(G_LOGGER.VERBOSE), TrtRunner(load_engine) as _:
            logging.info(f'Calibration completed, cache written to: {cache_file}')
            pass
