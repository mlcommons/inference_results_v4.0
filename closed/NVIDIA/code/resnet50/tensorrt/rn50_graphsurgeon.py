#! /usr/bin/env python3

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


from collections import namedtuple
from pathlib import Path

import argparse
import json
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import os

from nvmitten.constants import Precision
from nvmitten.nvidia.builder import ONNXNetwork

from code.common import logging


# Convenience shorthands for internal classes
Subnet = ONNXNetwork.Subnetwork
TensorDesc = ONNXNetwork.Tensor


@gs.Graph.register()
def AveragePool(self, name, in_t, attrs):
    """
    Create and add AveragePool op to Graph
    """
    out_t = self.layer(op="AveragePool", name=name,
                       inputs=[in_t], outputs=[name],
                       attrs=attrs)[0]
    out_t.name = "{}_out_0".format(name)
    return out_t


@gs.Graph.register()
def Reshape(self, name, in_t, shape, attrs=dict()):
    """
    Create and add Reshape op to Graph
    """
    out_t = self.layer(op="Reshape", name=name,
                       inputs=[in_t, shape], outputs=[name],
                       attrs=attrs)[0]
    out_t.name = "{}_out_0".format(name)
    return out_t


@gs.Graph.register()
def MatMul(self, name, A, B, attrs=dict()):
    """
    Create and add MatMul op to Graph
    """
    out_t = self.layer(op="MatMul", name=name,
                       inputs=[A, B], outputs=[name],
                       attrs=attrs)[0]
    out_t.name = "{}_out_0".format(name)
    return out_t


@gs.Graph.register()
def Conv(self, name, A, K, attrs):
    """
    Create and add Conv op to Graph
    """
    out_t = self.layer(op="Conv", name=name,
                       inputs=[A, K], outputs=[name],
                       attrs=attrs)[0]
    out_t.name = "{}_out_0".format(name)
    return out_t


@gs.Graph.register()
def TopK(self, name, in_t, attrs):
    """
    Create and add TopK op to Graph
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK
    """
    out_t = self.layer(op="TopK", name=name,
                       inputs=[in_t], outputs=["{}_value".format(name),
                                               "{}_index".format(name)],
                       attrs=attrs)
    out_t[0].name = "{}_output_value".format(name)
    out_t[1].name = "{}_output_index".format(name)
    out_t[0].dtype = np.float32
    out_t[1].dtype = np.int64
    return out_t


@gs.Graph.register()
def Cast(self, name, in_t, attrs):
    """
    Create and add Cast op to Graph
    """
    out_t = self.layer(op="Cast", name=name,
                       inputs=[in_t], outputs=[name],
                       attrs=attrs)[0]
    out_t.name = "{}_out_0".format(name)
    out_t.dtype = np.int32
    return out_t


@gs.Graph.register()
def RES2PLUGIN(self, plugin_op, plugin_name, plugin_in, plugin_out, attrs):
    """
    Create PLUGIN made for RES2; set plugin type by providing plugin_op
    """
    # NOTE: do NOT clear input tensor's output
    for _o in plugin_out:
        _o.inputs.clear()
    return self.layer(op=plugin_op, name=plugin_name,
                      inputs=plugin_in, outputs=plugin_out,
                      attrs=attrs)[0]


@gs.Graph.register()
def BETA1SmallKPlugin(self, plugin_op, plugin_name, plugin_in, plugin_out, attrs):
    """
    Create a plugin layer for beta=1 smallk.
    """
    for _o in plugin_out:
        _o.inputs.clear()
    return self.layer(op=plugin_op, name=plugin_name,
                      inputs=plugin_in, outputs=plugin_out,
                      attrs=attrs)[0]


class RN50GraphSurgeon(ONNXNetwork):
    """
    Using ONNX Graph Surgeon, this class updates the ResNet50 ONNX graph for:
    1. Op and Tensor names
    2. Endpoint of RN50 to be more lightweight
    3. Fuse ops
    4. Set dynamic range of tensors if with quantization from calibration results
    """

    subnetwork_map = {
        "dla": Subnet(outputs=[TensorDesc("fc_replaced_out_0",
                                          (gs.Tensor.DYNAMIC, 1000, 1, 1))]),
        "topk": Subnet(inputs=[TensorDesc("fc_replaced_out_0",
                                          (gs.Tensor.DYNAMIC, 1000, 1, 1))]),
        "preres2": Subnet(outputs=[TensorDesc("pool1_out_0",
                                   (gs.Tensor.DYNAMIC, 64, 56, 56))]),
        "preres3": Subnet(outputs=[TensorDesc("res2c_relu_out_0",
                                   (gs.Tensor.DYNAMIC, 256, 56, 56))]),
        "res2_3": Subnet(inputs=[TensorDesc("pool1_out_0",
                                            (gs.Tensor.DYNAMIC, 64, 56, 56))],
                         outputs=[TensorDesc("res3d_relu_out_0",
                                             (gs.Tensor.DYNAMIC, 512, 28, 28))]),
        "res3": Subnet(inputs=[TensorDesc("res2c_relu_out_0",
                                          (gs.Tensor.DYNAMIC, 256, 56, 56))],
                       outputs=[TensorDesc("res3d_relu_out_0",
                                           (gs.Tensor.DYNAMIC, 512, 28, 28))]),
        "postres3": Subnet(inputs=[TensorDesc("res3d_relu_out_0",
                                              (gs.Tensor.DYNAMIC, 512, 28, 28))]),
        "res3end": Subnet(inputs=[TensorDesc("res2c_relu_out_0",
                                             (gs.Tensor.DYNAMIC, 256, 56, 56))]),
    }
    """RN50 subnetwork definition:

    DLA - topK partition
                   __
       conv1         |
         |           |
        ...          |  -- DLA subnetwork (can fully run on DLA)
         |           |
    fc_replaced    __|
         |           |
     topk_layer    __|  -- topK subnetwork

    ==============================================================
    PreRes3 - Res3 - PostRes3 Partition
                                 __
       conv1                       |
         |                         |
        ...                        |  -- PreRes3 subnetwork
         |                         |
    RES2_FULL_FUSION             __|
         |                         |
        ...                        |
         |                         |  -- Res3 subnetwork
    SmallTileGEMM_TRT_res3d_       |
    branch2c_conv_residual_relu  __|
         |                         |
        ...                        |  -- PostRes3 subnetwork
         |                         |
     topk_layer                  __|

    ==============================================================
    PreRes2 - Res2_3 - PostRes3 Partition
                                 __
       conv1                       |
         |                         |
        ...                        |  -- PreRes2 subnetwork
         |                         |
       pool1                     __|
         |                         |
        ...                        |
         |                         |  -- Res2_3 subnetwork
    SmallTileGEMM_TRT_res3d_       |
    branch2c_conv_residual_relu  __|
         |                         |
        ...                        |  -- PostRes3 subnetwork
         |                         |
     topk_layer                  __|
    """

    def __init__(self,
                 onnx_path,
                 precision,
                 cache_file,
                 compute_sm,
                 device_type,
                 need_calibration,
                 disable_beta1_smallk=False):
        with (Path(__file__).parent.resolve() / "onnx_node_names.json").open('r') as f:
            op_name_remap = json.load(f)
        no_fuse = (device_type != 'gpu') or (need_calibration)
        super().__init__(onnx_path,
                         precision,
                         calib_cache_path=cache_file,
                         compute_sm=compute_sm,
                         op_name_remap=op_name_remap,
                         no_fusions=no_fuse)
        self.disable_beta1_smallk = disable_beta1_smallk

    def fuse_ops(self):
        Res2Mega = self.fuse_res2_mega
        Beta1Smallk = self.fuse_beta1_conv

        available_fusions_by_SM = {
            '75': [Res2Mega],
            '80': [Res2Mega, Beta1Smallk],
            '86': [Res2Mega, Beta1Smallk],
            '87': [Res2Mega, Beta1Smallk],
            '89': [Res2Mega, Beta1Smallk],
            '90': [Res2Mega, Beta1Smallk],
        }
        sm_id = str(int(self.compute_sm))
        if self.precision != Precision.INT8 or sm_id not in available_fusions_by_SM:
            return

        fusions = available_fusions_by_SM[sm_id]
        for fusion in fusions:
            fusion()

    def prefusion(self):
        """
        Handle op touchup
        NOTE: different precision may lead to different op mix

        Some notes for what this function does, as of 09/25/2020:

        Original ONNX graph has a tail with Squeeze-and-Excitation block,
        followed by SoftMax/ArgMax for classification.

        Suspecting the performance of kernel as a reason, we are changing
        this sub-graph (after ReLU*), with more lightweight Pooling-FC-TopK
        sub-graph. As per kernel performance, w/ INT8, Conv1x1x1 replaces FC

            [BEFORE]                           [AFTER]
              Add                                Add
               |                                  |
              ReLU*                              ReLU*
               |                                  |
            ReduceMean                          AvgPool
               |                ==>               |
             Reshape                            FC/Conv
               |                                  |
             Squeeze                             TopK
               |                                  |
             MatMul                          +----+-----+
               |                             |          |
              Add                          Value       Index
               |                                        |
            Identity                                   Cast
               |
          +----+----+
          |         |
        SoftMax   ArgMax
          |         |
        Identity  Identity
          |         |
         Prob      Index

        In order to realize this in a modular way, entry in the op_touchup_map
        is selected, based on the condition (i.e. if precision is INT8 or not)
        and the series of calls are made to add new sub-graph after ReLU*.
        After adding the proper combination of ops (add_* calls), original sub-graph
        is removed by de-registering outputs from graph and clean up the graph.

        NOTE: the above is not necessarily what this touchup_ops is limited to achieve;
              if the graph is in need to be manipulated in a different way, it can be
              mapped from here to op_touchup_map, which defines needed calls in order.
        """
        fc_impl = self.add_fc
        if self.precision == Precision.INT8:
            # Replace FC layer with 1x1 conv
            fc_impl = self.add_conv

        policy = []
        if not str(os.environ.get('SKIP_POLICIES','')).lower() in ["yes", "true", "1"]:
            policy = [self.add_squeeze,
                  fc_impl,
                  self.add_topk,
                  self.add_cast,
                  self.remove_obsolete]
        
        for _f in policy:
            _f()

    def fuse_beta1_conv(self):
        """
        Fuse all conv+scale+bias+beta+relu layers after res2
        """
        if self.disable_beta1_smallk:
            return

        logging.info("Replacing all branch2c beta=1 conv with smallk kernel.")
        plugin_op_name = "SmallTileGEMM_TRT"

        BetaConvTuple = namedtuple("BetaConvTuple", "residual conv_in relu_out")
        op_names_list = [BetaConvTuple("res3a", "res3a_branch2c", "res3a_relu"),
                         BetaConvTuple("res3b", "res3b_branch2c", "res3b_relu"),
                         BetaConvTuple("res3c", "res3c_branch2c", "res3c_relu"),
                         BetaConvTuple("res3d", "res3d_branch2c", "res3d_relu"),
                         BetaConvTuple("res4a", "res4a_branch2c", "res4a_relu"),
                         BetaConvTuple("res4b", "res4b_branch2c", "res4b_relu"),
                         BetaConvTuple("res4c", "res4c_branch2c", "res4c_relu"),
                         BetaConvTuple("res4d", "res4d_branch2c", "res4d_relu"),
                         BetaConvTuple("res4e", "res4e_branch2c", "res4e_relu"),
                         BetaConvTuple("res4f", "res4f_branch2c", "res4f_relu"),
                         BetaConvTuple("res5a", "res5a_branch2c", "res5a_relu"),
                         BetaConvTuple("res5b", "res5b_branch2c", "res5b_relu"),
                         BetaConvTuple("res5c", "res5c_branch2c", "res5c_relu")]

        for op_names_tuple in op_names_list:
            plugin_layer_name = f"{plugin_op_name}_{op_names_tuple.conv_in}_conv_residual_relu"
            op_dict = dict()
            [op_dict.update({_n.name: _n}) for _n in self.graph.nodes if _n.name in op_names_tuple]
            op_list = [op_dict[_n] for _n in op_names_tuple]
            assert len(op_names_tuple) == len(op_list), "Need to capture all op objects in op_names_tuple"

            plugin_inp = [op_dict[op_names_tuple.conv_in].inputs[0], op_dict[op_names_tuple.residual].inputs[1]]
            plugin_out = [op_dict[op_names_tuple.relu_out].outputs[0]]

            # Create dummy input for scale and rescale (all ones)
            K = op_dict[op_names_tuple.conv_in].inputs[1].shape[0]
            C = op_dict[op_names_tuple.conv_in].inputs[1].shape[1]
            scale = gs.Constant("scale", values=np.ones((K), dtype=np.float32))
            rescale = gs.Constant("rescale", values=np.ones((K), dtype=np.float32))

            # Dynamic ranges for input/output/residual (Ti/To/Tr)
            dyn_list = [
                self.dyn_range_map[op_dict[op_names_tuple.conv_in].inputs[0].name],
                self.dyn_range_map[op_dict[op_names_tuple.relu_out].outputs[0].name],
                self.dyn_range_map[op_dict[op_names_tuple.residual].inputs[1].name],
            ]

            dynamic_ranges = np.array(dyn_list, dtype=np.float32)
            dyn_const = gs.Constant("{}_dynamic_ranges".format(plugin_layer_name),
                                    values=dynamic_ranges)

            plugin_field_dict = {
                "inputChannels": gs.Constant("C", values=np.array([C], dtype=np.int32)),
                "filterDimR": gs.Constant("R", values=np.array([1], dtype=np.int32)),
                "filterDimS": gs.Constant("S", values=np.array([1], dtype=np.int32)),
                "weight": op_dict[op_names_tuple.conv_in].inputs[1],
                "bias": op_dict[op_names_tuple.conv_in].inputs[2],
                "scale": scale,
                "rescale": rescale,
                "dynamicRanges": dyn_const,
                "epilogueScaleBiasBetaRelu": gs.Constant("epilogue_sbbr", values=np.array([1], dtype=np.int32)),
                # Dummy fields to supress warnings
                "fairShareCacheSize": gs.Constant("fairShareCacheSize", values=np.array([0], dtype=np.int32)),
            }

            attrs = {
                "plugin_version": "1",
                "plugin_namespace": "",
            }
            attrs.update(plugin_field_dict)

            # replace ops with plugin
            logging.info(f"Fusing {plugin_layer_name} with smallk...")
            self.graph.BETA1SmallKPlugin(plugin_op_name, plugin_layer_name, plugin_inp, plugin_out, attrs)

        # graph cleanup
        self.cleanup_graph()

        # done
        logging.info("Plugin {} fused successful for res3/4/5 branch2c".format(plugin_op_name))

    def fuse_res2_mega(self):
        """
        Search and replace all the res2 layers with the res2 megakernel plugin.
        This fusion is for mega fusion of entire res2a_*
        """
        logging.info("Fusing ops in res2_mega")
        op_names_list = ["res2a_branch1",
                         "res2a_branch2a", "res2a_branch2a_relu",
                         "res2a_branch2b", "res2a_branch2b_relu",
                         "res2a_branch2c", "res2a", "res2a_relu",
                         "res2b_branch2a", "res2b_branch2a_relu",
                         "res2b_branch2b", "res2b_branch2b_relu",
                         "res2b_branch2c", "res2b", "res2b_relu",
                         "res2c_branch2a", "res2c_branch2a_relu",
                         "res2c_branch2b", "res2c_branch2b_relu",
                         "res2c_branch2c", "res2c", "res2c_relu"]

        # setup plugin info
        plugin_name = "RES2_FULL_FUSION"

        # prep fusion: constants and attributes
        op_dict = dict()
        [op_dict.update({_n.name: _n}) for _n in self.graph.nodes if _n.name in op_names_list]
        op_list = [op_dict[_n] for _n in op_names_list]
        assert len(op_names_list) == len(op_list), "Need to capture all op objects in op_names_list"

        plugin_inp = [op_list[0].inputs[0]]
        plugin_out = [op_list[-1].outputs[0]]

        scale64 = gs.Constant("scale64", values=np.ones((64), dtype=np.float32))
        scale256 = gs.Constant("scale256", values=np.ones((256), dtype=np.float32))
        rescale = gs.Constant("rescale", values=np.ones((256), dtype=np.float32))

        # build array with dynamic ranges required for the fusion plugin
        # NOTE: order matters
        dyn_list = [
            self.dyn_range_map[plugin_inp[0].name],

            self.dyn_range_map[op_list[0].outputs[0].name],

            self.dyn_range_map[op_list[2].outputs[0].name],
            self.dyn_range_map[op_list[4].outputs[0].name],
            self.dyn_range_map[op_list[5].outputs[0].name],
            self.dyn_range_map[op_list[7].outputs[0].name],

            self.dyn_range_map[op_list[9].outputs[0].name],
            self.dyn_range_map[op_list[11].outputs[0].name],
            self.dyn_range_map[op_list[12].outputs[0].name],
            self.dyn_range_map[op_list[14].outputs[0].name],

            self.dyn_range_map[op_list[16].outputs[0].name],
            self.dyn_range_map[op_list[18].outputs[0].name],
            self.dyn_range_map[op_list[19].outputs[0].name],
            self.dyn_range_map[op_list[21].outputs[0].name],
        ]

        dynamic_ranges = np.array(dyn_list, dtype=np.float32)
        dyn_const = gs.Constant("{}_dynamic_ranges".format(plugin_name),
                                values=dynamic_ranges)

        # this becomes attributes to ONNX node that fusion plugin uses
        # NOTE: order does not matter
        plugin_field_dict = {
            "c_res2a_br1_w": op_list[0].inputs[1],
            "s_res2a_br1_s": scale256,
            "s_res2a_br1_b": op_list[0].inputs[2],

            "c_res2a_br2a_w": op_list[1].inputs[1],
            "s_res2a_br2a_s": scale64,
            "s_res2a_br2a_b": op_list[1].inputs[2],
            "c_res2a_br2b_w": op_list[3].inputs[1],
            "s_res2a_br2b_s": scale64,
            "s_res2a_br2b_b": op_list[3].inputs[2],
            "c_res2a_br2c_w": op_list[5].inputs[1],
            "s_res2a_br2c_s": scale256,
            "s_res2a_br2c_b": op_list[5].inputs[2],

            "c_res2b_br2a_w": op_list[8].inputs[1],
            "s_res2b_br2a_s": scale64,
            "s_res2b_br2a_b": op_list[8].inputs[2],
            "c_res2b_br2b_w": op_list[10].inputs[1],
            "s_res2b_br2b_s": scale64,
            "s_res2b_br2b_b": op_list[10].inputs[2],
            "c_res2b_br2c_w": op_list[12].inputs[1],
            "s_res2b_br2c_s": scale256,
            "s_res2b_br2c_b": op_list[12].inputs[2],

            "c_res2c_br2a_w": op_list[15].inputs[1],
            "s_res2c_br2a_s": scale64,
            "s_res2c_br2a_b": op_list[15].inputs[2],
            "c_res2c_br2b_w": op_list[17].inputs[1],
            "s_res2c_br2b_s": scale64,
            "s_res2c_br2b_b": op_list[17].inputs[2],
            "c_res2c_br2c_w": op_list[19].inputs[1],
            "s_res2c_br2c_s": scale256,
            "s_res2c_br2c_b": op_list[19].inputs[2],

            "r_res2a_br2c_r": rescale,
            "r_res2b_br2c_r": rescale,
            "r_res2c_br2c_r": rescale,

            "dynamic_ranges": dyn_const,
        }

        attrs = {
            "plugin_version": "1",
            "plugin_namespace": "",
        }
        attrs.update(plugin_field_dict)

        # replace ops with plugin
        self.graph.RES2PLUGIN("RnRes2FullFusion_TRT", plugin_name, plugin_inp, plugin_out, attrs)

        # graph cleanup
        self.cleanup_graph()

        # done
        logging.info("Plugin {} successful".format(plugin_name))

    def add_squeeze(self):
        """
        add new squeeze layer
        """
        logging.info("Adding Squeeze")
        # find input to squeeze to be added
        last_relu_op = [_n for _n in self.graph.nodes if _n.name == "res5c_relu"][0]
        # add AveragePool
        attrs = {
            "kernel_shape": [7, 7]
        }
        squeeze_replaced_out = self.graph.AveragePool("squeeze_replaced", last_relu_op.outputs[0], attrs)

    def add_fc(self):
        """
        add FC layer
        """
        logging.info("Adding FC layer")
        # fetch some attrs from old fc1000; note MatMul doesn't have bias
        old_fc_op = [_n for _n in self.graph.nodes if _n.name == "fc1000"][0]
        old_fc_kernel = old_fc_op.inputs[1]
        fc_kernel_weights = old_fc_kernel.values[:, 1:]
        # instantiate fc weight
        # NOTE: expects KM weight, if transpose is not set (default not set)
        fc_weight = gs.Constant("fc_replaced_weight", values=fc_kernel_weights)
        # find input to fc to be added
        squeeze_replaced_op = [_n for _n in self.graph.nodes if _n.name == "squeeze_replaced"][0]
        squeeze_replaced_out = squeeze_replaced_op.outputs[0]
        # reshape input
        reshape_shape = np.array([-1, fc_kernel_weights.shape[0]], dtype=np.int64)
        fc_reshape_shape = gs.Constant("fc_reshape_shape", values=reshape_shape)
        # add FC: Reshape=>MatMul
        fc_reshape_out = self.graph.Reshape("fc_reshape_input", squeeze_replaced_out, fc_reshape_shape)
        fc_out = self.graph.MatMul("fc_replaced", fc_reshape_out, fc_weight)

    def add_conv(self):
        """
        add Conv layer
        """
        logging.info("Adding Conv layer, instead of FC")
        # fetch some attrs from old fc1000; note MatMul doesn't have bias
        old_fc_op = [_n for _n in self.graph.nodes if _n.name == "fc1000"][0]
        old_fc_kernel = old_fc_op.inputs[1]
        # instantiate fc weight and attrs
        # NOTE: ONNX uses MCkHkW format
        fc_kernel_weights = old_fc_kernel.values.transpose()[1:, :].reshape(1000, 2048, 1, 1)
        fc_weight = gs.Constant("fc_replaced_weight", values=fc_kernel_weights)
        attrs = {
            "kernel_shape": [1, 1]
        }
        # find input to fc to be added
        squeeze_replaced_op = [_n for _n in self.graph.nodes if _n.name == "squeeze_replaced"][0]
        squeeze_replaced_out = squeeze_replaced_op.outputs[0]
        # add FC: Conv
        fc_out = self.graph.Conv("fc_replaced", squeeze_replaced_out, fc_weight, attrs)

    def add_topk(self):
        """
        add topk layer
        """
        logging.info("Adding TopK layer")
        # find input to topk to be added
        fc_op = [_n for _n in self.graph.nodes if _n.name == "fc_replaced"][0]
        fc_op_out = fc_op.outputs[0]
        # set attrs
        attrs = {
            "axis": 1,
            "k": 1,
            "largest": 1,
        }
        # add TopK
        topk_out_list = self.graph.TopK("topk_layer", fc_op_out, attrs)

    def add_cast(self):
        """
        add cast layer
        """
        logging.info("Adding Cast Layer")
        # cast topk_layer_output_index output from int64 to int32
        topk_op = [_n for _n in self.graph.nodes if _n.name == "topk_layer"][0]
        topk_index_out = topk_op.outputs[1]
        # override name
        topk_index_out.name = "topk_layer_output_index_i64"
        # set attrs
        attrs = {
            "to": getattr(onnx.TensorProto, "INT32"),
        }
        # add cast
        cast_out_list = self.graph.Cast("cast_layer", topk_index_out, attrs)
        # override name
        cast_out_list.name = "topk_layer_output_index"

    def remove_obsolete(self):
        """
        Remove obsolete layers
        """
        logging.info("Removing obsolete layers")
        topk_op = [_n for _n in self.graph.nodes if _n.name == "topk_layer"][0]
        cast_op = [_n for _n in self.graph.nodes if _n.name == "cast_layer"][0]
        self.graph.outputs = [topk_op.outputs[0]] + cast_op.outputs
        self.cleanup_graph()


def parse_args():
    """
    Arguments that can be used for standalone run
    """
    parser = argparse.ArgumentParser(description=RN50GraphSurgeon.__doc__)
    parser.add_argument('--onnx-fpath',
                        dest='onnx_path',
                        type=str,
                        default='build/models/ResNet50/resnet50_v1.onnx',
                        help='Input ONNX file for ResNet50')
    parser.add_argument('--output-onnx-fname',
                        dest='out_onnx',
                        type=str,
                        default='rn50_discharged.onnx',
                        help='Output ONNX filename')
    parser.add_argument('--calibration-cache-fpath',
                        dest='calibcache',
                        type=str,
                        default='code/resnet50/tensorrt/calibrator.cache',
                        help='Calibration cache file')
    parser.add_argument('--compute-sm',
                        dest='compute_sm',
                        type=str,
                        default='75',
                        choices={'Unknown',
                                 '75',  # Turing
                                 '80',  # Ampere (A100, A30)
                                 '86',  # Ampere (A40, A10, A16, A2)
                                 '87',  # Orin AGX
                                 '90',  # Hopper
                                 },
                        help='GPU Architecture of choice')
    parser.add_argument('--precision',
                        dest='precision',
                        type=str,
                        default='int8',
                        choices={'int8', 'fp16', 'fp32'},
                        help='Compute precision')
    parser.add_argument('--non-gpu',
                        dest='nongpu',
                        default=False,
                        action='store_true',
                        help='Device is not GPU, i.e. DLA')
    parser.add_argument('--need-calibration',
                        dest='need_cal',
                        default=False,
                        action='store_true',
                        help='In case calibration is required; do not fuse for example')

    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            logging.info("Parsed args -- {}: {}".format(key, value))

    return args


def main(args):
    """
    Standalone run manipulates input ONNX graph and returns updated ONNX graph
    How to run:
        in container: python3 -m code.resnet50.tensorrt.rn50_graphsurgeon --help
    """
    rn50gs = RN50GraphSurgeon(args.onnx_path,
                              args.precision,
                              args.calibcache,
                              args.compute_sm,
                              ('dla' if args.nongpu else 'gpu'),
                              args.need_cal)
    model = rn50gs.process_onnx()
    onnx.save(rn50gs.model, args.out_onnx)


if __name__ == '__main__':
    args = parse_args()
    main(args)
