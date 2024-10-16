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

from nvmitten.constants import Precision
from nvmitten.nvidia.builder import ONNXNetwork

from code.common import logging, dict_get
from code.common.fields import Fields
from code.retinanet.tensorrt.onnx_generator.anchor_generator import AnchorGenerator


__doc__ = """Scripts for modifying Retinanet onnx graphs
"""
Subnet = ONNXNetwork.Subnetwork
TensorDesc = ONNXNetwork.Tensor


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


class RetinanetGraphSurgeon(ONNXNetwork):
    """
    The class will take the output of the torch2onnx graph from MLPerf inference repo,
    and apply the changes listed in process_onnx().
    """

    subnetwork_map = {
        "dla": Subnet(outputs=[TensorDesc("classification_head_100x100",
                                          (1, 2376, 100, 100)),
                               TensorDesc("classification_head_50x50",
                                          (1, 2376, 50, 50)),
                               TensorDesc("classification_head_25x25",
                                          (1, 2376, 25, 25)),
                               TensorDesc("classification_head_13x13",
                                          (1, 2376, 13, 13)),
                               TensorDesc("classification_head_7x7",
                                          (1, 2376, 7, 7)),
                               TensorDesc("regression_head_100x100",
                                          (1, 36, 100, 100)),
                               TensorDesc("regression_head_50x50",
                                          (1, 36, 50, 50)),
                               TensorDesc("regression_head_25x25",
                                          (1, 36, 25, 25)),
                               TensorDesc("regression_head_13x13",
                                          (1, 36, 13, 13)),
                               TensorDesc("regression_head_7x7",
                                          (1, 36, 7, 7)),
                               ]),
        # NMS PVA plugin is static and requires batch size 1
        "nmspva": Subnet(inputs=[TensorDesc("classification_head_100x100",
                                            (1, 2376, 100, 128)),
                                 TensorDesc("classification_head_50x50",
                                            (1, 2376, 50, 64)),
                                 TensorDesc("classification_head_25x25",
                                            (1, 2376, 25, 64)),
                                 TensorDesc("classification_head_13x13",
                                            (1, 2376, 13, 64)),
                                 TensorDesc("classification_head_7x7",
                                            (1, 2376, 7, 64)),
                                 TensorDesc("regression_head_100x100",
                                            (1, 36, 100, 128)),
                                 TensorDesc("regression_head_50x50",
                                            (1, 36, 50, 64)),
                                 TensorDesc("regression_head_25x25",
                                            (1, 36, 25, 64)),
                                 TensorDesc("regression_head_13x13",
                                            (1, 36, 13, 64)),
                                 TensorDesc("regression_head_7x7",
                                            (1, 36, 7, 64)),
                                 ]),
    }
    """RetinaNet subnetwork definition:

    DLA - NMS partition
                                                                __
                          images                                  |
                            |                                     |
                           ...                                    |  -- DLA subnetwork (can fully run on DLA)
                            |                                     |
    classification_head_7x7...    regression_head_100x100       __|
                            |                                     |
                       nmsopt_output                            __|  -- NMS subnetwork
    """

    # Constants for generating default anchor boxes for retinanet
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    def __init__(self,
                 onnx_path,
                 precision,
                 cache_file,
                 compute_sm,
                 device_type,
                 need_calibration,
                 disable_beta1_smallk=False,
                 nms_type="nms_opt"):
        no_fuse = (device_type != 'gpu') or (need_calibration)
        super().__init__(onnx_path,
                         precision,
                         calib_cache_path=cache_file,
                         compute_sm=compute_sm,
                         op_name_remap=dict(),  # We overwrite the rename_nodes method
                         no_fusions=no_fuse)
        self.device_type = device_type
        self.disable_beta1_smallk = disable_beta1_smallk
        self.nms_type = nms_type

        # Create the anchor generator instance for later use
        self.anchor_generator = AnchorGenerator(
            self.anchor_sizes, self.aspect_ratios
        )

    def rename_nodes(self, op_name_remap):
        """
        The backbone has 5 resNeXt blocks.
        For blocks starting res2a, the number corresponds to the starting layer
        for branch2a, branch2b, branch2c, (optional) branch1, residual.
        - branch2a/b/c has conv+mul+add+relu layers
        - branch1 has conv+mul+add layers
        - the residual has add+relu layers


        Args:
            op_name_remap (Dict[str, str]): This argument is ignored, but is here to adhere to the API.
        """
        logging.info("Renaming layers...")

        backbone_rename_map = {
            # Res1
            "/backbone/body/conv1/Conv": "res1_conv",
            "/backbone/body/bn1/Mul": "res1_scale",
            "/backbone/body/bn1/Add": "res1_bias",
            "/backbone/body/relu/Relu": "res1_relu",
            "/backbone/body/maxpool/MaxPool": "res1_maxpool"
        }
        resblock_layer_map = {
            'res2a': "layer1/layer1.0/",
            'res2b': "layer1/layer1.1/",
            'res2c': "layer1/layer1.2/",

            'res3a': "layer2/layer2.0/",
            'res3b': "layer2/layer2.1/",
            'res3c': "layer2/layer2.2/",
            'res3d': "layer2/layer2.3/",

            'res4a': "layer3/layer3.0/",
            'res4b': "layer3/layer3.1/",
            'res4c': "layer3/layer3.2/",
            'res4d': "layer3/layer3.3/",
            'res4e': "layer3/layer3.4/",
            'res4f': "layer3/layer3.5/",

            'res5a': "layer4/layer4.0/",
            'res5b': "layer4/layer4.1/",
            'res5c': "layer4/layer4.2/",
        }

        backbone_prefix = "/backbone/body/"

        # Construct the backbone_rename_map from the layer map
        for block, layer in resblock_layer_map.items():
            dic = {
                # 2a
                f"{backbone_prefix}{layer}conv1/Conv": f"{block}_branch2a_conv",
                f"{backbone_prefix}{layer}bn1/Mul": f"{block}_branch2a_scale",
                f"{backbone_prefix}{layer}bn1/Add": f"{block}_branch2a_bias",
                f"{backbone_prefix}{layer}relu/Relu": f"{block}_branch2a_relu",

                # 2b
                f"{backbone_prefix}{layer}conv2/Conv": f"{block}_branch2b_conv",
                f"{backbone_prefix}{layer}bn2/Mul": f"{block}_branch2b_scale",
                f"{backbone_prefix}{layer}bn2/Add": f"{block}_branch2b_bias",
                f"{backbone_prefix}{layer}relu_1/Relu": f"{block}_branch2b_relu",

                # 2c
                f"{backbone_prefix}{layer}conv3/Conv": f"{block}_branch2c_conv",
                f"{backbone_prefix}{layer}bn3/Mul": f"{block}_branch2c_scale",
                f"{backbone_prefix}{layer}bn3/Add": f"{block}_branch2c_bias",

                # Residual connection
                f"{backbone_prefix}{layer}Add": f"{block}_residual_add",
                f"{backbone_prefix}{layer}relu_2/Relu": f"{block}_relu",
            }

            # res-a layers, which has the branch1
            if ".0" in layer:
                br1 = {
                    # 1
                    f"{backbone_prefix}{layer}downsample/downsample.0/Conv": f"{block}_branch1_conv",
                    f"{backbone_prefix}{layer}downsample/downsample.1/Mul": f"{block}_branch1_scale",
                    f"{backbone_prefix}{layer}downsample/downsample.1/Add": f"{block}_branch1_bias",
                }
                dic.update(br1)

            backbone_rename_map.update(dic)

        count = 0
        for node in self.graph.nodes:
            if node.name in backbone_rename_map:
                new_name = backbone_rename_map[node.name]
                logging.debug("Renaming layer: {} -> {}".format(node.name, new_name))
                node.name = new_name
                count += 1
        logging.info(f"Renamed {count} layers.")

        """
        Update tensor name to be consistent to its producer op
        TODO: for now, only backbone resnext layers are renamed
        """
        logging.info("Renaming tensors to match layer names")
        for node in self.graph.nodes:
            if 'regression' not in node.name and 'res' in node.name:
                for t_idx, out_tensor in enumerate(node.outputs):
                    if not out_tensor.name or node.name not in out_tensor.name:
                        logging.debug("Naming tensor: {} -- {}_out_{}".format(node.name, node.name, t_idx))
                        out_tensor.name = "{}_out_{}".format(node.name, t_idx)
                # Rename the constant scale/bias tensor.
                if 'scale' in node.name or 'bias' in node.name:
                    for input_tensor in node.inputs:
                        if 'out' not in input_tensor.name:
                            tensor_name = f"{node.name}_value"
                            input_tensor.name = tensor_name

        assert len(self.graph.inputs) == 1, "only one input is expected: {}".format(self.graph.inputs)
        graph_input = self.graph.inputs[0]

    def prefusion(self):
        """
        Append the Non-Maximum Suppression (NMS) layer to the conv heads
        """
        logging.info(f"Adding NMS layer {self.nms_type} to the graph...")
        if self.nms_type == 'none':
            return
        elif self.nms_type == 'efficientnms':
            self.add_efficientnms()
        elif self.nms_type == 'nmsopt':
            self.add_nmsopt()
        elif self.nms_type == 'nmspva':
            self.add_nmspva()
        else:
            raise NotImplementedError(f"No such nms {self.nms_type}, exiting...")

    def add_efficientnms(self):
        """
        Add the open-sourced efficientNMS as documented in
        https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin

        Note the efficientNMS needs to be followed by a RetinaNet Concat Plugin, which
        arranges the output in the right order and concat into 1 tensor.
        """
        tensors = self.graph.tensors()
        # EfficientNMS use xywh-format anchor boxes, scaled down to [0,1]
        np_anchor_xywh_scaled = self.anchor_generator(scale_retinanet=True, order="xywh")[0].detach().cpu().numpy()
        anchor = np.expand_dims(np_anchor_xywh_scaled, axis=0)
        anchor_tensor = gs.Constant(name="anchor", values=anchor)

        op = 'EfficientNMS_TRT'
        node_name = 'efficientNMS'

        # Populate the plugin fields
        node_attrs = {
            "background_class": -1,
            "score_threshold": 0.05,
            "iou_threshold": 0.5,
            "max_output_boxes": 1000,
            "score_activation": True,
            "box_coding": 1,
        }
        attrs = {
            "plugin_version": "1",
            "plugin_namespace": "",
        }
        attrs.update(node_attrs)

        # Create ouptut tensors for the EfficientNMS
        num_detections = gs.Variable(name="num_detections",
                                     dtype=np.int32,
                                     shape=["batch_size", 1])
        detection_boxes = gs.Variable(name="detection_boxes",
                                      dtype=np.float32,
                                      shape=["batch_size", 1000, 4])
        detection_scores = gs.Variable(name="detection_scores",
                                       dtype=np.float32,
                                       shape=["batch_size", 1000])
        detection_classes = gs.Variable(name="detection_classes",
                                        dtype=np.int32,
                                        shape=["batch_size", 1000])

        nms_inputs = [tensors["bbox_regression"], tensors["cls_logits"], anchor_tensor]
        nms_outputs = [num_detections, detection_boxes, detection_scores, detection_classes]

        self.graph.layer(op="EfficientNMS_TRT",
                         name="EfficientNMS",
                         inputs=nms_inputs,
                         outputs=nms_outputs,
                         attrs=attrs)

        # Add Retinanet concat plugin
        concat_final_output = gs.Variable(name="concat_final_output",
                                          dtype=np.float32,
                                          shape=["batch_size", 7001])
        attrs = {
            "plugin_version": "1",
            "plugin_namespace": "",
        }
        self.graph.layer(op="RetinanetConcatNmsOutputsPlugin",
                         name="RetinanetConcatNmsOutputsPlugin",
                         inputs=[num_detections, detection_boxes, detection_scores, detection_classes],
                         outputs=[concat_final_output],
                         attrs=attrs)
        self.graph.outputs = [concat_final_output]

        self.cleanup_graph()

    def add_nmsopt(self):
        """
        Add the optimized NMS implementation from the MLPerf repo.
        """
        tensors = self.graph.tensors()
        # NMSOPT uses ltrb-format anchor boxes, scaled down to [0,1]
        np_anchor_ltrb_scaled = self.anchor_generator(scale_retinanet=True, order="ltrb")[0].detach().cpu().numpy()
        anchor = np.expand_dims(np_anchor_ltrb_scaled, axis=0)
        anchor = np.reshape(anchor, (1, -1, 1))
        anchor_tensor = gs.Constant(name="anchor", values=anchor)

        op = 'NMS_OPT_TRT'
        node_name = 'nmsopt'

        node_attrs = {
            "shareLocation": True,
            "varianceEncodedInTarget": True,  # RetinaNet variance is all 1
            "backgroundLabelId": -1,
            "numClasses": 264,
            "topK": 1000,
            "keepTopK": 1000,
            "confidenceThreshold": 0.05,
            "nmsThreshold": 0.5,
            "inputOrder": [0, 5, 10],
            "confSigmoid": True,
            "confSoftmax": False,
            "isNormalized": True,
            "codeType": 1,
            "numLayers": 5,
            "supportsInt8ConfInputs": True,
        }
        attrs = {
            "plugin_version": "2",
            "plugin_namespace": "",
        }
        attrs.update(node_attrs)

        # Add nmsopt layer
        nms_output = gs.Variable(name="nmsopt_output",
                                 dtype=np.float32,
                                 shape=["batch_size", 7001])

        # Feature size from small to large
        feature_map_sizes = [100, 50, 25, 13, 7]
        conv_loc_outputs = [tensors[f"regression_head_{size}x{size}"]
                            for size in feature_map_sizes]
        conv_conf_outputs = [tensors[f"classification_head_{size}x{size}"]
                             for size in feature_map_sizes]

        nms_inputs = conv_loc_outputs + conv_conf_outputs + [anchor_tensor]
        nms_outputs = [nms_output]

        self.graph.layer(op="NMS_OPT_TRT",
                         name="nmsopt",
                         inputs=nms_inputs,
                         outputs=nms_outputs,
                         attrs=attrs)

        self.graph.outputs = [nms_output]
        self.graph.cleanup().toposort()

    def add_nmspva(self):
        """
        Add the optimized NMS implementation from the MLPerf repo.
        """
        assert self.device_type == "dla", "PVA NMS should only be used by DLA engines. exiting..."

        tensors = self.graph.tensors()
        logging.info("Using xywh anchor")
        # NMSPVA uses ltrb-format anchor boxes, scaled down to [0,1]
        np_anchor_xywh_scaled = self.anchor_generator(scale_retinanet=True, order="xywh")[0].detach().cpu().numpy()
        anchor = np.expand_dims(np_anchor_xywh_scaled, axis=0)
        anchor_tensor = gs.Constant(name="anchor", values=anchor)

        node_op = 'RetinaNetNMSPVATRT'
        node_name = 'RetinaNetNMSPVA'

        node_attrs = {
            "nmsThreshold": 0.50,
            "score0Thresh": 0.11,
            "score1Thresh": 0.06,
            "score2Thresh": 0.06,
            "score3Thresh": 0.05,
            "score4Thresh": 0.05
        }

        attrs = {
            "plugin_version": "1",
            "plugin_namespace": "",
        }
        attrs.update(node_attrs)

        # NMS input
        logging.warning("PVA plugin only supports BS=1 and static input tensors. Enforcing tensor shapes!")
        enforce_shape = True
        if enforce_shape:
            for tensor in tensors.values():
                if tensor.shape is not None and not isinstance(tensor, gs.Constant):
                    tensor.shape[0] = 1
        feature_map_sizes = [100, 50, 25, 13, 7]
        regression_channel = 36
        score_channel = 2376
        conv_loc_outputs = []
        conv_conf_outputs = []
        for size in feature_map_sizes:
            regression_head = tensors[f"regression_head_{size}x{size}"]
            regression_head.shape[1] = regression_channel
            regression_head.shape[2] = size
            regression_head.shape[3] = size
            classification_head = tensors[f"classification_head_{size}x{size}"]
            classification_head.shape[1] = score_channel
            classification_head.shape[2] = size
            classification_head.shape[3] = size

            conv_loc_outputs.append(regression_head)
            conv_conf_outputs.append(classification_head)

        # NMS output
        if enforce_shape:
            batch_size = 1
        else:
            batch_size = "batch_size"
        nms_output = gs.Variable(name="nmsopt_output",
                                 dtype=np.float32,
                                 shape=[batch_size, 1, 1, 7001])

        # PVA plugin requires classfication heads first then regression heads input order
        nms_inputs = conv_conf_outputs + conv_loc_outputs + [anchor_tensor]
        nms_outputs = [nms_output]

        # Add nms pva layer
        self.graph.layer(op=node_op,
                         name=node_name,
                         inputs=nms_inputs,
                         outputs=nms_outputs,
                         attrs=attrs)

        self.graph.outputs = [nms_output]
        self.graph.cleanup().toposort()

    def fuse_beta1_conv(self):
        """
        Fuse all conv+scale+bias+beta+relu layers for backbone layers branch2c
        """
        if self.disable_beta1_smallk:
            return

        logging.info("Replacing all branch2c beta=1 conv with smallk kernel.")
        plugin_op_name = "SmallTileGEMM_TRT"

        # Check the dynamic range map exists
        assert self.dyn_range_map != {}, "The calibration cache has to exist. exiting..."

        beta1_op_list = [
            'res2a', 'res2b', 'res2c',
            'res3a', 'res3b', 'res3c', 'res3d',
            'res4a', 'res4b', 'res4c', 'res4d', 'res4e', 'res4f',
            # 'res5a', 'res5b', 'res5c', # C = 1024 is not supported for beta=1
        ]

        for op in beta1_op_list:
            plugin_layer_name = f"{op}_conv_residual_relu_smallk"
            op_dict = dict()
            [op_dict.update({_n.name: _n}) for _n in self.graph.nodes if op in _n.name]

            conv_op_name = f"{op}_branch2c_conv"
            scale_op_name = f"{op}_branch2c_scale"
            bias_op_name = f"{op}_branch2c_bias"
            residual_add_op_name = f"{op}_residual_add"
            final_relu_op_name = f"{op}_relu"

            plugin_input = [op_dict[conv_op_name].inputs[0],
                            op_dict[residual_add_op_name].inputs[1]]
            plugin_output = [op_dict[final_relu_op_name].outputs[0]]

            # Get kernel parameters and weights/scale/bias
            K = op_dict[conv_op_name].inputs[1].shape[0]
            C = op_dict[conv_op_name].inputs[1].shape[1]
            weight = op_dict[conv_op_name].inputs[1]
            scale = op_dict[scale_op_name].inputs[1]
            bias = op_dict[bias_op_name].inputs[1]
            rescale = gs.Constant("rescale", values=np.ones((K), dtype=np.float32))

            # Dynamic ranges for input/output/residual (Ti/To/Tr)
            dyn_list = [
                self.dyn_range_map[op_dict[conv_op_name].inputs[0].name],
                self.dyn_range_map[op_dict[final_relu_op_name].outputs[0].name],
                self.dyn_range_map[op_dict[residual_add_op_name].inputs[1].name],
            ]
            dynamic_ranges = np.array(dyn_list, dtype=np.float32)
            dyn_const = gs.Constant("{}_dynamic_ranges".format(plugin_layer_name),
                                    values=dynamic_ranges)

            plugin_field_dict = {
                "inputChannels": gs.Constant("C", values=np.array([C], dtype=np.int32)),
                "filterDimR": gs.Constant("R", values=np.array([1], dtype=np.int32)),
                "filterDimS": gs.Constant("S", values=np.array([1], dtype=np.int32)),
                "weight": weight,
                "bias": bias,
                "scale": scale,
                "rescale": rescale,
                "dynamicRanges": dyn_const,
                "epilogueScaleBiasBetaRelu": gs.Constant("epilogue_sbbr", values=np.array([1], dtype=np.int32)),
            }

            attrs = {
                "plugin_version": "1",
                "plugin_namespace": "",
            }
            attrs.update(plugin_field_dict)

            # replace ops with plugin
            logging.info(f"Fusing {plugin_layer_name} with smallk...")
            self.graph.BETA1SmallKPlugin(plugin_op_name, plugin_layer_name, plugin_input,
                                         plugin_output, attrs)

        # graph cleanup
        self.cleanup_graph()

        # done
        logging.info("Plugin {} created successful for res2/3/4/5 branch2c".format(plugin_op_name))

    def fuse_ops(self):
        available_fusions_by_SM = {
            # beta=1 smallk is officially supported through Cask since rel/5.3
            # TODO: [CFK-9235] to pick up sm90 beta=1 smallk
            '80': [],
            '87': [self.fuse_beta1_conv],
            '89': [],
            '90': [],
        }

        sm_id = str(int(self.compute_sm))
        if self.precision != Precision.INT8 or sm_id not in available_fusions_by_SM:
            return

        fusions = available_fusions_by_SM[sm_id]
        for fusion in fusions:
            fusion()


def parse_args():
    """
    Arguments that can be used for standalone run
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx-fpath',
                        type=str,
                        default='build/models/retinanet-resnext50-32x4d/submission/retinanet_resnext50_32x4d_efficientNMS.800x800.onnx',
                        help='Input ONNX file for ResNet50')
    parser.add_argument('--output-onnx-fpath',
                        type=str,
                        default='/tmp/retinanet_graphsurgeon.onnx',
                        help='Output ONNX filename')
    parser.add_argument('--calibration-cache-fpath',
                        type=str,
                        default='code/retinanet/tensorrt/calibrator.cache',
                        help='Calibration cache file')
    parser.add_argument('--compute-sm',
                        type=str,
                        required=True,
                        help='GPU Architecture of choice',
                        choices={'80', '86', '87', '89', '90'})
    parser.add_argument('--precision',
                        type=str,
                        default='int8',
                        choices={'int8', 'fp16', 'fp32'},
                        help='Compute precision')
    parser.add_argument('--is-dla',
                        default=False,
                        action='store_true',
                        help='Device is DLA')
    parser.add_argument('--nms-type',
                        default='efficientnms',
                        choices={'efficientnms', 'nmsopt', 'nmspva', 'none'},
                        help='which type of nms to use.')
    parser.add_argument('--need-calibration',
                        default=False,
                        action='store_true',
                        help='In case calibration is required; do not fuse for example')

    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            logging.debug("Parsed args -- {}: {}".format(key, value))

    return args


def main(args):
    """
    commandline entrance of the graphsurgeon. Example commands:
        python3 -m code.retinanet.tensorrt.retinanet_graphsurgeon --compute-sm=90 --output-onnx-fpath=/home/scratch.zhihanj_sw/temp/models/retinanet_graphsurgeon_beta1.onnx --nms-type=efficientnms
    """
    device_type = 'dla' if args.is_dla else 'gpu'
    retinanet_gs = RetinanetGraphSurgeon(args.onnx_fpath,
                                         args.precision,
                                         args.calibration_cache_fpath,
                                         args.compute_sm,
                                         device_type,
                                         args.need_calibration,
                                         nms_type=args.nms_type)
    model = retinanet_gs.create_onnx_model()
    onnx.save(model, args.output_onnx_fpath)


if __name__ == '__main__':
    args = parse_args()
    main(args)
