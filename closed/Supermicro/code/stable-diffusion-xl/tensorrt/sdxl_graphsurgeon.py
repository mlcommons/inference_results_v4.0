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


import os
import re
import tempfile
import argparse
import numpy as np
import onnx
import onnx_graphsurgeon as gs

from pathlib import Path
from polygraphy.backend.onnx.loader import fold_constants
from nvmitten.constants import Precision
from nvmitten.nvidia.builder import ONNXNetwork

from code.common import logging


__doc__ = """Scripts for modifying SDXL onnx graphs
"""


class SDXLGraphSurgeon(ONNXNetwork):
    """
    The class is the base class to optimize onnx models converted from SDXL pytorch models.
    """

    # onnx threshold of using onnx.save_model instead of onnx.save
    ONNX_LARGE_FILE_THRESHOLD = 2 ** 31

    def __init__(self,
                 onnx_path,
                 precision,
                 device_type,
                 model_name,
                 add_hidden_states):
        super().__init__(onnx_path,
                         Precision.FP16,  # TODO yihengz: Overwrite SDXL precision to bypass calibration cache load because we use explicit quantized model, update after picking up mitten fix
                         op_name_remap=dict())  # No rename for SDXL
        self.device_type = device_type
        self.name = model_name
        self.add_hidden_states = add_hidden_states
        self.use_l2_5_fusions = precision == Precision.INT8 and self.name == "UNetXL"

    def info(self, prefix):
        logging.info(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=False)
        self.graph = gs.import_onnx(onnx_graph)

    def infer_shapes(self):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > SDXLGraphSurgeon.ONNX_LARGE_FILE_THRESHOLD:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                tmp_path.mkdir(exist_ok=True)
                onnx_orig_path = tmp_path / "model.onnx"
                onnx_inferred_path = tmp_path / "inferred.onnx"
                onnx.save_model(onnx_graph,
                                str(onnx_orig_path),
                                save_as_external_data=True,
                                all_tensors_to_one_file=True,
                                convert_attribute=False)
                onnx.shape_inference.infer_shapes_path(str(onnx_orig_path), str(onnx_inferred_path))
                onnx_graph = onnx.load(str(onnx_inferred_path))
        else:
            onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)
        self.graph = gs.import_onnx(onnx_graph)

    def clip_add_hidden_states(self):
        hidden_layers = -1
        onnx_graph = gs.export_onnx(self.graph)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                name = onnx_graph.graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(int(name.split(".")[1].split("/")[0]), hidden_layers)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                if onnx_graph.graph.node[i].output[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers - 1):
                    onnx_graph.graph.node[i].output[j] = "hidden_states"
            for j in range(len(onnx_graph.graph.node[i].input)):
                if onnx_graph.graph.node[i].input[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers - 1):
                    onnx_graph.graph.node[i].input[j] = "hidden_states"
        self.graph = gs.import_onnx(onnx_graph)

    def fuse_mha_qkv_int8_sq(self):
        tensors = self.graph.tensors()
        keys = tensors.keys()

        # mha  : fuse QKV QDQ nodes
        # mhca : fuse KV QDQ nodes
        q_pat = '/down_blocks.\d+/attentions.\d+/transformer_blocks.\d+/attn\d+/to_q/input_quantizer/DequantizeLinear_output_0'
        k_pat = '/down_blocks.\d+/attentions.\d+/transformer_blocks.\d+/attn\d+/to_k/input_quantizer/DequantizeLinear_output_0'
        v_pat = '/down_blocks.\d+/attentions.\d+/transformer_blocks.\d+/attn\d+/to_v/input_quantizer/DequantizeLinear_output_0'

        qs = list(sorted(map(lambda x: x.group(0), filter(lambda x: x is not None, [re.match(q_pat, key) for key in keys]))))
        ks = list(sorted(map(lambda x: x.group(0), filter(lambda x: x is not None, [re.match(k_pat, key) for key in keys]))))
        vs = list(sorted(map(lambda x: x.group(0), filter(lambda x: x is not None, [re.match(v_pat, key) for key in keys]))))

        removed = 0
        assert len(qs) == len(ks) == len(vs), 'Failed to collect tensors'
        for q, k, v in zip(qs, ks, vs):
            is_mha = all(['attn1' in tensor for tensor in [q, k, v]])
            is_mhca = all(['attn2' in tensor for tensor in [q, k, v]])
            assert (is_mha or is_mhca) and (not (is_mha and is_mhca))

            if is_mha:
                tensors[k].outputs[0].inputs[0] = tensors[q]
                tensors[v].outputs[0].inputs[0] = tensors[q]
                del tensors[k]
                del tensors[v]

                removed += 2

            else:  # is_mhca
                tensors[k].outputs[0].inputs[0] = tensors[v]
                del tensors[k]

                removed += 1

        return removed  # expected 72 for L2.5

    def remove_FC_int8_qdq(self):
        tensors = self.graph.tensors()
        keys = tensors.keys()
        nodes = {node.name: node for node in self.graph.nodes}

        # remove QDQ nodes from linear layers after MHA/MHCA
        A_pat = '/down_blocks.\d+/attentions.\d+/transformer_blocks.\d+/attn\d+/Reshape_7_output_0'
        B_pat = 'down_blocks.\d+.attentions.\d+.transformer_blocks.\d+.attn\d+.to_out.0.weight'
        target_pat = '/down_blocks.\d+/attentions.\d+/transformer_blocks.\d+/attn\d+/to_out.0/MatMul'

        As = list(map(lambda x: x.group(0), filter(lambda x: x is not None, [re.match(A_pat, key) for key in keys])))
        Bs = list(map(lambda x: x.group(0), filter(lambda x: x is not None, [re.match(B_pat, key) for key in keys])))
        targets = list(map(lambda x: x.group(0), filter(lambda x: x is not None, [re.match(target_pat, key) for key in keys])))

        removed = 0
        for A, B, target in zip(As, Bs, targets):
            target_node = nodes[target]

            A_, B_ = target_node.inputs
            del tensors[A_.name]
            del tensors[B_.name]

            removed += 2
            target_node.inputs = [tensors[A], tensors[B]]

        return removed  # expected 96 for L2.5

    def prefusion(self):
        """
        Append the Non-Maximum Suppression (NMS) layer to the conv heads
        """
        self.info(f'{self.name}: original')

        if self.use_l2_5_fusions:
            if (removed := self.fuse_mha_qkv_int8_sq()) > 0:
                self.info(f'{self.name}: removing {removed} mha qkv int8 sq nodes')

            if (removed := self.remove_FC_int8_qdq()) > 0:
                self.info(f'{self.name}: removing {removed} qdq nodes for FC after mha/mhca')

        self.cleanup_graph()
        self.info(f'{self.name}: cleanup')

        self.fold_constants()
        self.info(f'{self.name}: fold constants')

        self.infer_shapes()
        self.info(f'{self.name}: shape inference')

        if self.add_hidden_states:
            self.clip_add_hidden_states()
            self.info(f'{self.name}: added hidden_states')

        self.info(f'{self.name}: GS finished')


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
                        default='/tmp/sdxl_graphsurgeon.onnx',
                        help='Output ONNX filename')
    parser.add_argument('--precision',
                        type=str,
                        default='fp16',
                        choices={'fp16', 'fp32'},
                        help='Compute precision')

    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            logging.debug("Parsed args -- {}: {}".format(key, value))

    return args


def main(args):
    """
    commandline entrance of the graphsurgeon. Example commands:
        python3 -m code.stable-diffusion-xl.tensorrt.sdxl_graphsurgeon --onnx-fpath=build/models/SDXL/onnx_models/unetxl/model.onnx --output-onnx-fpath=/tmp/unetxl_graphsurgeon/model.onnx
    """
    device_type = 'gpu'
    sdxl_gs = SDXLGraphSurgeon(args.onnx_fpath,
                               args.precision,
                               device_type)
    model = sdxl_gs.create_onnx_model()
    os.makedirs(Path(args.output_onnx_fpath).parent, exist_ok=True)
    if model.ByteSize() > SDXLGraphSurgeon.ONNX_LARGE_FILE_THRESHOLD:
        onnx.save_model(model,
                        args.output_onnx_fpath,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        convert_attribute=False)
    else:
        onnx.save(model, args.output_onnx_fpath)


if __name__ == '__main__':
    args = parse_args()
    main(args)
