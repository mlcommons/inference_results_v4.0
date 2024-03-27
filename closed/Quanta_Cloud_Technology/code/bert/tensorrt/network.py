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

import numpy as np
import onnx
import tensorrt as trt
import json

from os import PathLike, sep
from pathlib import Path
from typing import Tuple

from nvmitten.utils import logging

from code.common.systems.system_list import SystemClassifications
from .builder_utils import BertConfig


def onnx_to_tf_name(onnx_name):
    """
    Converting variables in the onnx checkpoint to names corresponding to the naming convention used in the TF version, expected by the builder
    """
    onnx_name = onnx_name.lower()
    toks = [t.strip('_') for t in onnx_name.split('.')]
    if toks[0] == 'bert':  # embeddings or encoder
        if toks[1] == 'encoder':  # transformer

            if toks[-2] == 'layernorm':  # bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            elif (toks[-2] == 'dense' or toks[-2] in {'key', 'value', 'query'}) and toks[-1] == 'weight':
                toks[-1] = 'kernel'
            elif (toks[-3] == 'dense' or toks[-3] in {'key', 'value', 'query'}) and toks[-1] == 'amax':
                if toks[-2] == 'weight_quantizer':
                    toks[-2] = 'kernel'
                elif toks[-2] == 'input_quantizer':
                    toks[-2] = 'input'

            if 'final_input_quantizer' not in toks[2]:
                toks = toks[3:]
                toks[0] = 'l{}'.format(int(toks[0]))
        else:
            if toks[-2] == 'layernorm':  # bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            else:  # embeddings: drop "_weight" suffix
                if toks[-1] == 'amax':
                    toks[-2] = 'amax'
                toks = toks[:-1]
    elif 'qa' in onnx_name:
        name = 'cls_squad_output_bias' if toks[-1] == 'bias' else 'cls_squad_output_weights'
        return name
    else:
        raise RuntimeError(f"Encountered unknown case: {onnx_name}")
    parsed = '_'.join(toks)
    return parsed


class BERTNetwork:
    """Wrapper around trt.INetworkDefinition for BERT.
    """

    def __init__(self, model_path: PathLike, network: trt.INetworkDefinition, config: BertConfig, dtype: trt.DataType):
        model = onnx.load(model_path)
        weights_dict = {onnx_to_tf_name(w.name): np.frombuffer(w.raw_data, np.float32).reshape(w.dims)
                        for w in model.graph.initializer}
        self.weights_dict = weights_dict
        self.config = config
        self.dtype = dtype
        self.plg_registry = trt.get_plugin_registry()

        self.network = self.build_network(network)

    def add_gelu(self, network, input_tensor):
        """This will trigger FC+GELU fusion in TRT"""
        shape = (1, ) * len(input_tensor.shape)
        POW = network.add_constant(shape, trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
        MULTIPLY = network.add_constant(shape, trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
        SQRT = network.add_constant(shape, trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
        ONE = network.add_constant(shape, trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
        HALF = network.add_constant(shape, trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
        X_pow = network.add_elementwise(input_tensor, POW.get_output(0), trt.ElementWiseOperation.POW)
        X_pow_t = X_pow.get_output(0)
        X_mul = network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
        X_add = network.add_elementwise(input_tensor, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
        X_sqrt = network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
        X_sqrt_tensor = X_sqrt.get_output(0)
        X_tanh = network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
        X_tanh_tensor = X_tanh.get_output(0)
        X_one = network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
        CDF = network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
        gelu_layer = network.add_elementwise(CDF.get_output(0), input_tensor, trt.ElementWiseOperation.PROD)

        # enable elementwise fusing for int8 && fp16
        POW.precision = trt.DataType.FLOAT
        MULTIPLY.precision = trt.DataType.FLOAT
        SQRT.precision = trt.DataType.FLOAT
        ONE.precision = trt.DataType.FLOAT
        HALF.precision = trt.DataType.FLOAT
        X_pow.precision = trt.DataType.FLOAT
        X_mul.precision = trt.DataType.FLOAT
        X_add.precision = trt.DataType.FLOAT
        X_sqrt.precision = trt.DataType.FLOAT
        X_tanh.precision = trt.DataType.FLOAT
        X_one.precision = trt.DataType.FLOAT
        CDF.precision = trt.DataType.FLOAT
        gelu_layer.precision = trt.DataType.FLOAT
        return gelu_layer

    def add_embeddings_layer(self, network):
        pc_emb = self.plg_registry.get_plugin_creator("CustomEmbLayerNormPluginDynamic", "2", "")

        wbeta = trt.PluginField("bert_embeddings_layernorm_beta",
                                self.weights_dict["bert_embeddings_layernorm_beta"],
                                trt.PluginFieldType.FLOAT32)
        wgamma = trt.PluginField("bert_embeddings_layernorm_gamma",
                                 self.weights_dict["bert_embeddings_layernorm_gamma"],
                                 trt.PluginFieldType.FLOAT32)
        wwordemb = trt.PluginField("bert_embeddings_word_embeddings",
                                   self.weights_dict["bert_embeddings_word_embeddings"],
                                   trt.PluginFieldType.FLOAT32)
        wtokemb = trt.PluginField("bert_embeddings_token_type_embeddings",
                                  self.weights_dict["bert_embeddings_token_type_embeddings"],
                                  trt.PluginFieldType.FLOAT32)
        wposemb = trt.PluginField("bert_embeddings_position_embeddings",
                                  self.weights_dict["bert_embeddings_position_embeddings"],
                                  trt.PluginFieldType.FLOAT32)
        output_fp16 = trt.PluginField("output_fp16",
                                      np.array([int(trt.float16)]).astype(np.int32),
                                      trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([wbeta, wgamma, wwordemb, wtokemb, wposemb, output_fp16])
        embln_plugin = pc_emb.create_plugin("embeddings", pfc)

        input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=(-1,))
        segment_ids = network.add_input(name="segment_ids", dtype=trt.int32, shape=(-1,))
        cu_seqlens = network.add_input(name="cu_seqlens", dtype=trt.int32, shape=(-1,))
        # dummy input used to indicate maximum sequence length to plugins
        max_seqlen = network.add_input(name="max_seqlen", dtype=trt.int32, shape=(-1,))
        inputs = [input_ids, segment_ids, cu_seqlens, max_seqlen]

        emb_layer = network.add_plugin_v2(inputs, embln_plugin)
        emb_layer.name = 'embln'

        embeddings = emb_layer.get_output(0)
        embeddings.dtype = self.dtype
        mask = emb_layer.get_output(1)

        return {"emb_layer": emb_layer,
                "embeddings": embeddings,
                "mask": mask,
                "max_seqlen": max_seqlen,
                "cu_seqlens": cu_seqlens}

    def add_encoder_layer(self, network, input_tensor, max_seqlen, cu_seqlens, layer, mask):
        raise NotImplementedError

    def add_encoder_stack(self, network, emb_layer_data):
        embeddings = emb_layer_data["embeddings"]
        for layer in range(self.config.L):
            embeddings = self.add_encoder_layer(network,
                                                embeddings,
                                                emb_layer_data["max_seqlen"],
                                                emb_layer_data["cu_seqlens"],
                                                layer,
                                                emb_layer_data["mask"])
        return embeddings

    def add_final_fc(self, network, embeddings):
        Wsquad = self.weights_dict['cls_squad_output_weights']
        Bsquad = self.weights_dict['cls_squad_output_bias']

        squad_output = network.add_fully_connected(embeddings, 2, Wsquad, Bsquad)
        squad_output.name = 'squad_logits'
        logits = squad_output.get_output(0)
        return logits

    def build_network(self, network):
        emb_layer_data = self.add_embeddings_layer(network)
        final_embeddings = self.add_encoder_stack(network, emb_layer_data)
        logits = self.add_final_fc(network, final_embeddings)

        # NOTE: TRT9 may allow setting the DTYPE only when the tensor is marked input/output
        logits.dtype = trt.float16
        network.mark_output(logits)
        return network


class BERTVarSeqLenFP16(BERTNetwork):
    def __init__(self,
                 network: trt.INetworkDefinition,
                 config: BertConfig,
                 model_path: PathLike = "build/models/bert/bert_large_v1_1.onnx"):
        super().__init__(model_path, network, config, trt.float16)

    def add_encoder_layer(self, network, input_tensor, max_seqlen, cu_seqlen, layer, mask):
        """Builds one encoder layer in FP16 with var seqlen.
        Sets the dynamic ranges extracted from the qat checkpoint."""

        qkv_plg_creator = self.plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "2", "")
        pc_skln = self.plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "2", "")
        N = self.config.N
        H = self.config.H
        prefix = 'l{}_'.format(layer)

        # FC QKV
        Wqkv = np.zeros((3, self.config.hidden_size, self.config.hidden_size), np.float32)
        Bqkv = np.zeros((3, self.config.hidden_size), np.float32)
        Wqkv[0, :, :] = self.weights_dict[prefix + 'attention_self_query_kernel']
        Wqkv[1, :, :] = self.weights_dict[prefix + 'attention_self_key_kernel']
        Wqkv[2, :, :] = self.weights_dict[prefix + 'attention_self_value_kernel']
        Bqkv[0, :] = self.weights_dict[prefix + 'attention_self_query_bias']
        Bqkv[1, :] = self.weights_dict[prefix + 'attention_self_key_bias']
        Bqkv[2, :] = self.weights_dict[prefix + 'attention_self_value_bias']

        Wqkv = np.ascontiguousarray(Wqkv.reshape((3, N, H, N, H)).transpose((1, 0, 2, 3, 4)))
        Bqkv = np.ascontiguousarray(Bqkv.reshape((3, N, H)).transpose((1, 0, 2)))

        fc_qkv = network.add_fully_connected(input_tensor, self.config.qkv_size, Wqkv, Bqkv)
        fc_qkv.name = prefix + 'fc_qkv'
        fc_qkv_out = fc_qkv.get_output(0)
        fc_qkv_out.name = prefix + 'attention_self_qkv_mult'
        # QKV2CTX
        pf_type = trt.PluginField("type_id", np.array([int(self.dtype)], np.int32), trt.PluginFieldType.INT32)
        pf_hidden_size = trt.PluginField("hidden_size", np.array([self.config.hidden_size], np.int32), trt.PluginFieldType.INT32)
        pf_num_heads = trt.PluginField("num_heads", np.array([self.config.N], np.int32), trt.PluginFieldType.INT32)
        pf_has_mask = trt.PluginField("has_mask", np.array([1], np.int32), trt.PluginFieldType.INT32)
        pf_var_seqlen = trt.PluginField("var_seqlen", np.array([int(1)], np.int32), trt.PluginFieldType.FLOAT32)

        pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_has_mask, pf_type, pf_var_seqlen])
        qkv2ctx_plug = qkv_plg_creator.create_plugin("qkv2ctx", pfc)

        qkv2ctx_layer = network.add_plugin_v2([fc_qkv_out, mask, cu_seqlen, max_seqlen], qkv2ctx_plug)
        qkv2ctx_layer.name = prefix + 'qkv_to_ctx'
        qkv2ctx_out = qkv2ctx_layer.get_output(0)
        # FC AOUT
        Waout = self.weights_dict[prefix + 'attention_output_dense_kernel']
        Baout = self.weights_dict[prefix + 'attention_output_dense_bias']
        fc_aout = network.add_fully_connected(qkv2ctx_out, self.config.hidden_size, Waout, Baout)
        fc_aout.precision = self.dtype
        fc_aout.name = prefix + 'fc_aout'
        fc_aout_out = fc_aout.get_output(0)
        fc_aout_out.dtype = self.dtype
        # Skip-Layernorm 1
        pf_ld = trt.PluginField("ld", np.array([self.config.hidden_size], np.int32), trt.PluginFieldType.INT32)
        pf_type = trt.PluginField("type_id", np.array([int(self.dtype)], np.int32), trt.PluginFieldType.INT32)
        pf_beta = trt.PluginField("beta", self.weights_dict[prefix + 'attention_output_layernorm_beta'], trt.PluginFieldType.FLOAT32)
        pf_gamma = trt.PluginField("gamma", self.weights_dict[prefix + 'attention_output_layernorm_gamma'], trt.PluginFieldType.FLOAT32)
        pf_bias = trt.PluginField("bias", Baout, trt.PluginFieldType.FLOAT32)
        fields = [pf_ld, pf_beta, pf_gamma, pf_type]
        pfc = trt.PluginFieldCollection(fields)
        skipln_plug = pc_skln.create_plugin("skipln", pfc)

        fc_aout_out.dtype = self.dtype
        skipln_inputs = [fc_aout_out, input_tensor]
        skln1 = network.add_plugin_v2(skipln_inputs, skipln_plug)
        skln1.name = prefix + 'skln_1'
        skln1_out = skln1.get_output(0)
        skln1_out.dtype = self.dtype
        # FC MID
        Wmid = self.weights_dict[prefix + 'intermediate_dense_kernel']
        Bmid = self.weights_dict[prefix + 'intermediate_dense_bias']
        fc_mid = network.add_fully_connected(skln1_out, self.config.mid_size, Wmid, Bmid)
        fc_mid.name = prefix + 'fc_mid'
        fc_mid_out = fc_mid.get_output(0)
        # GELU
        gelu_layer = self.add_gelu(network, fc_mid_out)
        gelu_layer.name = prefix + 'gelu'
        gelu_out = gelu_layer.get_output(0)
        # FC OUT
        Wout = self.weights_dict[prefix + 'output_dense_kernel']
        Bout = self.weights_dict[prefix + 'output_dense_bias']
        fc_out = network.add_fully_connected(gelu_out, self.config.hidden_size, Wout, Bout)
        fc_out.name = prefix + 'fc_out'
        fc_out.precision = self.dtype
        fc_out_out = fc_out.get_output(0)
        fc_out_out.dtype = self.dtype
        # Skip-Layernorm 2
        pf_beta = trt.PluginField("beta", self.weights_dict[prefix + 'output_layernorm_beta'], trt.PluginFieldType.FLOAT32)
        pf_gamma = trt.PluginField("gamma", self.weights_dict[prefix + 'output_layernorm_gamma'], trt.PluginFieldType.FLOAT32)
        pf_bias = trt.PluginField("bias", Bout, trt.PluginFieldType.FLOAT32)
        fields = [pf_ld, pf_beta, pf_gamma, pf_type]
        pfc = trt.PluginFieldCollection(fields)
        skipln_plug = pc_skln.create_plugin("skipln", pfc)
        skln1_out.dtype = self.dtype
        skipln_inputs = [fc_out_out, skln1_out]
        skln2 = network.add_plugin_v2(skipln_inputs, skipln_plug)
        skln2.name = prefix + 'skln_2'
        skln2_out = skln2.get_output(0)

        return skln2_out


class BERTVarSeqLenINT8(BERTNetwork):
    def __init__(self,
                 network: trt.INetworkDefinition,
                 config: BertConfig,
                 use_small_tile_gemm_plugin: bool = True,
                 model_path: PathLike = "build/models/bert/bert_large_v1_1_fake_quant.onnx"):
        self.use_small_tile_gemm_plugin = use_small_tile_gemm_plugin
        super().__init__(model_path, network, config, trt.int8)

    def add_small_tile_gemm_fc(self,
                               network,
                               input_tensor,
                               input_channels,
                               output_channels,
                               layer_name,
                               weight,
                               bias,
                               input_dr,
                               output_dr,
                               use_gelu=False):
        """ Build one plugin layer of the Small-Tile GEMM kernel"""
        logging.info(f"Replacing {layer_name} with small-tile GEMM plugin.")
        plugin_name = "SmallTileGEMM_TRT"
        plugin_layer_name = layer_name + plugin_name
        plugin_version = '1'
        plugin_creator = trt.get_plugin_registry().get_plugin_creator(plugin_name, plugin_version, '')
        if plugin_creator is None:
            raise Exception("Cannot find small tile GEMM plugin creator for top_mlp")

        scale = np.ones([output_channels], dtype=np.float32)

        fields = []
        fields.append(trt.PluginField("inputChannels", np.array([input_channels],
                                                                dtype=np.int32), trt.PluginFieldType.INT32))
        fields.append(trt.PluginField("weight", weight, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("bias", bias, trt.PluginFieldType.FLOAT32))
        fields.append(trt.PluginField("scale", scale, trt.PluginFieldType.FLOAT32))
        # Deprecated, but left for backward compatibility
        fields.append(trt.PluginField("fairShareCacheSize", np.array([120],
                                                                     dtype=np.int32), trt.PluginFieldType.INT32))
        fields.append(trt.PluginField("dynamicRanges", np.array([input_dr, output_dr],
                                                                dtype=np.float32), trt.PluginFieldType.FLOAT32))

        if use_gelu:
            rescale = np.ones([output_channels], dtype=np.float32)
            fields.append(trt.PluginField("rescale", rescale, trt.PluginFieldType.FLOAT32))
            fields.append(trt.PluginField("epilogueScaleBiasGelu", np.array([1],
                                                                            dtype=np.int32), trt.PluginFieldType.INT32))
        else:
            fields.append(trt.PluginField("epilogueScaleBias", np.array([1],
                                                                        dtype=np.int32), trt.PluginFieldType.INT32))

        fields = trt.PluginFieldCollection(fields)

        plugin = plugin_creator.create_plugin(plugin_layer_name, fields)
        if plugin is None:
            raise Exception("Cannot create BERT Small-Tile GEMM plugin for {}.".format(plugin_layer_name))
        plugin_layer = network.add_plugin_v2([input_tensor], plugin)
        return plugin_layer

    def add_encoder_layer(self, network, input_tensor, max_seqlen, cu_seqlens, layer, mask):
        """Builds one encoder layer in INT8 with var seqlen.
        Sets the dynamic ranges extracted from the qat checkpoint."""

        qkv_plg_creator = self.plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "2", "")
        pc_skln = self.plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "2", "")
        # Number of heads
        N = self.config.N
        # Hidden sizes (embedding sizes) // number of heads
        H = self.config.H
        prefix = 'l{}_'.format(layer)

        dr_input = self.weights_dict[prefix + 'attention_self_query_input_amax']
        assert(dr_input == self.weights_dict[prefix + 'attention_self_key_input_amax'])
        assert(dr_input == self.weights_dict[prefix + 'attention_self_value_input_amax'])
        input_tensor.set_dynamic_range(-dr_input, dr_input)

        # FC QKV
        dr_qkv = max(
            self.weights_dict[prefix + 'attention_self_qv_a_input_quantizer_amax'],
            self.weights_dict[prefix + 'attention_self_qv_b_input_quantizer_amax'],
            self.weights_dict[prefix + 'attention_self_av_b_input_quantizer_amax'],
        )
        Wqkv = np.zeros((3, self.config.hidden_size, self.config.hidden_size), np.float32)
        Bqkv = np.zeros((3, self.config.hidden_size), np.float32)
        Wqkv[0, :, :] = self.weights_dict[prefix + 'attention_self_query_kernel']
        Wqkv[1, :, :] = self.weights_dict[prefix + 'attention_self_key_kernel']
        Wqkv[2, :, :] = self.weights_dict[prefix + 'attention_self_value_kernel']
        Bqkv[0, :] = self.weights_dict[prefix + 'attention_self_query_bias']
        Bqkv[1, :] = self.weights_dict[prefix + 'attention_self_key_bias']
        Bqkv[2, :] = self.weights_dict[prefix + 'attention_self_value_bias']

        Wqkv = np.ascontiguousarray(Wqkv.reshape((3, N, H, N, H)).transpose((1, 0, 2, 3, 4)))
        Bqkv = np.ascontiguousarray(Bqkv.reshape((3, N, H)).transpose((1, 0, 2)))

        if self.use_small_tile_gemm_plugin:
            # Replace QKV FC with GEMM plugin
            # [BS, 1024, 1, 1] -> [BS, 3072, 1, 1]
            fc_qkv_input_channels = input_tensor.shape[1]
            fc_qkv_layer_name = prefix + 'fc_qkv'
            fc_qkv_plugin = self.add_small_tile_gemm_fc(network,
                                                        input_tensor,
                                                        fc_qkv_input_channels,
                                                        self.config.qkv_size,
                                                        fc_qkv_layer_name,
                                                        Wqkv,
                                                        Bqkv,
                                                        dr_input,
                                                        dr_qkv,
                                                        use_gelu=False)
            fc_qkv_out = fc_qkv_plugin.get_output(0)
        else:
            fc_qkv = network.add_convolution(input_tensor, self.config.qkv_size, (1, 1), Wqkv, Bqkv)
            fc_qkv.name = prefix + 'fc_qkv'
            fc_qkv_out = fc_qkv.get_output(0)

        fc_qkv_out.name = prefix + 'attention_self_qkv_mult'
        fc_qkv_out.set_dynamic_range(-dr_qkv, dr_qkv)

        # QKV2CTX
        dr_probs = self.weights_dict[prefix + 'attention_self_av_a_input_quantizer_amax']
        dq_probs = dr_probs / 127.0
        pf_type = trt.PluginField("type_id", np.array([int(trt.int8)], np.int32), trt.PluginFieldType.INT32)
        pf_hidden_size = trt.PluginField("hidden_size", np.array([self.config.hidden_size], np.int32), trt.PluginFieldType.INT32)
        pf_num_heads = trt.PluginField("num_heads", np.array([self.config.N], np.int32), trt.PluginFieldType.INT32)
        pf_has_mask = trt.PluginField("has_mask", np.array([1], np.int32), trt.PluginFieldType.INT32)
        pf_dq_probs = trt.PluginField("dq_probs", np.array([dq_probs], np.float32), trt.PluginFieldType.FLOAT32)
        pf_var_seqlen = trt.PluginField("var_seqlen", np.array([int(1)], np.int32), trt.PluginFieldType.FLOAT32)

        pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_has_mask, pf_type, pf_dq_probs, pf_var_seqlen])
        qkv2ctx_plug = qkv_plg_creator.create_plugin("qkv2ctx", pfc)

        dr_ctx = self.weights_dict[prefix + 'attention_output_dense_input_amax']
        qkv2ctx_layer = network.add_plugin_v2([fc_qkv_out, mask, cu_seqlens, max_seqlen], qkv2ctx_plug)
        qkv2ctx_layer.name = prefix + 'qkv_to_ctx'
        qkv2ctx_out = qkv2ctx_layer.get_output(0)
        qkv2ctx_out.set_dynamic_range(-dr_ctx, dr_ctx)

        # FC AOUT
        dr_fc_aout = self.weights_dict[prefix + 'attention_output_add_local_input_quantizer_amax']
        Waout = self.weights_dict[prefix + 'attention_output_dense_kernel']
        Baout = self.weights_dict[prefix + 'attention_output_dense_bias']

        if self.use_small_tile_gemm_plugin:
            # Replace fc aout with small-Tile GEMM
            # [BS, 1024, 1, 1] -> [BS, 1024, 1, 1]
            fc_aout_input_channels = qkv2ctx_out.shape[1]
            fc_aout_layer_name = prefix + 'fc_aout'
            fc_aout_plugin = self.add_small_tile_gemm_fc(network,
                                                         qkv2ctx_out,
                                                         fc_aout_input_channels,
                                                         self.config.hidden_size,
                                                         fc_aout_layer_name,
                                                         Waout,
                                                         Baout,
                                                         dr_ctx,
                                                         dr_fc_aout,
                                                         use_gelu=False)
            fc_aout_out = fc_aout_plugin.get_output(0)
        else:
            fc_aout = network.add_convolution(qkv2ctx_out, self.config.hidden_size, (1, 1), Waout, Baout)
            fc_aout.precision = self.dtype
            fc_aout.name = prefix + 'fc_aout'
            fc_aout_out = fc_aout.get_output(0)

        fc_aout_out.dtype = self.dtype
        fc_aout_out.name = prefix + 'attention_fc_aout'
        fc_aout_out.set_dynamic_range(-dr_fc_aout, dr_fc_aout)

        # Skip-Layernorm 1
        dr_skln1 = self.weights_dict[prefix + 'intermediate_dense_input_amax']
        pf_ld = trt.PluginField("ld", np.array([self.config.hidden_size], np.int32), trt.PluginFieldType.INT32)
        pf_type = trt.PluginField("type_id", np.array([int(self.dtype)], np.int32), trt.PluginFieldType.INT32)
        pf_beta = trt.PluginField("beta", self.weights_dict[prefix + 'attention_output_layernorm_beta'], trt.PluginFieldType.FLOAT32)
        pf_gamma = trt.PluginField("gamma", self.weights_dict[prefix + 'attention_output_layernorm_gamma'], trt.PluginFieldType.FLOAT32)
        pf_bias = trt.PluginField("bias", Baout, trt.PluginFieldType.FLOAT32)
        fields = [pf_ld, pf_beta, pf_gamma, pf_type]
        pfc = trt.PluginFieldCollection(fields)
        skipln_plug = pc_skln.create_plugin("skipln", pfc)

        fc_aout_out.dtype = self.dtype

        skipln_inputs = [fc_aout_out, input_tensor]
        skln1 = network.add_plugin_v2(skipln_inputs, skipln_plug)
        skln1.name = prefix + 'skln_1'
        skln1_out = skln1.get_output(0)
        skln1_out.dtype = self.dtype
        skln1_out.set_dynamic_range(-dr_skln1, dr_skln1)

        # FC MID
        Wmid = self.weights_dict[prefix + 'intermediate_dense_kernel']
        Bmid = self.weights_dict[prefix + 'intermediate_dense_bias']

        dr_gelu = self.weights_dict[prefix + 'output_dense_input_amax']

        if self.use_small_tile_gemm_plugin:
            # Replace FC MID with small-tile GEMM kernel (with Gelu epilogue)
            # [BS, 1024, 1, 1] -> [BS, 4096, 1, 1]
            fc_mid_input_channels = skln1_out.shape[1]
            fc_mid_layer_name = prefix + 'fc_mid_gelu'
            fc_mid_plugin = self.add_small_tile_gemm_fc(network,
                                                        skln1_out,
                                                        fc_mid_input_channels,
                                                        self.config.mid_size,
                                                        fc_mid_layer_name,
                                                        Wmid,
                                                        Bmid,
                                                        dr_skln1,
                                                        dr_gelu,
                                                        use_gelu=True)
            gelu_out = fc_mid_plugin.get_output(0)
        else:
            fc_mid = network.add_convolution(skln1_out, self.config.mid_size, (1, 1), Wmid, Bmid)
            fc_mid.name = prefix + 'fc_mid'
            fc_mid_out = fc_mid.get_output(0)
            fc_mid_out.name = prefix + 'fc_mid_out'
            # GELU
            gelu_layer = self.add_gelu(network, fc_mid_out)
            gelu_layer.name = prefix + 'gelu'
            gelu_out = gelu_layer.get_output(0)

        gelu_out.name = prefix + 'gelu_out'
        gelu_out.dtype = self.dtype
        gelu_out.set_dynamic_range(-dr_gelu, dr_gelu)

        # FC OUT
        dr_fc_out = self.weights_dict[prefix + 'output_add_local_input_quantizer_amax']
        Wout = self.weights_dict[prefix + 'output_dense_kernel']
        Bout = self.weights_dict[prefix + 'output_dense_bias']
        fc_out = network.add_convolution(gelu_out, self.config.hidden_size, (1, 1), Wout, Bout)
        fc_out.name = prefix + 'fc_out'
        fc_out.precision = self.dtype
        fc_out_out = fc_out.get_output(0)
        fc_out_out.dtype = self.dtype
        fc_out_out.name = prefix + 'fc_out_out'
        fc_out_out.set_dynamic_range(-dr_fc_out, dr_fc_out)

        # Skip-Layernorm 2
        pf_beta = trt.PluginField("beta", self.weights_dict[prefix + 'output_layernorm_beta'], trt.PluginFieldType.FLOAT32)
        pf_gamma = trt.PluginField("gamma", self.weights_dict[prefix + 'output_layernorm_gamma'], trt.PluginFieldType.FLOAT32)
        pf_bias = trt.PluginField("bias", Bout, trt.PluginFieldType.FLOAT32)
        fields = [pf_ld, pf_beta, pf_gamma, pf_type]
        pfc = trt.PluginFieldCollection(fields)
        skipln_plug = pc_skln.create_plugin("skipln", pfc)

        skln1_out.dtype = self.dtype  # It does not build without setting this here, in addition to above. WHY??!?!

        skipln_inputs = [fc_out_out, skln1_out]
        skln2 = network.add_plugin_v2(skipln_inputs, skipln_plug)
        skln2.name = prefix + 'skln_2'
        skln2_out = skln2.get_output(0)
        skln2_out.name = prefix + "skln_2_out"

        return skln2_out

    def add_embeddings_layer(self, network):
        emb_layer_data = super().add_embeddings_layer(network)
        emb_layer_data["mask"].set_dynamic_range(-1, 1)
        return emb_layer_data

    def add_final_fc(self, network, embeddings):
        Wsquad = self.weights_dict['cls_squad_output_weights']
        Bsquad = self.weights_dict['cls_squad_output_bias']
        dr_out = self.weights_dict['bert_encoder_final_input_quantizer_amax']
        embeddings.set_dynamic_range(-dr_out, dr_out)

        squad_output = network.add_convolution(embeddings, 2, (1, 1), Wsquad, Bsquad)
        squad_output.name = 'squad_FC_layer'
        logits = squad_output.get_output(0)
        logits.name = "squad_logits"
        logits.set_dynamic_range(-1, 1)

        # output shape will be sum_s x 2 (x 1 x 1)
        return logits


class BERTFP8FasterTransformer(BERTNetwork):
    def __init__(self,
                 network: trt.INetworkDefinition,
                 config: BertConfig,
                 max_seq_len: int,
                 ft_weights_dir: PathLike,
                 model_path: PathLike = "build/models/bert/bert_large_v1_1.onnx"):
        self.max_seq_len = max_seq_len
        # Verify the FP8 weight and scales are in place.
        if not Path(ft_weights_dir).exists():
            raise RuntimeError(f"FP8 weight is not found in {ft_weights_dir}, Exiting...")
        self.ft_weights_dir = str(ft_weights_dir)

        # The plugin code is naive and assumes that the directory has a directory marker at the end. Add it here.
        if not self.ft_weights_dir.endswith(sep):
            self.ft_weights_dir += sep

        super().__init__(model_path, network, config, trt.DataType.FP8)  # dtype is unused, just here for consistency

    def add_gelu(self, network, input_tensor):
        # Delete base implementation to prevent calling
        raise NotImplementedError

    def add_embeddings_layer(self, network):
        # Delete base implementation to prevent calling
        raise NotImplementedError

    def add_encoder_stack(self, network, emb_layer_data):
        # Delete base implementation to prevent calling
        raise NotImplementedError

    def add_final_fc(self, network, embeddings):
        # Delete base implementation to prevent calling
        raise NotImplementedError

    def build_network(self, network):
        if not (SystemClassifications.is_hopper() or SystemClassifications.is_ada()):
            raise RuntimeError("FP8 only supported on Hopper and Ada")

        pc_ft = self.plg_registry.get_plugin_creator("BertFp8Plugin", "1", "")
        fields = [
            trt.PluginField("num_heads", np.array([self.config.N], dtype=np.int32), trt.PluginFieldType.INT32),
            trt.PluginField("size_per_head", np.array([self.config.H], dtype=np.int32), trt.PluginFieldType.INT32),
            trt.PluginField("num_layers", np.array([self.config.L], dtype=np.int32), trt.PluginFieldType.INT32),
            trt.PluginField("max_seq_len", np.array([self.max_seq_len], dtype=np.int32), trt.PluginFieldType.INT32),
            trt.PluginField("vocab_size", np.array([self.config.vocab_size], dtype=np.int32), trt.PluginFieldType.INT32),
            trt.PluginField("max_position_embeddings",
                            np.array([self.config.max_position_embeddings], dtype=np.int32),
                            trt.PluginFieldType.INT32),
            trt.PluginField("token_type_vocab_size",
                            np.array([self.config.type_vocab_size], dtype=np.int32),
                            trt.PluginFieldType.INT32),
            trt.PluginField("remove_padding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32),
            trt.PluginField("fp8_mode", np.array([2], dtype=np.int32), trt.PluginFieldType.INT32),
            trt.PluginField("weightDirPath",
                            np.array(list(self.ft_weights_dir.encode()), dtype=np.int8),
                            trt.PluginFieldType.CHAR)
        ]
        pfc = trt.PluginFieldCollection(fields)
        ft_fp8_plugin = pc_ft.create_plugin("ft_plugin", pfc)

        input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=(-1, -1))
        token_type_ids = network.add_input(name="token_type_ids", dtype=trt.int32, shape=(-1, -1))
        sequence_lengths = network.add_input(name="sequence_lengths", dtype=trt.int32, shape=(-1,))
        inputs = [input_ids, token_type_ids, sequence_lengths]
        ft_bert_layer = network.add_plugin_v2(inputs, ft_fp8_plugin)
        ft_bert_layer.name = 'ft_bert'

        # (bs, 384, 1024)
        last_embeddings = ft_bert_layer.get_output(0)
        last_embeddings.name = 'last_embeddings'

        Wsquad = self.weights_dict['cls_squad_output_weights']
        Bsquad = self.weights_dict['cls_squad_output_bias']

        last_embeddings_packed = network.add_shuffle(last_embeddings)
        last_embeddings_packed.reshape_dims = (-1, 1024, 1, 1)

        squad_output = network.add_convolution(last_embeddings_packed.get_output(0), 2, (1, 1), Wsquad, Bsquad)
        squad_output.name = 'squad_FC_Layer'
        logits = squad_output.get_output(0)
        logits.name = 'squad_logits'

        # output shape will be [bs * 384, 2, 1, 1]
        logits.dtype = trt.float16
        network.mark_output(logits)
        return network
