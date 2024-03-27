#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import numpy_helper

def update_onnx_models():
    model = onnx.load(f"resnet50_v1.onnx")
    onnx.checker.check_model(model)

    new_avg_pool_node = onnx.helper.make_node(
        name="resnet_model/AvgPool", op_type="AveragePool",
        inputs=["resnet_model/Relu_48:0"], outputs=["resnet_model/AvgPool:0"],
        kernel_shape=(7, 7))
    new_conv_node = onnx.helper.make_node(
        name="resnet_model/Dense", op_type="Conv",
        inputs=["resnet_model/AvgPool:0", "resnet_model/dense/kernel/read:0",
            "resnet_model/dense/bias/read:0"],
        outputs=["resnet_model/Dense:0"],
        kernel_shape=(1, 1), dilations=(1, 1), strides=(1, 1),
        pads=(0, 0, 0, 0))
    new_argmax_node = onnx.helper.make_node(
        name="ArgMax", op_type="ArgMax",
        inputs=["resnet_model/Dense:0"], outputs=["ArgMax:0"],
        axis=1, keepdims=1)

    insert_node_id = -1
    for node_id, node in enumerate(list(model.graph.node)):
        if node.name in ["resnet_model/Mean", "resnet_model/Squeeze",
                "resnet_model/dense/MatMul", "resnet_model/dense/BiasAdd",
                "ArgMax", "softmax_tensor"]:
            model.graph.node.remove(node)
            if insert_node_id == -1:
                insert_node_id = node_id
    model.graph.node.insert(insert_node_id, new_avg_pool_node)
    model.graph.node.insert(insert_node_id + 1, new_conv_node)
    model.graph.node.insert(insert_node_id + 2, new_argmax_node)

    for initializer in model.graph.initializer:
        if initializer.name == "resnet_model/dense/kernel/read:0":
            weight = numpy_helper.to_array(initializer)
            weight = weight.reshape(2048, 1001, 1, 1).transpose(1, 0, 2, 3)
            initializer.CopyFrom(numpy_helper.from_array(weight, initializer.name))

    output_tmp = onnx.helper.make_tensor_value_info(
        "ArgMax:0", onnx.TensorProto.INT64, ("unk__617", 1, 1, 1))
    model.graph.output.pop()
    model.graph.output.pop()
    model.graph.output.append(output_tmp)

    onnx.checker.check_model(model)
    onnx.save(model, f"resnet50_v1_opt.onnx")


update_onnx_models()
