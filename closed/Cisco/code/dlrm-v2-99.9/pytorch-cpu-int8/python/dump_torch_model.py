"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import collections
import json
import logging
import os
import sys
import threading
import time
from multiprocessing import JoinableQueue

import mlperf_loadgen as lg
import numpy as np
import torch

import dataset
import multihot_criteo
from backend_pytorch_native import get_backend

from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRMTrain
from torchrec import EmbeddingBagCollection
from torchrec.modules.embedding_configs import EmbeddingBagConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

# pylint: disable=missing-docstring

# the datasets we support
SUPPORTED_DATASETS = {
    "multihot-criteo-sample":
        (multihot_criteo.MultihotCriteo, multihot_criteo.pre_process_criteo_dlrm, multihot_criteo.DlrmPostProcess(),
         {"randomize": 'total',  "memory_map": True}),
}

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "multihot-criteo-sample",
        "backend": "pytorch-native",
        "model": "dlrm",
    }
}

last_timeing = []

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="name of the mlperf model, ie. dlrm")
    parser.add_argument("--model-path", required=True, help="path to the model file")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)

    return args

def main():
    args = get_args()

    print("dumping model")
    backend = get_backend(args.backend, args.dataset, args.use_gpu, debug=args.debug)
    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=128,
            num_embeddings=backend.num_embeddings_per_feature[feature_idx],
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]

    dlrm_model_1 = DLRM_DCN(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("cpu")
        ),
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=backend.dense_arch_layer_sizes,
        over_arch_layer_sizes=backend.over_arch_layer_sizes,
        dcn_num_layers=backend.dcn_num_layers,
        dcn_low_rank_dim=backend.dcn_low_rank_dim,
        dense_device=torch.device("cpu"),
    )
    model_1 = DLRMTrain(dlrm_model_1)

    from torchsnapshot import Snapshot
    snapshot = Snapshot(path=args.model_path)
    snapshot.restore(app_state={"model": model_1})
    state_dict = model_1.model.state_dict()
    state_dict2 = dict()
    for k, v in state_dict.items():
        if type(v) == torch.Tensor:
            state_dict2[k] = v
        else:
            dst_tensor = torch.empty(v.shape)
            v.gather(0, dst_tensor)
            state_dict2[k] = dst_tensor

    param = {'dense_arch.model._mlp.0._linear.bias': 'dense_arch.model._mlp.0.bias',
    'dense_arch.model._mlp.0._linear.weight': 'dense_arch.model._mlp.0.weight',
    'dense_arch.model._mlp.1._linear.bias': 'dense_arch.model._mlp.2.bias',
    'dense_arch.model._mlp.1._linear.weight': 'dense_arch.model._mlp.2.weight',
    'dense_arch.model._mlp.2._linear.bias': 'dense_arch.model._mlp.4.bias',
    'dense_arch.model._mlp.2._linear.weight': 'dense_arch.model._mlp.4.weight',
    'inter_arch.crossnet.V_kernels.0': 'inter_arch.crossnet.MLPs.V0.weight',
    'inter_arch.crossnet.V_kernels.1': 'inter_arch.crossnet.MLPs.V1.weight',
    'inter_arch.crossnet.V_kernels.2': 'inter_arch.crossnet.MLPs.V2.weight',
    'inter_arch.crossnet.W_kernels.0': 'inter_arch.crossnet.MLPs.W0.weight',
    'inter_arch.crossnet.W_kernels.1': 'inter_arch.crossnet.MLPs.W1.weight',
    'inter_arch.crossnet.W_kernels.2': 'inter_arch.crossnet.MLPs.W2.weight',
    'inter_arch.crossnet.bias.0': 'inter_arch.crossnet.MLPs.W0.bias',
    'inter_arch.crossnet.bias.1': 'inter_arch.crossnet.MLPs.W1.bias',
    'inter_arch.crossnet.bias.2': 'inter_arch.crossnet.MLPs.W2.bias',

    'over_arch.model.0._mlp.0._linear.bias': 'over_arch.model._mlp.0.bias',
    'over_arch.model.0._mlp.0._linear.weight': 'over_arch.model._mlp.0.weight',
    'over_arch.model.0._mlp.1._linear.bias': 'over_arch.model._mlp.2.bias',
    'over_arch.model.0._mlp.1._linear.weight': 'over_arch.model._mlp.2.weight',
    'over_arch.model.0._mlp.2._linear.bias': 'over_arch.model._mlp.4.bias',
    'over_arch.model.0._mlp.2._linear.weight': 'over_arch.model._mlp.4.weight',
    'over_arch.model.0._mlp.3._linear.bias': 'over_arch.model._mlp.6.bias',
    'over_arch.model.0._mlp.3._linear.weight': 'over_arch.model._mlp.6.weight',
    'over_arch.model.1.bias': 'over_arch.model._mlp.8.bias',
    'over_arch.model.1.weight': 'over_arch.model._mlp.8.weight',

    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_0.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.0.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_1.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.1.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_2.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.2.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_3.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.3.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_4.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.4.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_5.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.5.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_6.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.6.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_7.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.7.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_8.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.8.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_9.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.9.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_10.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.10.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_11.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.11.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_12.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.12.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_13.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.13.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_14.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.14.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_15.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.15.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_16.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.16.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_17.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.17.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_18.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.18.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_19.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.19.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_20.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.20.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_21.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.21.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_22.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.22.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_23.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.23.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_24.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.24.weight',
    'sparse_arch.embedding_bag_collection.embedding_bags.t_cat_25.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.25.weight'}

    for k, v in state_dict2.items():
        print(k)
    for tr, t in param.items():
        # assert state_dict2[t].shape == state_dict[tr].shape
        state_dict2[t] = state_dict2.pop(tr)
    torch.save(state_dict2, args.model_path + "/../dlrm-multihot-pytorch.pt")

if __name__ == "__main__":
    main()
