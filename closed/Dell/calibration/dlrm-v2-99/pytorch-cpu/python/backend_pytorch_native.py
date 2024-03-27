"""
pytoch native backend for dlrm
"""
import os
import torch
import backend
import numpy as np
import sys

from typing import List, Optional, Union, Callable

from dataset import Dataset
from model.dlrm_model import DLRMMLPerf
from torch import nn
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.datasets.random import RandomRecDataset
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
import intel_extension_for_pytorch as ipex

# Modules for distributed running
import torch.multiprocessing as mp

def get_backend(backend, dataset, use_gpu, debug):
    if backend == "pytorch-native":
        n_cores = int(os.environ.get("WORLD_SIZE", 1))
        if n_cores > 1:
            if dataset == "debug":
                # 1. Syntetic debug dataset
                backend = BackendPytorchNative(
                    num_embeddings_per_feature = [2 for _ in range(26)],
                    embedding_dim=128,
                    dcn_num_layers=3,
                    dcn_low_rank_dim=512,
                    dense_arch_layer_sizes=[512, 256, 128],
                    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
                    use_gpu=use_gpu,
                    debug=True
                )
            elif dataset == "multihot-criteo-sample":
                # 2. Syntetic multihot criteo sample
                backend = BackendPytorchNative(
                    num_embeddings_per_feature = [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36],
                    embedding_dim=128,
                    dcn_num_layers=3,
                    dcn_low_rank_dim=512,
                    dense_arch_layer_sizes=[512, 256, 128],
                    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
                    use_gpu=use_gpu,
                    debug=debug
                )
            elif dataset == "multihot-criteo":
                # 3. Syntetic multihot criteo
                backend = BackendPytorchNative(
                    num_embeddings_per_feature = [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36],
                    embedding_dim=128,
                    dcn_num_layers=3,
                    dcn_low_rank_dim=512,
                    dense_arch_layer_sizes=[512, 256, 128],
                    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
                    use_gpu=use_gpu,
                    debug=debug
                )
            else:
                raise ValueError("only debug|multihot-criteo-sample|multihot-criteo dataset options are supported")
        else:
            if dataset == "debug":
                # 1. Syntetic debug dataset
                backend = BackendPytorchNative(
                    num_embeddings_per_feature = [2 for _ in range(26)],
                    embedding_dim=128,
                    dcn_num_layers=3,
                    dcn_low_rank_dim=512,
                    dense_arch_layer_sizes=[512, 256, 128],
                    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
                    use_gpu=use_gpu,
                    debug=True
                )
            elif dataset == "multihot-criteo-sample":
                # 2. Syntetic multihot criteo sample
                backend = BackendPytorchNative(
                    num_embeddings_per_feature = [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36],
                    embedding_dim=128,
                    dcn_num_layers=3,
                    dcn_low_rank_dim=512,
                    dense_arch_layer_sizes=[512, 256, 128],
                    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
                    use_gpu=use_gpu,
                    debug=debug
                )
            elif dataset == "multihot-criteo":
                # 3. Syntetic multihot criteo
                backend = BackendPytorchNative(
                    num_embeddings_per_feature = [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36],
                    embedding_dim=128,
                    dcn_num_layers=3,
                    dcn_low_rank_dim=512,
                    dense_arch_layer_sizes=[512, 256, 128],
                    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
                    use_gpu=use_gpu,
                    debug=debug
                )
            else:
                raise ValueError("only debug|multihot-criteo-sample|multihot-criteo dataset options are supported")

    else:
        raise ValueError("unknown backend: " + backend)
    return backend

def unpack(t: KeyedJaggedTensor) -> (torch.Tensor, List[torch.Tensor], List[torch.Tensor]):
    dense = t.dense_features
    sparse = t.sparse_features
    features = [sparse[f'cat_{i}'] for i in range(26)]
    index = [f.values().long() for f in features]
    offset = [f.offsets() for f in features]
    return (dense, tuple(index), tuple(offset))

class BackendPytorchNative(backend.Backend):
    def __init__(
        self,
        num_embeddings_per_feature,
        embedding_dim=128,
        dcn_num_layers=3,
        dcn_low_rank_dim=512,
        dense_arch_layer_sizes=[512, 256, 128],
        over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
        use_gpu=False,
        debug=False,
    ):
        super(BackendPytorchNative, self).__init__()
        self.i = 0
        self.sess = None
        self.model = None

        self.embedding_dim = embedding_dim
        self.num_embeddings_per_feature = num_embeddings_per_feature
        self.dcn_num_layers = dcn_num_layers
        self.dcn_low_rank_dim = dcn_low_rank_dim
        self.dense_arch_layer_sizes = dense_arch_layer_sizes
        self.over_arch_layer_sizes = over_arch_layer_sizes
        self.debug = debug

        print("Using CPU...")

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native-dlrm"

    def load(self, args, dataset):
        # debug prints
        # print(model_path, inputs, outputs)
        max_batchsize = args.max_batchsize
        model_path = args.model_path
        inputs = args.inputs
        outputs = args.outputs
        int8_configure_dir = args.int8_configure_dir
        calibration = args.calibration
        print(f"Loading model from {model_path}")
        print("Initializing model...")
        model = DLRMMLPerf(
            embedding_dim=self.embedding_dim,
            num_embeddings_pool=self.num_embeddings_per_feature,
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=self.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.over_arch_layer_sizes,
            dcn_num_layers=self.dcn_num_layers,
            dcn_low_rank_dim=self.dcn_low_rank_dim,
            use_int8=args.use_int8,
            use_bf16=args.use_bf16
        )
        # path_to_sharded_weights should have 2 subdirectories - batched and sharded
        # If we need to load the weights on different device or world size, we would need to change the process
        # group accordingly. If we would want to load on 8 GPUs, the process group created above should be fine
        # to understand sharding, --print_sharding_plan flag should be used while running dlrm_main.py in
        # torcherec implementation
        if args.use_bf16:
            if not self.debug:
                print("Loading model weights...")
                model.load_state_dict(torch.load(model_path))
            model.training = False
            self.model = ipex.optimize(model, torch.bfloat16, None, inplace=True)
            self.model.sparse_arch.embedding_bag_collection.embedding_bags.bfloat16()
            print('bf16 model ready...')
        elif args.use_int8:
            if calibration:
                if not self.debug:
                    print("Loading model fp32 weights...")
                    model.load_state_dict(torch.load(model_path))
                self.model = model
            else:
                del model
                print("Loading model int8 weights...")
                self.model = torch.jit.load(args.int8_model_dir)
            print('int8 model ready...')
        else:
            if not self.debug:
                print("Loading model weights...")
                model.load_state_dict(torch.load(model_path))
            self.model = model
            print('fp32 model ready...')
        self.model = self.model.cpu().share_memory()
        return self

    def predict(self, batch, ids = None):
        return self.batch_predict(*unpack(batch))

    def batch_predict(self, densex, index, offset):
        with torch.no_grad():
            out = self.model(densex, index, offset)
            return out
