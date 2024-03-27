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
import intel_extension_for_pytorch as ipex

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

# pylint: disable=missing-docstring

# the datasets we support
SUPPORTED_DATASETS = {
    "debug":
        (multihot_criteo.MultihotCriteo, multihot_criteo.pre_process_criteo_dlrm, multihot_criteo.DlrmPostProcess(),
         {"randomize": 'total',  "memory_map": True}),
    "multihot-criteo-sample":
        (multihot_criteo.MultihotCriteo, multihot_criteo.pre_process_criteo_dlrm, multihot_criteo.DlrmPostProcess(),
         {"randomize": 'total',  "memory_map": True}),
    "multihot-criteo":
        (multihot_criteo.MultihotCriteo, multihot_criteo.pre_process_criteo_dlrm, multihot_criteo.DlrmPostProcess(),
         {"randomize": 'total',  "memory_map": True}),
}

# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "multihot-criteo",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
    "dlrm-debug-pytorch": {
        "dataset": "debug",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 128,
    },
    "dlrm-multihot-sample-pytorch": {
        "dataset": "multihot-criteo-sample",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
    "dlrm-multihot-pytorch": {
        "dataset": "multihot-criteo",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    }
}

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
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
    parser.add_argument("--scenario", default="SingleStream",
                        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())))
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--max-batchsize", type=int, help="max batch size in a single inference")
    parser.add_argument("--output", help="test results")
    parser.add_argument("--inputs", help="model inputs (currently not used)")
    parser.add_argument("--outputs", help="model outputs (currently not used)")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--find-peak-performance", action="store_true", help="enable finding peak performance pass")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--mlperf_conf", default="mlperf.conf", help="mlperf rules config")
    # file for user LoadGen settings such as target QPS
    parser.add_argument("--user_conf", default="user.conf", help="user config for user LoadGen settings such as target QPS")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--duration", type=int, help="duration in milliseconds (ms)")
    parser.add_argument("--target-qps", type=int, help="target/expected qps")
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--count-samples", type=int, help="dataset items to use")
    parser.add_argument("--count-queries", type=int, help="number of queries to use")
    parser.add_argument("--samples-per-query-multistream", default=8, type=int, help="query length for multi-stream scenario (in terms of aggregated samples)")
    # --samples-per-query-offline is equivalent to perf_sample_count
    parser.add_argument("--samples-per-query-offline", type=int, default=2048, help="query length for offline scenario (in terms of aggregated samples)")
    parser.add_argument("--samples-to-aggregate-fix", type=int, help="number of samples to be treated as one")
    parser.add_argument("--samples-to-aggregate-min", type=int, help="min number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-max", type=int, help="max number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-quantile-file", type=str, help="distribution quantile used to generate number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-trace-file", type=str, default="dlrm_trace_of_aggregated_samples.txt")
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--calibration", action="store_true", help="Whether calibration only for this run.")
    parser.add_argument("--int8-configure-dir", type=str, default="./int8_configure.json", help="int8 recipe location")
    parser.add_argument("--int8-model-dir", type=str, default="./dlrm_int8.pt", help="int8 model location")
    parser.add_argument("--use-int8", action="store_true", default=False)
    parser.add_argument("--use-bf16", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.numpy_rand_seed)

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)
    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args

def convert_int8(max_batchsize: int,
                 calibration: bool,
                 model: torch.nn.Module,
                 int8_configure_dir: str,
                 int8_model_dir: str,
                 ds):
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig, HistogramObserver
    from intel_extension_for_pytorch.quantization import prepare, convert
    qconfig = QConfig(
        activation=HistogramObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.qint8, bins=128, upsample_rate=384),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    )
    multi_hot = [3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1]
    dsx = torch.randn((max_batchsize, 13), dtype=torch.float)
    lsi = [torch.ones((max_batchsize * h), dtype=torch.long) for h in multi_hot]

    model = prepare(
        model,
        qconfig,
        example_inputs=(dsx, lsi),
        inplace=True
    )
    print('model', model)
    if calibration:
        # calibration first
        assert ds is not None
        count = ds.get_item_count()
        num_samples = 128000
        all_sample_ids = range(0, num_samples)
        ds.load_query_samples(all_sample_ids)
        for i in range(0, num_samples, max_batchsize):
            sample_s = i
            sample_e = min(num_samples, i + max_batchsize)
            densex, index, labels = ds.test_data.load_batch(range(sample_s, sample_e))
            model(densex, index)
        model.save_qconf_summary(qconf_summary=int8_configure_dir)
        print(f"calibration done and save to {int8_configure_dir}")
        # return model
    # else:
        # quantization second
        # model.load_qconf_summary(qconf_summary = int8_configure_dir)
        convert(model, inplace=True)
        model.eval()
        model = torch.jit.trace(model, (dsx, lsi), check_trace=True)
        model = torch.jit.freeze(model)
        model(dsx, lsi)
        model(dsx, lsi)
        # dump model third
        torch.jit.save(model, int8_model_dir)
        print("save model done")
        return model

def main():
    args = get_args()

    backend = get_backend(args.backend, args.dataset, args.use_gpu, debug=args.debug)
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]

    # --count-samples can be used to limit the number of samples used for testing
    ds = wanted_dataset(num_embeddings_per_feature=[40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,3],
                        data_path=args.dataset_path,
                        name=args.dataset,
                        pre_process=pre_proc,  # currently an identity function
                        count=args.count_samples,
                        samples_to_aggregate_fix=args.samples_to_aggregate_fix,
                        samples_to_aggregate_min=args.samples_to_aggregate_min,
                        samples_to_aggregate_max=args.samples_to_aggregate_max,
                        samples_to_aggregate_quantile_file=args.samples_to_aggregate_quantile_file,
                        samples_to_aggregate_trace_file=args.samples_to_aggregate_trace_file,
                        max_ind_range=args.max_ind_range,
                        **kwargs)
    # load model to backend
    model = backend.load(args, ds)
    # calibration
    if args.calibration:
        dlrm_model = model.model
        convert_int8(args.max_batchsize, args.calibration,
                     dlrm_model,
                     args.int8_configure_dir, args.int8_model_dir, ds)
        print("calibration done")
        return
    print('model load done')
    # #
    # # make one pass over the dataset to validate accuracy
    # #
    print('load dataset')
    count = ds.get_item_count()
    # # warmup
    results = np.zeros(count).astype(np.float32)
    targets = np.zeros(count).astype(np.float32)
    ds.load_query_samples(range(0, count, args.max_batchsize))
    print(f"load samples {count} done")
    # load data
    batchsize = args.max_batchsize
    # ipex.enable_onednn_fusion(False)
    for i in range(0, count, batchsize):
        sample_s = i
        sample_e = min(i + batchsize, count)
        densex, index, labels = ds.val_data.load_batch(range(sample_s, sample_e))
        t = time.time()
        r = backend.batch_predict(densex, index)
        inft = (time.time() - t) * 1000
        results[sample_s:sample_e] = r.detach().cpu().numpy()
        targets[sample_s:sample_e] = labels.detach().cpu().float().numpy()
        # results.append(r[0].detach().cpu().numpy())
        # targets.append(batch.labels.detach().cpu().float().numpy())
        res_np = results[0:sample_e].copy()
        tgt_np = targets[0:sample_e].copy()
        roc_auc = ipex._C.roc_auc_score(torch.tensor(tgt_np).reshape(-1), torch.tensor(res_np).reshape(-1))
        print(f'roc_auc of {roc_auc[0]} dur {inft} ms')
        # if i >= batchsize *2:
        #    sys.exit(0)

if __name__ == "__main__":
    main()