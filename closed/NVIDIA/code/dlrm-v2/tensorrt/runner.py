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


from pathlib import Path
from code.common import logging
from code.common.runner import EngineRunner
from tqdm import tqdm
from torchrec.datasets.criteo import INT_FEATURE_COUNT, CAT_FEATURE_COUNT, DEFAULT_CAT_NAMES
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

import argparse
import array
try:
    import mlperf_loadgen as lg
except:
    print("Loadgen Python bindings are not installed. Functionality may be limited.")
import numpy as np
import os
import sklearn.metrics
import tensorrt as trt
import torch

from .criteo import CriteoDay23Dataset, CriteoQSL
from .model import DLRMv2_Model, DLRMv2TRTNetwork


def build_engine(batch_size: int = 4096, save_to: os.PathLike = None):
    dlrm_network = DLRMv2TRTNetwork(batch_size)

    builder_config = dlrm_network.builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    # builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    builder_config.clear_flag(trt.BuilderFlag.TF32)
    builder_config.set_flag(trt.BuilderFlag.FP16)

    for input_idx in range(dlrm_network.network.num_inputs):
        input_shape = dlrm_network.network.get_input(input_idx).shape
        input_shape[0] = batch_size
        dlrm_network.network.get_input(input_idx).shape = input_shape

    engine = dlrm_network.builder.build_engine(dlrm_network.network, builder_config)
    engine_inspector = engine.create_engine_inspector()
    layer_info = engine_inspector.get_engine_information(trt.LayerInformationFormat.ONELINE)
    logging.info("========= TensorRT Engine Layer Information =========")
    logging.info(layer_info)

    del engine_inspector

    logging.debug("Serializing and saving engine")
    if save_to is None:
        save_to = Path(f"dlrm_v2_bs_{batch_size}_test.engine")
    else:
        save_to = Path(save_to)
    buf = engine.serialize()
    with save_to.open(mode='wb') as f:
        f.write(buf)


class TestHarness:

    def __init__(self, engine_path: os.PathLike, batch_size, precision="fp32"):
        self.batch_size = batch_size
        self.precision = precision

        # Create TRT engine runner
        self.engine = EngineRunner(engine_path)

        # Load dataset
        self.ds = CriteoDay23Dataset("/home/mlperf_inf_dlrmv2/criteo/day23/fp32",
                                     mode="validation",
                                     precision=self.precision)

    def sparse_to_kjt(self, sparse, batch_size: int = None):
        # https://github.com/mlcommons/inference/blob/master/recommendation/dlrm_v2/pytorch/python/multihot_criteo.py#L382
        if batch_size is None:
            batch_size = sparse[0].shape[0]

        lengths = torch.ones((CAT_FEATURE_COUNT * batch_size), dtype=torch.int32)
        feature_sizes = [arr.shape[-1] for arr in sparse]
        for i, size in enumerate(feature_sizes):
            _start = i * batch_size
            _end = (i + 1) * batch_size
            lengths[_start:_end] = size
        offsets = torch.cumsum(torch.concat((torch.tensor([0]), lengths)), dim=0)
        length_per_key = [batch_size * size for size in feature_sizes]
        offset_per_key = torch.cumsum(torch.concat((torch.tensor([0]), torch.tensor(length_per_key))), dim=0)
        values = torch.concat([torch.from_numpy(feature).flatten() for feature in sparse])
        index_per_key = {key: i for i, key in enumerate(DEFAULT_CAT_NAMES)}
        return KeyedJaggedTensor(keys=DEFAULT_CAT_NAMES,
                                 values=values,
                                 lengths=lengths,
                                 offsets=offsets,
                                 stride=batch_size,
                                 length_per_key=length_per_key,
                                 offset_per_key=offset_per_key.tolist(),
                                 index_per_key=index_per_key)

    def do_batch(self, batch):
        # Create input to TRT engine
        sparse_input = np.ascontiguousarray(np.hstack(batch['sparse']), dtype=np.int32)
        if self.precision == "fp32":
            num_dtype = np.float32
        elif self.precision == "int8":
            num_dtype = np.int8
        else:
            raise RuntimeError("Invalid precision")

        numeric_input = np.ascontiguousarray(batch["dense"], dtype=num_dtype)
        trt_input = [numeric_input, sparse_input]

        # Run TRT engine inference
        outputs = self.engine(trt_input, self.batch_size)[0]
        assert len(outputs) == self.batch_size

        return outputs

    def run_inference(self, n_batches: int = None):
        # Enumerate all batches that need to be run by index
        iters = []
        start = 0
        end = None  # Explicitly define to not rely on Python variable leaking
        while start < self.ds.size:
            if n_batches is not None and len(iters) >= n_batches:
                break

            end = start + self.batch_size
            if end > self.ds.size:
                end = self.ds.size
            batch_bounds = (start, end)
            iters.append(batch_bounds)
            start = end

        n_batches = len(iters)
        n_correct = 0
        n_total = 0
        all_outputs = np.zeros((end,))
        all_labels = np.zeros((end,), dtype=np.int32)

        for i, (start, end) in enumerate((pbar := tqdm(iters))):
            eff_batch_size = end - start
            eff_indices = np.arange(start, end)

            _indices = np.zeros((self.batch_size,), dtype=np.int32)
            _indices[:eff_batch_size] = eff_indices
            batch = self.ds.get_batch(indices=_indices)

            labels = batch["labels"][:eff_batch_size].flatten()
            outputs = self.do_batch(batch)[:eff_batch_size]

            # Do class prediction via rounding
            pred = np.array(outputs.round(), dtype=np.int32)
            n_correct += (pred == labels).sum()
            n_total += eff_batch_size
            acc = n_correct / n_total * 100.0

            all_outputs[start:end] = outputs
            all_labels[start:end] = labels

            pbar.set_description(f"[{i+1}/{n_batches}] {n_correct}/{n_total} ({acc:.02f}%)")

        auc = sklearn.metrics.roc_auc_score(all_labels, all_outputs)
        print("AUC score:", auc)
        return all_outputs, all_labels


class ResultsAggregator:
    def __init__(self):
        self.outputs = []
        self.labels = []
        self.ids = []
        self.n_correct = 0
        self.n_total = 0

    def reset(self):
        self.outputs = []
        self.labels = []
        self.ids = []
        self.n_correct = 0
        self.n_total = 0

    def add(self, outputs, labels, ids):
        pred = np.array(outputs.round(), dtype=np.int32)
        self.n_correct += (pred == labels).sum()
        self.n_total += len(outputs)
        self.outputs.append(outputs)
        self.labels.append(labels)
        self.ids.append(ids)


class DLRMv2SUT:
    def __init__(self, engine_path: os.PathLike, batch_size):
        self.engine_path = engine_path
        self.batch_size = batch_size
        self.runner = TestHarness(engine_path, batch_size)
        self.qsl = CriteoQSL(self.runner.ds)
        self.results_aggregator = ResultsAggregator()

    def flush_query(self):
        return

    def issue_query(self, query_samples):
        indices = [q.index for q in query_samples]
        query_ids = [q.id for q in query_samples]
        n_samples = len(query_samples)

        idx_remap = self.qsl.get_query_samples(indices)
        current_batch = np.zeros((self.batch_size,), dtype=np.int32)
        batch_offset = 0
        current_ids = []

        def _run_batch():
            nonlocal current_batch, batch_offset, current_ids

            batch = self.runner.ds.get_batch(indices=current_batch)
            labels = batch["labels"][:batch_offset].flatten()
            outputs = self.runner.do_batch(batch)[:batch_offset]
            self.results_aggregator.add(outputs, labels, current_ids)

            # Submit query response
            # From reference impl:
            # https://github.com/mlcommons/inference/blob/master/recommendation/dlrm_v2/pytorch/python/main.py#L306
            # https://github.com/mlcommons/inference/blob/master/recommendation/dlrm_v2/pytorch/python/multihot_criteo.py#L443-L460

            # Unsure why this is needed, but my guess is it's to make sure the array.array objects aren't cleaned up by
            # GC after each loop iter.
            _refs = []
            resp = []

            for qid in current_ids:
                s = qid["start_idx"]
                e = s + qid["size"]
                sample_outputs = outputs[s:e].astype(np.float32)
                sample_labels = labels[s:e].astype(np.float32)
                combined = np.vstack((sample_outputs, sample_labels)).T
                resp_array = array.array('B', combined.tobytes())
                _refs.append(resp_array)

                bi = resp_array.buffer_info()
                resp.append(lg.QuerySampleResponse(qid["id"], bi[0], bi[1]))
            lg.QuerySamplesComplete(resp)

            # Reset
            current_batch.fill(0)
            batch_offset = 0
            current_ids = []

        for i, (sample_idx, true_indices) in enumerate(idx_remap.items()):
            partition_size = len(true_indices)

            # If we cannot add this sample, run the batch
            if partition_size + batch_offset > len(current_batch):
                _run_batch()

            current_batch[batch_offset:batch_offset + partition_size] = true_indices
            current_ids.append({"id": query_ids[i],
                               "start_idx": batch_offset,
                                "size": partition_size})
            batch_offset += partition_size

        if batch_offset > 0:
            _run_batch()

    def as_loadgen_sut(self):
        return lg.ConstructSUT(self.issue_query, self.flush_query)


def run_loadgen_test(engine_path: os.PathLike,
                     max_batch_size: int,
                     output_path: os.PathLike,
                     scenario: lg.TestScenario,
                     target_qps: float,
                     mlperf_conf_path: os.PathLike = "build/inference/mlperf.conf",
                     user_conf_path: os.PathLike = None,
                     mode: lg.TestMode = lg.TestMode.PerformanceOnly):
    # DLRMv2 is a DC only workload. scenario can only be Offline or Server
    if scenario not in [lg.TestScenario.Offline, lg.TestScenario.Server]:
        raise RuntimeError(f"Invalid scenario: {scenario}")

    print("Creating test parameters...")
    settings = lg.TestSettings()
    settings.FromConfig(mlperf_conf_path, "dlrm_v2", scenario.name)
    if user_conf_path is not None:
        settings.FromConfig(user_conf_path, "dlrm_v2", scenario.name)
    settings.scenario = scenario
    settings.mode = mode

    if scenario == lg.TestScenario.Offline:
        settings.offline_expected_qps = target_qps
    elif scenario == lg.TestScenario.Server:
        settings.server_target_qps = target_qps

    print("Setting output parameters...")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    output_settings = lg.LogOutputSettings()
    output_settings.outdir = str(output_path.absolute())
    output_settings.prefix = "mlperf_log_"
    output_settings.copy_summary_to_stdout = True

    log_settings = lg.LogSettings()
    log_settings.log_output = output_settings

    print("Creating SUT Wrapper...")
    sut_wrapper = DLRMv2SUT(engine_path, max_batch_size)

    sut = sut_wrapper.as_loadgen_sut()
    total_sample_count = sut_wrapper.qsl.item_count
    perf_sample_count = total_sample_count
    qsl = sut_wrapper.qsl.as_loadgen_qsl(total_sample_count, perf_sample_count)

    print("Starting test...")
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)

    print("Test done.")

    if mode == lg.TestMode.AccuracyOnly:
        all_labels = np.concatenate(sut_wrapper.results_aggregator.labels)
        all_outputs = np.concatenate(sut_wrapper.results_aggregator.outputs)
        auc = sklearn.metrics.roc_auc_score(all_labels, all_outputs)
        print("AUC:", auc)

    sample_size_total = 0
    n_samples_run = 0
    for batch_ids in sut_wrapper.results_aggregator.ids:
        for item in batch_ids:
            sample_size_total += item["size"]
            n_samples_run += 1
    print("Average number of items per sample based on partition:", sample_size_total / n_samples_run)

    print("Cleaning up...")
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action",
                        default="generate_engine",
                        choices=["generate_engine", "run_harness"],
                        help="Action to run")
    parser.add_argument("--batch_size", type=int, default=4096, help="Maximum batch size to do inference on")
    parser.add_argument("--engine_path", help="Path to save engine to. (Default: 'dlrm_v2_bs_<BATCHSIZE>_test.engine')")
    parser.add_argument("--accuracy", action="store_true", help="If set, runs in accuracy mode.")
    parser.add_argument("--scenario",
                        choices=["Offline", "Server"],
                        default="Offline",
                        help="Valid scenario for DLRMv2. Either Offline or Server. (Default: 'Offline')")
    parser.add_argument("--expected_qps",
                        type=float,
                        default=300.0,
                        help="Offline expected qps or server target qps. (Default: 300.0)")
    parser.add_argument("--loadgen_out_dir",
                        default="dlrm_v2_loadgen_out",
                        help="Output directory for loadgen logs. (Default: 'dlrm_v2_loadgen_out')")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    batch_size = args.batch_size
    engine_path = args.engine_path
    if not engine_path:
        engine_path = f"dlrm_v2_bs_{batch_size}_test.engine"

    mode = lg.TestMode.PerformanceOnly
    if args.accuracy:
        mode = lg.TestMode.AccuracyOnly

    scenario = lg.TestScenario.Offline if args.scenario == "Offline" else lg.TestScenario.Server

    if args.action == "generate_engine":
        build_engine(batch_size, save_to=engine_path)
    elif args.action == "run_harness":
        run_loadgen_test(engine_path, batch_size, args.loadgen_out_dir, scenario, args.expected_qps, mode=mode)
