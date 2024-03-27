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


import numpy as np
import os
import struct
from pathlib import Path

from code.common.systems.system_list import SystemClassifications
if not SystemClassifications.is_soc():
    from torchrec.datasets.criteo import CAT_FEATURE_COUNT

from code.common import logging

try:
    import mlperf_loadgen as lg
except:
    logging.info("Loadgen Python bindings are not installed. Functionality may be limited.")

CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE = [40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000, 3067956,
                                             405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000, 40000000, 40000000,
                                             590152, 12973, 108, 36]
CRITEO_SYNTH_MULTIHOT_SIZES = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]


class CriteoDay23Dataset:
    """Represents the Day 23 Criteo Dataset used for MLPerf Inference.
    """

    def __init__(self, data_dir: os.PathLike, mode: str = "full", precision="fp32"):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.precision = precision

        if precision == "fp32":
            self.dense_path = self.data_dir / "day_23_dense.npy"

        elif precision == "int8":
            self.dense_path = self.data_dir.parent / "int8" / "day_23_dense.npy"

        else:
            raise RuntimeError("Invalid precision")

        self.labels_path = self.data_dir / "day_23_labels.npy"
        self.sparse_multihot_dir = self.data_dir / "day_23_sparse_multi_hot_unpacked"
        self.sparse_concat_path = self.data_dir / "day_23_sparse_concatenated.npy"

        self.labels, self.dense, self.sparse = self.load_data()

    @property
    def size(self):
        if self.mode == "full":
            return len(self.dense)
        elif self.mode == "validation":
            n_samples = len(self.dense)
            half_index = int(n_samples // 2 + n_samples % 2)
            return half_index
        else:
            raise ValueError("invalid mode")

    def sparse_input_path(self, feat_idx: int):
        return self.sparse_multihot_dir / f"{feat_idx}.npy"

    def load_data(self):
        logging.info(f"Loading labels from {self.labels_path}")
        labels = np.load(self.labels_path)

        logging.info(f"Loading dense inputs from {self.dense_path}")
        dense_inputs = np.load(self.dense_path)

        if not self.sparse_concat_path.exists():
            logging.info(f"Loading sparse inputs from {self.sparse_multihot_dir}")
            sparse_inputs = []
            for i in range(CAT_FEATURE_COUNT):
                logging.info(f"\tLoading Categorical feature {i}...")
                sparse_inputs.append(np.load(self.sparse_input_path(i)))
            sparse_inputs = np.hstack(sparse_inputs)

        else:
            logging.info(f"Loading sparse inputs from {self.sparse_concat_path}")
            sparse_inputs = np.load(self.sparse_concat_path)

        assert sparse_inputs.shape == (178274637, sum(CRITEO_SYNTH_MULTIHOT_SIZES))
        return labels, dense_inputs, sparse_inputs

    def generate_val_map(self, val_map_dir: os.PathLike):
        """Generate sample indices for validation set. The validation set is the first half of Day 23, according to the
        reference implementation:
        https://github.com/mlcommons/inference/blob/master/recommendation/dlrm_v2/pytorch/python/multihot_criteo.py#L323
        """
        n_samples = len(self.dense)
        half_index = int(n_samples // 2 + n_samples % 2)
        val_map_txt = Path(val_map_dir) / "val_map.txt"
        with val_map_txt.open(mode='w') as f:
            for i in range(half_index):
                logging.info(f"{i:08d}", file=f)

    def generate_cal_map(self, cal_map_dir: os.PathLike):
        """Generate sample indices for calibration set. The calibration set, according the reference implementation at
        https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch#calibration-set
        is indices 89137319 through 89265318 inclusive.
        """
        lower_bound = 89137319
        upper_bound = 89265318
        assert (upper_bound - lower_bound + 1 == 128000)
        cal_map_txt = Path(cal_map_dir) / "cal_map.txt"
        with cal_map_txt.open(mode='w') as f:
            for i in range(lower_bound, upper_bound + 1):
                logging.info(f"{i:08d}", file=f)

    def get_batch(self, num_samples=None, indices=None):
        if indices is None:
            assert num_samples is not None
            indices = np.random.choice(self.size, size=num_samples, replace=False)

        batch = {
            "dense": self.dense[indices],
            "sparse": self.sparse[indices],
            "labels": self.labels[indices],
        }
        return batch

    def dump_concatenated_sparse_input(self):
        np.save(
            self.data_dir / "day_23_sparse_concatenated.npy",
            self.sparse
        )

    def dump_dense_input(self, precision: str = 'fp16'):
        # NOTE(vir): fp32 dense input file is downloaded not generated
        assert precision in ['fp16', 'int8']

        if precision == 'fp16':
            fp16_path = self.data_dir.parent / precision / 'day_23_dense.npy'
            fp16_path.parent.mkdir(parents=False, exist_ok=True)

            fp16_dense_input = self.dense.astype(np.float16)
            np.save(fp16_path, fp16_dense_input)

        elif precision == 'int8':
            int8_path = self.data_dir.parent / precision / 'day_23_dense.npy'
            int8_path.parent.mkdir(parents=False, exist_ok=True)

            calibration_scale = "3de5d364"  # from calibration.cache
            scale = 127 * struct.unpack('!f', bytes.fromhex(calibration_scale))[0]
            factor = 127.0 / scale

            transformed_dense = np.log(np.maximum(0, self.dense) + 1)
            int8_dense = np.minimum(127.0, np.maximum(-128.0, factor * transformed_dense)).astype(np.int8)
            np.save(int8_path, int8_dense)


def convert_sample_partition_to_npy(txt_path: os.PathLike):
    p = Path(txt_path)
    # Need to convert to a numpy file
    indices = [0]
    with p.open() as f:
        while (line := f.readline()):
            if len(line) == 0:
                continue

            start, end, count = line.split(", ")
            assert int(start) == indices[-1]
            indices.append(int(end))
    partition = np.array(indices, dtype=np.int32)
    np.save(p.with_suffix(".npy"), partition)
    return partition


class CriteoQSL:
    def __init__(self,
                 ds: CriteoDay23Dataset,
                 partition_path: os.PathLike = "/home/mlperf_inf_dlrmv2/criteo/day23/sample_partition.txt"):
        self.ds = ds
        self.partitions = self.load_partition(Path(partition_path))
        self.item_count = len(self.partitions) - 1
        self.active_ids = dict()

    def load_partition(self, p):
        partition = None
        if p.suffix == ".txt":
            partition = convert_sample_partition_to_npy(p)
        elif p.suffix == ".npy":
            partition = np.load(p)
        else:
            raise RuntimeError("Sample partition file must be a .txt or .npy file")
        return partition

    def unload_query_samples(self, sample_list):
        for sample_idx in sample_list:
            self.active_ids.pop(sample_idx)

    def load_query_samples(self, sample_list):
        # Criteo is weird, we need to translate sample indices to true dataset indices using the sample partition
        for sample_idx in sample_list:
            self.active_ids[sample_idx] = np.arange(self.partitions[sample_idx],
                                                    self.partitions[sample_idx + 1],
                                                    dtype=np.int32)

    def get_query_samples(self, sample_list):
        return {
            idx: self.active_ids[idx]
            for idx in sample_list
        }

    def as_loadgen_qsl(self, total_sample_count, performance_sample_count):
        return lg.ConstructQSL(total_sample_count,
                               performance_sample_count,
                               self.load_query_samples,
                               self.unload_query_samples)
