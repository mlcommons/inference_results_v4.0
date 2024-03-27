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
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import Benchmark
from configs.configuration import BenchmarkConfiguration


class GPUBaseConfig(BenchmarkConfiguration):
    benchmark = Benchmark.DLRMv2
    precision = "int8"

    model_path = "/home/mlperf_inf_dlrmv2/model/model_weights"
    mega_table_npy_file = '/home/mlperf_inf_dlrmv2/model/embedding_weights/mega_table_int8.npy'
    mega_table_scales_npy_file = '/home/mlperf_inf_dlrmv2/model/embedding_weights/mega_table_scales.npy'
    row_frequencies_npy_filepath = '/home/mlperf_inf_dlrmv2/criteo/day23/row_frequencies.npy'
    reduced_precision_io = 2

    input_dtype = "fp16"
    input_format = "linear"
    tensor_path = "/home/mlperf_inf_dlrmv2/criteo/day23/fp16/day_23_dense.npy,/home/mlperf_inf_dlrmv2/criteo/day23/fp32/day_23_sparse_concatenated.npy"
    sample_partition_path = "/home/mlperf_inf_dlrmv2/criteo/day23/sample_partition.npy"
    # map_path = "data_maps/criteo_multihot/val_map.txt"

    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    coalesced_tensor = True
    use_graphs = False
