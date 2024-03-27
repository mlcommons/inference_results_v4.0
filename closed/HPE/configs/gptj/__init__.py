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
    benchmark = Benchmark.GPTJ

    tensor_path = "build/preprocessed_data/cnn_dailymail_tokenized_gptj/input_ids_padded.npy,build/preprocessed_data/cnn_dailymail_tokenized_gptj/input_lengths.npy"
    llm_gen_config_path = "code/gptj/tensorrt/generation_config.json"
    use_token_latencies = False
    precision = "fp16"
    input_dtype = "int32"
    input_format = "linear"
    use_graphs = False
    coalesced_tensor = True
    max_num_tokens = 4096
