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

from code.common import logging, args_to_string
from code.common.harness import BaseBenchmarkHarness
import code.common.arguments as common_args


class LLAMA2Harness(BaseBenchmarkHarness):
    """Llama v2 harness."""

    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)
        custom_args = [
            "gpu_inference_streams",
            "gpu_copy_streams",
            "devices",
            "tensor_parallelism",
            "pipeline_parallelism",
            "use_inflight_batching",
            "kvcache_free_gpu_mem_frac",
            "enable_sort",
            "gpu_rank_map",
            "llm_gen_config_path",
            "use_token_latencies",
        ]
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS + custom_args

    def _get_harness_executable(self):
        """Return path to Llama2 harness binary."""
        return "./build/bin/harness_llm"

    def _build_custom_flags(self, flag_dict):
        s = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name
        return s

    def _get_engine_fpath(self, device_type, batch_size):
        # Override this function to pick up the right engine file
        return f"{self.engine_dir}/bs{batch_size}-{self.config_ver}-tp{self.tp_size}-pp{self.pp_size}/rank0.engine"
