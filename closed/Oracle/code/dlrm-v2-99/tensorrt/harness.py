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

from pathlib import Path
from tqdm import tqdm
import os
import re
import requests
from requests.exceptions import HTTPError

from code.common import logging, dict_get, run_command, args_to_string
from code.common.harness import BaseBenchmarkHarness
from code.common.systems.system_list import SystemClassifications
import code.common.arguments as common_args

from .criteo import CriteoDay23Dataset, convert_sample_partition_to_npy


SAMPLE_PARTITION_TRACE_ZENODO_URL = "https://zenodo.org/record/3941795/files/dlrm_trace_of_aggregated_samples.txt?download=1"


class DLRMv2Harness(BaseBenchmarkHarness):
    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)
        custom_args = [
            "gpu_copy_streams",
            "complete_threads",
            "sample_partition_path",
            "warmup_duration",
            "gpu_inference_streams",
            "num_staging_threads",
            "num_staging_batches",
            "max_pairs_per_staging_thread",
            "gpu_num_bundles",
            "check_contiguity",
            "start_from_device",
            "use_jemalloc",
            "qsl_numa_override",
            "use_batcher_thread_per_device",
        ]
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS + custom_args

        self.generate_required_files()

    def _get_harness_executable(self):
        return "./build/bin/harness_dlrm_v2"

    def _build_custom_flags(self, flag_dict):
        # Handle use_jemalloc
        self.use_jemalloc = dict_get(flag_dict, "use_jemalloc", False)
        flag_dict['use_jemalloc'] = None

        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name
        if self.system_id == 'L4x1':
            argstr += " --eviction_last=0.5"

        return argstr

    def generate_required_files(self):
        if SystemClassifications.is_soc():
            logging.warning("SoC does not support DLRMv2! Bypass DLRMv2 on SoC systems...")
            return

        tensor_paths = self.args["tensor_path"].split(',')
        assert len(tensor_paths) == 2, "DLRMv2 requires 2 input tensor files"

        dense_input_filepath = Path(tensor_paths[0])
        sparse_input_filepath = Path(tensor_paths[1])

        # load dataset to generate missing files only if needed, takes a long time
        if (not dense_input_filepath.exists()) or (not sparse_input_filepath.exists()):
            ds = CriteoDay23Dataset("/home/mlperf_inf_dlrmv2/criteo/day23/fp32", mode="full")

        # generate lower precision dense input files if needed
        if not dense_input_filepath.exists():
            # Create the file
            logging.info(f"Dense input file does not exist for precision: {self.args['input_dtype']}. Generating...")
            ds.dump_dense_input(self.args['input_dtype'])
            logging.info(f"Generated dense input file for precision: {self.args['input_dtype']}.")
        else:
            logging.info(f"Found dense input file for precision: {self.args['input_dtype']}.")

        # generate sparse input concat file if needed
        if not sparse_input_filepath.exists():
            # Create the file
            logging.info("Coalesced sparse input file does not exist. Generating...")
            ds.dump_concatenated_sparse_input()
            logging.info("Generated coalesced sparse inputs.")
        else:
            logging.info("Found coalesced sparse input file.")

        # Get sample_partition file
        sample_partition_path = Path(self.args["sample_partition_path"])
        txt_path = sample_partition_path.with_suffix(".txt")
        if not sample_partition_path.exists() or not txt_path.exists():
            logging.info("Downloading sample partition file...")
            # Required file is numpy file. Save as a .txt first

            # TODO: Use nvmitten.pipeline.resource.Resource. This file is currently non-functional, and it is too close
            # to deadline to make a new Mitten release.
            resp = requests.get(SAMPLE_PARTITION_TRACE_ZENODO_URL, stream=True, allow_redirects=True)
            resp.raise_for_status()

            file_size = int(resp.headers.get("content-length", -1))
            pbar = tqdm(total=file_size, leave=False)
            with txt_path.open(mode='wb') as handle:
                for data in resp.iter_content(chunk_size=4096):
                    handle.write(data)
                    pbar.update(len(data))
            pbar.close()

            logging.info("Converting to npy file...")
            convert_sample_partition_to_npy(txt_path)
        else:
            logging.info("Found sample partition file.")
