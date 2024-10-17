#! /usr/bin/env python3
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

import argparse
import glob
import json
import os

import code.common.arguments as common_args
from code.common.utils import Tree
from scripts.utils import SimpleLogger
from code.common.log_parser import get_power_summary, read_loadgen_result_by_key, get_perf_regression_ratio, extract_mlperf_result_from_log

logger = SimpleLogger(indent_size=4, prefix="")


def get_result_summaries(log_dir):
    """
    Returns a summary of the results in a particular base log directory. Returns as a dict with the following structure:

        {
            <config_name>: {
                <benchmark name>: {
                    "performance": <result string>,
                    "accuracy": <result string>,
                },
    """
    summary = Tree()
    metadata_files = glob.glob(os.path.join(log_dir, "**", "metadata.json"), recursive=True)
    for fname in metadata_files:
        with open(fname) as f:
            _dat = json.load(f)
        keyspace = [_dat["config_name"], _dat["benchmark_full"]]
        if _dat["test_mode"] == "PerformanceOnly":
            keyspace.append("performance")
        elif _dat["test_mode"] == "AccuracyOnly":
            keyspace.append("accuracy")
        summary.insert(keyspace, _dat["summary_string"])
    return summary.tree  # Return the internal dict instead of the Tree object


def main():
    log_dir = common_args.parse_args(["log_dir"])["log_dir"]

    result_summaries = get_result_summaries(log_dir)
    logger.log(f"\n{'='*24} Result summaries: {'='*24}\n")
    for config_name in result_summaries:
        logger.log(f"{config_name}:")
        logger.inc_indent_level()
        for benchmark in result_summaries[config_name]:
            logger.log(f"{benchmark}:")
            logger.inc_indent_level()
            for k, v in result_summaries[config_name][benchmark].items():
                logger.log(f"{k}: {v}")
            logger.dec_indent_level()
        logger.dec_indent_level()
        logger.log("")

    # If this is a power run, we should print out the average power
    power_vals = get_power_summary(log_dir)
    if power_vals != None:
        logger.log(f"\n{'='*24} Power results: {'='*24}\n")
        for config_name in result_summaries:
            logger.log(f"{config_name}:")
            logger.inc_indent_level()
            for benchmark in result_summaries[config_name]:
                if len(power_vals) > 0:
                    avg_power = sum(power_vals) / len(power_vals)
                    logger.log(f"{benchmark}: avg power under load: {avg_power:.2f}W with {len(power_vals)} power samples")
                else:
                    logger.log(f"{benchmark}: cannot find any power samples in the test window. Is the timezone setting correct?")
            logger.dec_indent_level()
            logger.log("")

    # Process each run here
    metadata_files = glob.glob(os.path.join(log_dir, "**", "metadata.json"), recursive=True)
    for metadata_path in metadata_files:
        parts = metadata_path.split("/")
        # Ex: /work/build/logs/2022.03.23-06.05.20/DGX-A100_A100-SXM-80GBx1_TRT/bert-99/Server/mlperf_log_detail.txt
        system_id, benchmark, scenario = parts[-4:-1]
        result_dir = os.path.dirname(metadata_path)

        # Check if the current run is a PerformanceOnly run
        with open(metadata_path) as f:
            _dat = json.load(f)
        config_name = _dat["config_name"]

        # If we are running AccuracyOnly mode, no extra perf status is needed.
        # TODO: This needs to be fixed if "SubmissionRun" is ever used.
        if _dat["test_mode"] == "AccuracyOnly":
            return

        # Print the extra stats (for comparison against existing benchmark) here.
        print(f"\n{'=' * 24} Extra Perf Stats: {'=' * 24}\n")

        logger.log(f"{config_name}:")
        # The full log is stored under build/artifacts dir, make sure it's cloned.
        artifacts_log_dir = "build/artifacts/closed/NVIDIA/results"

        try:
            perf_number, metric, is_strict_match = extract_mlperf_result_from_log(result_dir, system_id, benchmark, scenario, False)
            # is_strict_match will be false if running MaxQ without measuring the power. Return perf stats only.
            if not is_strict_match:
                print(f"    WARNING: the returned perf metric {metric} does not match the official metric. Skipping the comparison...")
            else:
                regression_ratio, current_perf = get_perf_regression_ratio(perf_number, artifacts_log_dir, system_id, benchmark, scenario)

                if current_perf == 0:
                    print(f"    WARNING: Perf value of 0 found in results.")
                else:
                    print(f"    {metric}: {perf_number:.2f} is {regression_ratio:.2f} of the current results {current_perf:.2f}.")
        except FileNotFoundError as e:
            print(f"    FileNotFoundError: {e}. Non-NVIDIA users ignore this. NVIDIA users run `make pull_artifacts_repo`.")

        # Extra stats for Server
        if scenario == "Server":
            if 'llama2' in benchmark:
                ttft_99 = read_loadgen_result_by_key(result_dir, "result_first_token_99.00_percentile_latency_ns", None)
                target_ttft = read_loadgen_result_by_key(result_dir, "requested_server_ttft_latency", None)
                tpot_99 = read_loadgen_result_by_key(result_dir, "result_time_per_output_token_99.00_percentile_ns", None)
                target_tpot = read_loadgen_result_by_key(result_dir, "requested_server_tpot_latency", None)
                ttft_ratio = ttft_99 / target_ttft
                tpot_ratio = tpot_99 / target_tpot
                print(f"    Server 99-percentile TTFT {ttft_99:.0f} ns is {ttft_ratio:.2f} of the target_latency {target_ttft:.0f} ns")
                print(f"    Server 99-percentile TPOT {tpot_99:.0f} ns is {tpot_ratio:.2f} of the target_latency {target_tpot:.0f} ns")
            else:
                latency_99 = read_loadgen_result_by_key(result_dir, "result_99.00_percentile_latency_ns", None)
                target_latency = read_loadgen_result_by_key(result_dir, "requested_server_target_latency_ns", None)
                latency_ratio = latency_99 / target_latency
                print(f"    Server 99-percentile latency {latency_99:.0f} ns is {latency_ratio:.2f} of the target_latency {target_latency:.0f} ns")

                # If the offline result exists, print how much percentage
                try:
                    server_offline_ratio, offline_perf = get_perf_regression_ratio(perf_number, "results", system_id, benchmark, "Offline")
                    print(f"    Server QPS {perf_number:.0f} is {server_offline_ratio:.2f} of the current Offline QPS {offline_perf:.0f} under results/")
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    main()
