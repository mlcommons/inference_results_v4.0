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
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import logging
from pathlib import Path
from cuda import cudart
from importlib import import_module

from code.common import logging, run_command, args_to_string
from code.common.harness import BaseBenchmarkHarness
from code.common.systems.system_list import SystemClassifications

# dash in stable-diffusion-xl breaks traditional way of module import
Dataset = import_module("code.stable-diffusion-xl.tensorrt.dataset").Dataset
SDXLServer = import_module("code.stable-diffusion-xl.tensorrt.backend").SDXLServer
CUASSERT = import_module("code.stable-diffusion-xl.tensorrt.utilities").CUASSERT
CLIP = import_module("code.stable-diffusion-xl.tensorrt.network").CLIP
CLIPWithProj = import_module("code.stable-diffusion-xl.tensorrt.network").CLIPWithProj
UNetXL = import_module("code.stable-diffusion-xl.tensorrt.network").UNetXL
VAE = import_module("code.stable-diffusion-xl.tensorrt.network").VAE

import code.common.arguments as common_args
try:
    import mlperf_loadgen as lg
except:
    logging.warning("Loadgen Python bindings are not installed. Installing Loadgen Python bindings!")
    run_command("make build_loadgen")
    import mlperf_loadgen as lg

scenario_map = {
    "Offline": lg.TestScenario.Offline,
    "SingleStream": lg.TestScenario.SingleStream,
    "Server": lg.TestScenario.Server,
}
test_mode_map = {
    "PerformanceOnly": lg.TestMode.PerformanceOnly,
    "AccuracyOnly": lg.TestMode.AccuracyOnly,
    "SubmissionRun": lg.TestMode.SubmissionRun,
}
log_mode_map = {
    "AsyncPoll": lg.LoggingMode.AsyncPoll,
    "EndOfTestOnly": lg.LoggingMode.EndOfTestOnly,
    "Synchronous": lg.LoggingMode.Synchronous,
}


class SDXLHarness(BaseBenchmarkHarness):
    """SDXL harness."""

    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)
        custom_args = [
            "gpu_inference_streams",
            "gpu_copy_streams",
            "devices",
            "sdxl_batcher_time_limit",
            "use_graphs",
        ]
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS + custom_args

    def _get_harness_executable(self):
        """Return python command to SDXL harness python file."""
        return "code/stable-diffusion-xl/tensorrt/harness.py"

    def _construct_terminal_command(self, argstr):
        cmd = f"{self.executable.replace('code/stable-diffusion-xl/tensorrt/harness.py', 'python3 -m code.stable-diffusion-xl.tensorrt.harness')} {argstr}"
        return cmd

    def _build_custom_flags(self, flag_dict):
        s = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name
        return s

    def _get_engine_fpath(self, device_type, batch_size, model):
        # Override this function to pick up the right engine file based on model
        return f"{self.engine_dir}/{self.name}-{model.name}-{self.scenario.valstr()}-{device_type}-b{batch_size}-{model.precision}.{self.config_ver}.plan"

    # Currently, SDXL is using non-standard directory structure and filenames to store its engine files. Since
    # BaseBenchmarkHarness calls this function at the end of __init__, we have to put our custom directory structure setup
    # here instead of in __init__.
    def enumerate_engines(self):

        clip_with_proj_precision = 'fp32' if SystemClassifications.is_orin() else 'fp16'

        pipeline_models = [
            CLIP(max_batch_size=self.args["gpu_batch_size"], precision='fp16'),
            CLIPWithProj(max_batch_size=self.args["gpu_batch_size"], precision=clip_with_proj_precision),
            UNetXL(max_batch_size=self.args["gpu_batch_size"], precision=self.precision),
            VAE(max_batch_size=self.args["gpu_batch_size"], precision='fp32'),
        ]
        gpu_engine_list = []
        for model in pipeline_models:
            engine_fpath = self._get_engine_fpath("gpu", self.args["gpu_batch_size"], model)
            self.check_file_exists(engine_fpath)
            gpu_engine_list.append(engine_fpath)
        self.gpu_engine = ','.join(gpu_engine_list)


def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', '1'):
            return True
        elif v.lower() in ('false', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    # Test args
    parser.add_argument("--scenario", choices=["Offline", "Server", "SingleStream"], default="Offline")
    parser.add_argument("--test_mode", choices=["PerformanceOnly", "AccuracyOnly", "SubmissionRun"], default="PerformanceOnly")
    parser.add_argument("--model", default="stable-diffusion-xl")
    parser.add_argument("--server_num_issue_query_threads", type=int, default=0, help="Number of IssueQuery threads used in Server scenario")

    # QSL args
    parser.add_argument("--tensor_path", type=str)
    parser.add_argument("--performance_sample_count", type=int, default=5000)

    # SUT args
    parser.add_argument("--devices", type=str, default="all", help="Comma-separated numbered devices, use 'all' by default")
    parser.add_argument("--gpu_engines", type=str, default="all", help="Comma-separated TRT GPU engine files")
    parser.add_argument("--gpu_batch_size", type=int, default=1, help="Max Batch size to use for all devices and engines")
    parser.add_argument("--gpu_inference_streams", type=int, default=1, help="Number of streams for inference")
    parser.add_argument("--gpu_copy_streams", type=int, default=1, help="Number of streams for data transfer")
    parser.add_argument("--verbose", type=str2bool, default=False, help="SUT verbose logging")
    parser.add_argument("--verbose_nvtx", type=str2bool, default=False, help="SUT NVTX kDETAILED ProfilingVerbosity")
    parser.add_argument("--plugins", type=str, help="Comma-separated list of shared objects for plugins")
    parser.add_argument("--use_graphs", type=str2bool, default=False, help="Use cuda graph for inference")
    parser.add_argument("--sdxl_batcher_time_limit", type=float, default=-1, help="SDXL harness server scenario batcher time out threashold in seconds")

    # Config args
    parser.add_argument("--mlperf_conf_path", help="Path to mlperf.conf")
    parser.add_argument("--user_conf_path", help="Path to user.conf")

    # Log args
    parser.add_argument("--log_mode", type=str, default="AsyncPoll", help="Logging mode for Loadgen")
    parser.add_argument("--log_mode_async_poll_interval_ms", type=int, default=1000, help="Specify the poll interval for asynchrounous logging")
    parser.add_argument("--logfile_outdir", type=str, default='', help="Specify the existing output directory for the LoadGen logs")
    parser.add_argument("--logfile_prefix", type=str, default='', help="Specify the filename prefix for the LoadGen log files")
    parser.add_argument("--logfile_suffix", type=str, default='', help="Specify the filename suffix for the LoadGen log files")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    test_settings = lg.TestSettings()
    test_settings.scenario = scenario_map[args.scenario]
    test_settings.mode = test_mode_map[args.test_mode]
    # Load config
    test_settings.FromConfig(args.mlperf_conf_path, "stable-diffusion-xl", args.scenario)
    test_settings.FromConfig(args.user_conf_path, "stable-diffusion-xl", args.scenario)
    test_settings.server_coalesce_queries = True

    # Log settings
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.logfile_outdir
    log_output_settings.prefix = args.logfile_prefix
    log_output_settings.suffix = args.logfile_suffix
    Path(args.logfile_outdir).mkdir(parents=True, exist_ok=True)

    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.log_mode = log_mode_map[args.log_mode]
    log_settings.log_mode_async_poll_interval_ms = args.log_mode_async_poll_interval_ms

    # SUT settings
    gpu_engines = args.gpu_engines.split(',')
    assert len(gpu_engines) == 4, "SDXL harness requires 4 TRT GPU engines"
    if args.devices == "all":
        device_count = CUASSERT(cudart.cudaGetDeviceCount())
        devices = list(range(device_count))
    else:
        devices = [int(x) for x in args.devices.split(',')]

    ds = Dataset(args.tensor_path)
    qsl = lg.ConstructQSL(ds.caption_count, args.performance_sample_count, ds.load_query_samples, ds.unload_query_samples)
    server = SDXLServer(devices=devices,
                        dataset=ds,
                        gpu_engine_files=gpu_engines,
                        gpu_batch_size=args.gpu_batch_size,
                        gpu_inference_streams=args.gpu_inference_streams,
                        gpu_copy_streams=args.gpu_copy_streams,
                        use_graphs=args.use_graphs,
                        verbose=args.verbose,
                        verbose_nvtx=args.verbose_nvtx,
                        enable_batcher=(args.scenario == "Server"),
                        batch_timeout_threashold=args.sdxl_batcher_time_limit,
                        )

    logging.info("Start Warm Up!")
    server.warm_up()
    logging.info("Warm Up Done!")

    logging.info("Start Test!")
    lg.StartTestWithLogSettings(server.sut, qsl, test_settings, log_settings)
    server.finish_test()
    logging.info("Test Done!")

    logging.info("Destroying SUT...")
    lg.DestroySUT(server.sut)

    logging.info("Destroying QSL...")
    lg.DestroyQSL(qsl)


if __name__ == "__main__":
    main()
