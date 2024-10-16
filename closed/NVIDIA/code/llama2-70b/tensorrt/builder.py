#!/usr/bin/env python3
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

from __future__ import annotations
from os import PathLike
from pathlib import Path
from typing import Optional

import importlib.util
import time
import subprocess
import sys

from nvmitten.constants import Precision
from nvmitten.nvidia.builder import (TRTBuilder,
                                     MLPerfInferenceEngine,
                                     LegacyBuilder)
from nvmitten.pipeline import Operation
from nvmitten.utils import logging

from code.common.mitten_compat import ArgDiscarder


class LLAMA2EngineBuilderOp(TRTBuilder,
                            MLPerfInferenceEngine,
                            Operation,
                            ArgDiscarder):
    """LLAMA2 offloads the engine building to the implementation in the TRT-LLM examples. This class still is a sublass
    of nvmitten.nvidia.builder.TRTBuilder solely for consistency and to use the same class initializer.
    """

    @classmethod
    def immediate_dependencies(cls):
        # TODO: Integrate dataset scripts as deps
        return None

    def __init__(self,
                 config_ver: str = "default",
                 model_path: str = "build/models/Llama2/Llama-2-70b-chat-hf",
                 # Generated from 2/6/2024 TRTLLM devdrop
                 fp8_quant_model_path: str = "build/models/Llama2/fp8-quantized-ammo/llama2-70b-chat-hf-tp2pp1-fp8/",
                 trt_llm_path: str = "build/TRTLLM",
                 # Override the normal default values
                 workspace_size: int = 60 << 30,
                 # Benchmark specific values
                 batch_size: int = 16,
                 use_fp8: bool = False,
                 use_inflight_batching: bool = False,
                 tensor_parallelism: int = 2,
                 pipeline_parallelism: int = 1,
                 max_num_tokens: int = 16384,
                 **kwargs):
        super().__init__(workspace_size=workspace_size,
                         **kwargs)

        self.config_ver = config_ver
        self.batch_size = batch_size
        self.model_path = model_path
        self.use_fp8 = use_fp8
        # TODO: A temporary WAR for H200
        if tensor_parallelism == 1:
            fp8_quant_model_path = "build/models/Llama2/fp8-quantized-ammo/llama2-70b-chat-hf-tp1pp1-fp8-02072024/"
        self.fp8_quant_model_path = Path(fp8_quant_model_path)
        self.use_inflight_batching = use_inflight_batching
        self.dtype = "fp8" if self.use_fp8 else "fp16"
        self.need_quantization = False
        # TODO: Parameterize these
        self.tp_size = tensor_parallelism
        self.pp_size = pipeline_parallelism
        self.world_size = self.tp_size * self.pp_size
        assert self.world_size > 0
        # required for generating engine
        self.max_num_tokens = max_num_tokens

        # If the local model path exists, use it instead of the network storage one
        local_model_path = Path("/raid/data/mlperf-llm/Llama-2-70b-chat-hf")
        if local_model_path.exists():
            logging.info(f"using local Llama2 model from {local_model_path}")
            self.model_path = str(local_model_path)

        # https://gitlab-master.nvidia.com/ftp/tekit/-/tree/main/examples/llama?ref_type=heads#fp8-post-training-quantization
        # Please refer to README.md for quantization.
        if self.use_fp8:
            if not self.fp8_quant_model_path.exists():
                raise FileNotFoundError(f"Could not locate Llama2 fp8 quantized checkpoint model path: ({self.fp8_quant_model_path}). Please check README.md")
                # TODO If the default model does not exist, build a quantized model locally
                # self.need_quantization = True
        else:
            raise NotImplementedError(f"Only fp8 supported as of now. Set precision to FP16 and use_fp8 to True. If you need to perform quantization, please refer to README.md")

        self.trt_llm_path = Path(trt_llm_path)

        if self.precision == "fp8":
            raise NotImplementedError(f"To enable FP8 precision, set precision to FP16 and use_fp8 to True.")
        elif self.precision != Precision.FP16:
            raise NotImplementedError(f"Precision {self.precision} is not supported yet.")

    def build_quantized_model(self):
        """
        Use AMMO to build the quantized FP8 model for TRTLLM usage
        """
        quantize_script_path = self.trt_llm_path / "examples/llama/quantize.py"
        if not quantize_script_path.exists():
            raise FileNotFoundError(f"Could not locate Llama2 quantize script ({quantize_script_path}), please run `make clone_trt_llm`")

        quantize_dir = self.fp8_quant_model_path.parent
        quantize_dir.mkdir(parents=True, exist_ok=True)
        flags = [
            f"--model_dir={self.model_path}",
            "--dtype=float16",
            "--qformat=fp8",
            f"--export_path={quantize_dir}",
            "--calib_size=512"
        ]

        quantize_cmd = [sys.executable, str(quantize_script_path.absolute())] + flags
        logging.info(f"Building Llama2 FP8 quantization model in {quantize_dir}.")
        logging.info(f"Command: {' '.join(quantize_cmd)}")

        stdout_log = self.fp8_quant_model_path.with_suffix(".stdout")
        stderr_log = self.fp8_quant_model_path.with_suffix(".stderr")

        tik = time.time()
        ret = subprocess.run(quantize_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        tok = time.time()

        # Save stdout and stderr logs
        with stdout_log.open(mode='w') as f:
            f.write(ret.stdout)
        with stderr_log.open(mode='w') as f:
            f.write(ret.stderr)

        if ret.returncode != 0:
            logging.error(ret.stderr)
            raise RuntimeError(f"Quantization recipe failed. Logs dumped to {stderr_log}.")

        logging.info(f"Quantized model completes in {tok-tik}s. Saved to {self.fp8_quant_model_path}")

    def build_engine(self,
                     network: trt.INetworkDefinition,
                     builder_config: trt.IBuilderConfig,
                     batch_size: int,
                     engine_fpath: PathLike):
        """Builds the engine via the Llama2 builder provided in the TRT-LLM examples.

        Args:
            network: Unused.
            builder_config: Unused.
            batch_size (int): Batch size to build the engine for.
            engine_fpath (PathLike): Location to save the engine file(s) to.
        """
        if not importlib.util.find_spec("tensorrt_llm"):
            raise ModuleNotFoundError("Cannot import tensorrt_llm module. Please run `make build_trt_llm`.")

        if self.need_quantization:
            self.build_quantized_model()

        build_script = Path("tensorrt_llm/commands/build.py")
        build_script_path = self.trt_llm_path / build_script
        if not build_script_path.exists():
            raise FileNotFoundError(f"Could not locate Llama2 build script ({build_script_path}), please run `make clone_trt_llm`")

        engine_fpath = Path(engine_fpath)
        if engine_fpath.is_file():
            logging.warning(f"{engine_fpath} already exists. This file will be overwritten")
        engine_dir = engine_fpath.parent
        engine_dir.mkdir(parents=True, exist_ok=True)

        flags = ["--gpt_attention_plugin=float16",
                 f"--max_batch_size={self.batch_size}",
                 "--max_input_len=1024",
                 "--max_output_len=1024",
                 "--max_beam_width=1",
                 f"--max_num_tokens={self.max_num_tokens}",
                 f"--output_dir={str(engine_dir.absolute())}",
                 f"--checkpoint_dir={str(self.fp8_quant_model_path.absolute())}",
                 "--context_fmha=enable",
                 "--remove_input_padding=enable",
                 "--paged_kv_cache=enable",
                 "--strongly_typed",
                 "--use_custom_all_reduce=enable",
                 ]

        if self.world_size > 1:
            flags.append(f"--workers={self.world_size}")

        build_cmd = [sys.executable, "-m", '.'.join(build_script.with_suffix('').parts)] + flags
        logging.info(f"Building Llama2 engine in {engine_dir}.")
        logging.info(f"Command executing in build/TRTLLM dir: {' '.join(build_cmd)}")


        stdout_log = engine_fpath.with_suffix(".stdout")
        stderr_log = engine_fpath.with_suffix(".stderr")

        tik = time.time()
        ret = subprocess.run(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd="build/TRTLLM")
        tok = time.time()

        # Save stdout and stderr logs
        with stdout_log.open(mode='w') as f:
            f.write(ret.stdout)
        with stderr_log.open(mode='w') as f:
            f.write(ret.stderr)

        if ret.returncode != 0:
            logging.error(ret.stderr)
            raise RuntimeError(f"Engine build failed. Logs dumped to {engine_dir}.")

        logging.info(f"Engine build complete in {tok-tik}s. Saved to {engine_fpath}")

    def run(self, scratch_space, dependency_outputs):
        engine_dir = self.engine_dir(scratch_space)
        # We distinguish engines by the dir name for TRTLLM
        # TODO: Add TP/PP
        engine_dir = engine_dir / f"bs{self.batch_size}-{self.config_ver}-tp{self.tp_size}-pp{self.pp_size}"
        # For TRT-LLM, the engine name is fixed.
        engine_name = f"llama_float16_tp{self.tp_size}_rank0.engine"
        engine_fpath = engine_dir / engine_name
        self.build_engine(None, None, self.batch_size, engine_fpath)


class LLAMA2(LegacyBuilder):
    """Temporary spoofing class to wrap around Mitten to adhere to the old API.
    """

    def __init__(self, args):
        super().__init__(LLAMA2EngineBuilderOp(**args))
