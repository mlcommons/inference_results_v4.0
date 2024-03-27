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


__doc__ = """Scripts that tests the accuracy of Llama2-70B model, using either engines generated
from TRT-LLM or Pytorch reference implementation
"""

import argparse
import ctypes
import json
import evaluate
import nltk
import os
import time
import subprocess
import pickle
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Dict, Tuple, List, Optional

import torch
from transformers import LlamaTokenizerFast, AutoModelForCausalLM

import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

# from mpi4py import MPI

from code.common import logging
from code.common.constants import TRT_LOGGER, Scenario

# prep ROUGE
nltk.download("punkt", quiet=False)

# Global vars
G_LLAMA70B_MAX_INPUT_SEQLEN = 1024
G_LLAMA70B_MAX_OUTPUT_SEQLEN = 1024
G_LLAMA70B_MAX_SEQLEN = 4096
G_LLAMA70B_NUM_LAYERS = 80
G_LLAMA70B_VOCAB_SIZE = 32000

G_OPENORCA_CALSET_PATH = None
G_OPENORCA_CALMAP_PATH = None
G_OPENORCA_VALSET_PATH = "build/preprocessed_data/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
G_OPENORCA_VALMAP_PATH = None
G_OPENORCA_CALIBRATION_CACHE_PATH = None

G_EOS_TOKEN = 2
G_PAD_TOKEN = 2


def prepare_openorca():
    # Load from OpenORCA
    with open(G_OPENORCA_VALSET_PATH, 'rb') as fh:
        orca_df = pickle.load(fh)

    source_ids = orca_df['tok_input'].tolist()
    source_lengths = orca_df['tok_input_length'].tolist()
    target_ids = orca_df['tok_output'].tolist()
    target_texts = orca_df["output"].tolist()

    logging.info(
        f"Loaded {len(source_lengths)} samples from {G_OPENORCA_VALSET_PATH}")
    return source_ids, source_lengths, target_ids, target_texts


def postprocess_text(preds, targets):
    # Post-process output texts for ROUGE evaluation
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


def calculate_rouge_score(preds, targets):
    logging.info("Calculating ROUGE scores...")
    metric = evaluate.load("rouge")
    preds, targets = postprocess_text(preds, targets[0:len(preds)])
    result = metric.compute(
        predictions=preds, references=targets, use_stemmer=True, use_aggregator=False)
    result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    result["gen_len"] = np.sum(prediction_lens)
    result["gen_num"] = len(preds)

    return result


def read_trtllm_config(config_path: Path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # builder config
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config']['pipeline_parallel']
    world_size = tp_size * pp_size
    num_heads = config['builder_config']['num_heads'] // tp_size
    num_kv_heads = config['builder_config']['num_kv_heads'] // tp_size
    hidden_size = config['builder_config']['hidden_size'] // tp_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    quant_mode = QuantMode(config['builder_config']['quant_mode'])
    tokens_per_block = config['plugin_config']['tokens_per_block']
    precision = config['builder_config']['precision']
    use_custom_all_reduce = config['plugin_config']['use_custom_all_reduce']

    # plugin config
    gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin'] == 'float16'
    remove_input_padding = config['plugin_config']['remove_input_padding']
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    use_context_fmha_for_generation = config['plugin_config']['use_context_fmha_for_generation']

    # sanity check
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    model_config = ModelConfig(vocab_size=vocab_size,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               num_kv_heads=num_kv_heads,
                               hidden_size=hidden_size,
                               gpt_attention_plugin=gpt_attention_plugin,
                               remove_input_padding=remove_input_padding,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               quant_mode=quant_mode,
                               dtype=precision,
                               use_custom_all_reduce=use_custom_all_reduce,
                               use_context_fmha_for_generation=use_context_fmha_for_generation,
                               )

    return model_config, tp_size, pp_size, precision


def get_engine_name(model, dtype, tp_size, pp_size, rank):
    if pp_size == 1:
        return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)
    return '{}_{}_tp{}_pp{}_rank{}.engine'.format(model, dtype, tp_size, pp_size, rank)


class TRTLLMRunner:
    """
    TRT-LLM runner class for LLM, specifically for TRT-LLM-generated engines.
    Encapsulate the preparation of generation sessions and inference loops
    """

    def __init__(
        self,
        engine_dir: str,
        batch_size: int,
        gen_args: dict,
        model_config,
        runtime_rank: int,
        verbose: bool = False,
    ):
        self.engine_dir = Path(engine_dir)
        self.batch_size = batch_size
        self.gen_args = gen_args
        self.model_config = model_config
        self.verbose = verbose
        self.num_beams = self.gen_args['num_beams']
        self.max_output_length = self.gen_args['max_output_length']
        self.top_k = self.gen_args['top_k']
        self.top_p = self.gen_args['top_p']
        self.temperature = self.gen_args['temperature']
        self.tp_size = self.gen_args['tp_size']
        self.pp_size = self.gen_args['pp_size']
        self.precision = self.gen_args['precision']
        self.runtime_rank = runtime_rank

        tensorrt_llm.logger.set_level("error" if not self.verbose else "verbose")

        logging.info(f"Rank{self.runtime_rank}:: Model configs: {self.model_config}")
        world_size = self.tp_size * self.pp_size

        # temporary check
        assert self.pp_size == 1, "Error: infer.py does not support pipeline parallelism yet"
        assert self.runtime_rank == tensorrt_llm.mpi_rank()
        assert self.num_beams == 1, "Llama2 70B only supports beam size == 1 for now"

        runtime_mapping = tensorrt_llm.Mapping(world_size,
                                               self.runtime_rank,
                                               tp_size=self.tp_size,
                                               pp_size=self.pp_size)

        self.device_id = self.runtime_rank % runtime_mapping.gpus_per_node
        logging.info(f"Rank{self.runtime_rank}:: Setting device to {self.device_id}")
        torch.cuda.set_device(self.device_id)

        self.sampling_config = SamplingConfig(end_id=G_EOS_TOKEN,
                                              pad_id=G_PAD_TOKEN,
                                              num_beams=self.num_beams,
                                              top_k=self.top_k,
                                              top_p=self.top_p,
                                              temperature=self.temperature)
        logging.info(f"Rank{self.runtime_rank}:: Loaded sampling config: {self.sampling_config}")

        # Reading from engines
        engine_file = self.engine_dir / get_engine_name("llama", self.precision, self.tp_size, self.pp_size, self.runtime_rank)
        assert engine_file.exists(), f"Rank{self.runtime_rank}:: Cannot find engine file: {engine_file}"
        logging.info(f"Rank{self.runtime_rank}:: Loading engine from file {engine_file}...")
        with open(engine_file, 'rb') as f:
            engine_buffer = f.read()

        self.decoder = tensorrt_llm.runtime.GenerationSession(self.model_config,
                                                              engine_buffer,
                                                              runtime_mapping,
                                                              debug_mode=False,
                                                              debug_tensors_to_save=None)

        logging.info(f"Rank{self.runtime_rank}:: Loading engine from file {engine_file} done, decoder is set up")

    def __call__(self, inputs):
        """
        Entry point of the LLM inference, which calls the decode function.
        """
        # torch.cuda.set_device(self.device_id)
        processed_output_ids, output_lengths = None, None

        input_ids = torch.tensor(np.concatenate(inputs[0]), dtype=torch.int32, device="cuda").unsqueeze(0)
        input_lengths = torch.tensor(inputs[1], dtype=torch.int32, device="cuda")
        max_input_length = torch.max(input_lengths).item()
        max_output_length = self.max_output_length
        self.decoder.setup(input_lengths.size(0),
                           max_input_length,
                           max_output_length,
                           beam_width=self.num_beams)

        outputs = self.decoder.decode(input_ids,
                                      input_lengths,
                                      self.sampling_config,
                                      streaming=False,
                                      output_sequence_lengths=True,
                                      return_dict=True)

        if self.runtime_rank == 0:
            output_ids = outputs['output_ids']
            output_lengths = outputs['sequence_lengths']

            # output_ids shape is [BS, beam, seqlen]
            # Copy the output portion from output_ids for decoding
            processed_output_ids = torch.full((input_lengths.size(0), self.max_output_length), G_EOS_TOKEN)
            for batch_idx in range(input_lengths.size(0)):
                output_begin = input_lengths[batch_idx]
                output_end = output_lengths[batch_idx][0]
                seqlen = output_end - output_begin
                processed_output_ids[batch_idx][:seqlen] = output_ids[batch_idx][0][output_begin:output_end]

        # MPI.COMM_WORLD.Barrier()
        # print(f"Rank {self.runtime_rank} passed the Barrier")

        return processed_output_ids, output_lengths


class TRTTester:
    """
    Wrapper class to encapsulate the TRT tester util functions.
    """

    def __init__(self,
                 mpi_rank: int,
                 engine_dir: str,
                 batch_size: int,
                 precision: str,
                 model_path: str,
                 fp8_quantized_model_path: Optional[str] = ".",
                 num_beams: Optional[int] = 1,
                 use_dla: Optional[bool] = False,
                 skip_engine_build: Optional[bool] = False,
                 engine_build_only: Optional[bool] = False,
                 verbose: Optional[bool] = False,
                 fp8: Optional[bool] = True,
                 top_k: Optional[int] = 1,
                 top_p: Optional[float] = 0.0,
                 temperature: Optional[float] = 0.0,
                 ):
        """
        Test GPT model through the TRT path.
        """
        self.batch_size = batch_size
        self.verbose = verbose
        self.engine_dir = Path(engine_dir)
        self.cache_file = G_OPENORCA_CALIBRATION_CACHE_PATH
        self.precision = precision
        self.model_path = model_path
        self.fp8_quantized_model_path = fp8_quantized_model_path
        self.fp8 = fp8
        self.engine_build_only = engine_build_only
        self.runtime_rank = mpi_rank

        # Tokenizer
        self.tokenizer = LlamaTokenizerFast.from_pretrained(self.model_path)

        # TensorRT engine related fields
        if use_dla:
            self.dla_core = 0
        else:
            self.dla_core = None

        # Initiate the plugin and logger
        # Use the global singleton, which is required by TRT.
        self.logger = TRT_LOGGER
        self.logger.min_severity = trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO
        trt.init_libnvinfer_plugins(self.logger, "")

        logging.info(f"Loading plugins from the plugin .so")
        try:
            import tensorrt_llm
        except:
            logging.error("TRT-LLM is not installed, please run make clone_trt_llm && make build_trt_llm")
            raise
        tensorrt_llm.plugin._load_plugin_lib()

        if self.runtime_rank == 0:
            if skip_engine_build:
                engine_dir_path = Path(engine_dir)
                engine_json_path = engine_dir_path / "config.json"
                if not (engine_dir_path.exists() and engine_json_path.exists()):
                    raise RuntimeError(
                        f"Cannot find engine file in the {engine_dir}. Please generate engine.")
            else:
                self.create_trt_engine()

        config_path = self.engine_dir / "config.json"
        self.model_config, self.tp_size, self.pp_size, self.precision = read_trtllm_config(config_path)

        self.gen_args = {
            "max_output_length": G_LLAMA70B_MAX_OUTPUT_SEQLEN,
            "num_beams": num_beams,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "precision": self.precision
        }

        # Create runner wrapper from the engine file
        self.runner = TRTLLMRunner(
            engine_dir,
            self.batch_size,
            self.gen_args,
            self.model_config,
            self.runtime_rank,
            self.verbose
        )

    def create_trt_engine(self):
        # Build the engine by calling TRT-LLM.
        engine_dir = Path(self.engine_dir)
        engine_dir.mkdir(parents=True, exist_ok=True)

        builder_path = Path("build/TRTLLM/examples/llama2/build.py")
        if not builder_path.exists():
            raise RuntimeError(f"TRTLLM not found under build/TRTLLM, please run make clone_trt_llm")
        build_cmd = [
            "python",
            "build/TRTLLM/examples/llama2/build.py",
            "--dtype=float16",
            "--log_level=verbose",
            "--enable_context_fmha",
            "--remove_input_padding",
            "--use_inflight_batching",
            "--paged_kv_cache",
            "--use_gpt_attention_plugin=float16",
            "--use_layernorm_plugin=float16",
            "--use_gemm_plugin=float16",
            f"--max_batch_size={self.batch_size}",
            f"--max_input_len={G_LLAMA70B_MAX_INPUT_SEQLEN}",
            f"--max_output_len={G_LLAMA70B_MAX_OUTPUT_SEQLEN}",
            f"--vocab_size={G_LLAMA70B_VOCAB_SIZE}",
            f"--max_beam_width={self.gen_args['num_beams']}",
            f"--output_dir={engine_dir}",
            f"--model_dir={self.model_path}",
            f"--world_size={self.gen_args['world_size']}",
            f"--tp_size={self.gen_args['tp_size']}",
            f"--pp_size={self.gen_args['pp_size']}",
        ]
        if self.fp8:
            assert Path(self.fp8_quantized_model_path).exists(), "FP8 quantized model not found"
            build_cmd += [
                "--enable_fp8",
                f"--quantized_fp8_model_path={self.fp8_quantized_model_path}",
                "--fp8_kv_cache",
                "--strongly_typed",
            ]
        if self.world_size > 1:
            build_cmd += [
                "--parallel_build",
            ]

        logging.info(f"Building engine in {engine_dir}, command: {' '.join(build_cmd)}")
        tik = time.time()
        ret = subprocess.run(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if ret.returncode != 0:
            raise RuntimeError(f"Engine build fails! stderr: {ret.stderr}")
        tok = time.time()

        logging.info(f"Engine built complete and took {tok-tik}s.")

        if self.engine_build_only:
            logging.info(f"--engine_build_only specified, exiting...")
            exit(0)

    def run_inference(self, num_samples):
        """
        Perform the actual inference and calculate ROUGE accuracy
        """
        results = dict()

        source_ids, source_lengths, target_ids, target_texts = prepare_openorca()

        # Start batch inferencing
        batch_idx = 0
        pred_ids = []
        total_time = 0.0
        for start_idx in range(0, num_samples, self.batch_size):
            # Print Progress
            if self.runtime_rank == 0:
                if batch_idx % 10 == 0:
                    logging.info(
                        f"Processing batch: {batch_idx} sample: {start_idx}/{num_samples}")
                start_time = time.time()
            end_idx = min(start_idx + self.batch_size, num_samples)

            input_batch = [source_ids[start_idx:end_idx], source_lengths[start_idx:end_idx]]

            # Input batch: [input ID: (BS, max_seq_len), input length: (BS)]
            output_ids, output_lengths = self.runner(input_batch)

            if self.runtime_rank == 0:
                duration = time.time() - start_time
                logging.info(
                    f"Batch {batch_idx} >>> inference time: {duration:.2f}s")
                total_time += duration
                pred_ids.extend(output_ids)

            batch_idx += 1

        if self.runtime_rank == 0:
            logging.info(
                f"Total inference time for {num_samples} samples: {total_time:.2f}s")
            pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            results = calculate_rouge_score(pred_texts, target_texts)

        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine_dir",
                        help="Specify where the Llama2 engine file is",
                        default="build/engines/DGX-H100_H100-SXM-80GBx8/llama2/Offline/bs32-custom_k_99_MaxP-tp2-pp1",
                        required=False)
    parser.add_argument("--model_path",
                        help="Specify the dir path containing the Llama2 PyTorch model checkpoint",
                        default="build/models/Llama2/Llama-2-70b-chat-hf")
    parser.add_argument("--fp8_quantized_model_path",
                        help="Specify the dir path containing FP8 quantized Llama2 model",
                        default="build/models/Llama2/Llama-2-70b-chat-hf")
    parser.add_argument("--batch_size",
                        help="batch size that fits in the device memory",
                        type=int,
                        default=1)
    parser.add_argument("--num_beams",
                        help="The maximum beam width of the decoding op.",
                        type=int,
                        default=1)
    parser.add_argument("--num_samples",
                        help="Number of samples to run. We use total 24576 samples from the Open ORCA validation set",
                        type=int,
                        default=24576)
    parser.add_argument("--trt_precision",
                        help="Run TensorRT in the specified precision",
                        choices=("float16"),
                        default="float16")
    parser.add_argument("--use_dla",
                        help="Use DLA instead of gpu",
                        action="store_true")
    parser.add_argument("--skip_engine_build",
                        help="Skip the TRT engine build phase if possible.",
                        action="store_true")
    parser.add_argument("--engine_build_only",
                        help="Build the engine and skip the testing part",
                        action="store_true")
    parser.add_argument("--verbose",
                        help="verbose output",
                        action="store_true")
    args = parser.parse_args()

    # TRT Tester
    my_rank = tensorrt_llm.mpi_rank()
    logging.info(
        f"Rank{my_rank}:: Running accuracy test for Llama2 70B using engines in the {args.engine_dir} ...")
    tester = TRTTester(
        tensorrt_llm.mpi_rank(),
        args.engine_dir,
        args.batch_size,
        args.trt_precision,
        args.model_path,
        args.fp8_quantized_model_path,
        args.num_beams,
        args.use_dla,
        args.skip_engine_build,
        args.engine_build_only,
        args.verbose)

    # run inference
    rouge = tester.run_inference(args.num_samples)
    if my_rank == 0:
        logging.info(f"TRT ROUGE Score: {rouge}")

    # TRT
    # To run the TRT tester:
    # mpirun -n 2 --allow-run-as-root python3 -m code.llama2-70b.tensorrt.infer --engine_dir build/engines/DGX-H100_H100-SXM-80GBx8/llama2/Offline/bs32-custom_k_99_MaxP-tp2-pp1 --num_samples=16 --batch_size=16 --skip_engine_build


if __name__ == "__main__":
    main()
