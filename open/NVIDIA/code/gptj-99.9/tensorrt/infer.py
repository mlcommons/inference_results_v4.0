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


__doc__ = """Scripts that tests the accuracy of GPTJ-6B model, using either engines generated
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
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Dict, Tuple, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from code.common import logging
from code.common.constants import TRT_LOGGER, Scenario

# Intel reference implementation, uses 2048-128 as the maximum input seqlen
G_GPTJ6B_MAX_INPUT_SEQLEN = 1919
G_GPTJ6B_MAX_OUTPUT_SEQLEN = 128
G_GPTJ6B_MAX_SEQLEN = 2047
G_GPTJ6B_NUM_LAYERS = 28
G_GPTJ6B_VOCAB_SIZE = 50401
G_CNNDAILYMAIL_CALSET_PATH = None
G_CNNDAILYMAIL_CALMAP_PATH = None
G_CNNDAILYMAIL_VALSET_PATH = "/home/mlperf_inference_data/data/cnn-daily-mail/cnn_eval.json"
G_CNNDAILYMAIL_VALMAP_PATH = None
G_CNNDAILYMAIL_CALIBRATION_CACHE_PATH = None

# Prompt for GPTJ model input
G_PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)

nltk.download("punkt", quiet=False)


def prepare_tokenizer(checkpoint_path, padding_side="left"):
    """
    Prepare the tokenizer for the cnn dailymail
    """
    logging.info(f"Initializing tokenizer from {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=G_GPTJ6B_MAX_SEQLEN,
        padding_side=padding_side,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def preprocess_cnndailymail():
    # Load from CNN dailymail
    with open(G_CNNDAILYMAIL_VALSET_PATH, 'r') as fh:
        list_data_dict = json.load(fh)

    sources = [G_PROMPT_INPUT.format_map(
        example) for example in list_data_dict]
    targets = [f"{example['output']}" for example in list_data_dict]

    logging.info(
        f"Loaded {len(sources)} samples from {G_CNNDAILYMAIL_VALSET_PATH}")
    return sources, targets


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
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    quant_mode = QuantMode(config['builder_config']['quant_mode'])
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    dtype = config['builder_config']['precision']

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               remove_input_padding=remove_input_padding,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               quant_mode=quant_mode,
                               dtype=dtype)

    max_input_len = config['builder_config']['max_input_len']

    return model_config, world_size, dtype, max_input_len


class TRTLLMRunner:
    """
    TRT-LLM runner class for LLM, specifically for TRT-LLM-generated engines.
    Encapsulate the preparation of generation sessions and inference loops
    """

    def __init__(
        self,
        engine_file: str,
        batch_size: int,
        tokenizer,
        gen_kwargs: dict,
        decoding_step: bool = False,
        verbose: bool = False,
    ):
        self.engine_file = Path(engine_file)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.gen_kwargs = gen_kwargs
        self.decoding_step = decoding_step
        self.verbose = verbose
        self.num_beams = self.gen_kwargs['num_beams']
        self.device = torch.device(f'cuda:0')  # Hard code to use device 0

        tensorrt_llm.logger.set_level("error" if not self.verbose else "verbose")

        config_path = self.engine_file.parent / "config.json"
        self.model_config, world_size, dtype, _ = read_trtllm_config(config_path)
        logging.info(f"Model configs: {self.model_config}")
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size,
                                               runtime_rank,
                                               tp_size=world_size)
        logging.info(f"Setting torch device to {runtime_rank % runtime_mapping.gpus_per_node}")
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

        self.sampling_config = SamplingConfig(end_id=tokenizer.eos_token_id,
                                              pad_id=tokenizer.pad_token_id,
                                              num_beams=self.num_beams,
                                              min_length=self.gen_kwargs['min_new_tokens'])
        logging.info(f"Loaded sampling config: {self.sampling_config}")

        # Reading from engines
        logging.info(f"Loading engine from file {self.engine_file}...")
        with open(self.engine_file, 'rb') as f:
            engine_buffer = f.read()

        self.decoder = tensorrt_llm.runtime.GenerationSession(self.model_config,
                                                              engine_buffer,
                                                              runtime_mapping)

    def __call__(self, inputs):
        """
        Entry point of the LLM inference, which calls the decode function.
        """
        input_ids = inputs[0]
        input_lengths = inputs[1]
        max_input_length = torch.max(input_lengths).item()
        self.decoder.setup(input_lengths.size(0),
                           max_input_length,
                           self.gen_kwargs["max_new_tokens"],
                           beam_width=self.num_beams)

        outputs = self.decoder.decode(input_ids,
                                      input_lengths,
                                      self.sampling_config,
                                      output_sequence_lengths=True,
                                      return_dict=True)
        output_ids = outputs['output_ids']
        sequence_lengths = outputs['sequence_lengths']
        torch.cuda.synchronize()

        cum_log_probs = self.decoder.cum_log_probs if self.num_beams > 1 else None

        # output_ids shape is [BS, beam, seqlen]
        # Copy the output portion from output_ids for decoding
        output_processed = torch.full((input_lengths.size(0), self.gen_kwargs["max_new_tokens"]), self.tokenizer.eos_token_id)
        max_seqlen = torch.max(sequence_lengths)
        for batch_idx in range(input_lengths.size(0)):
            output_begin = input_lengths[batch_idx]
            output_end = sequence_lengths[batch_idx][0]
            seqlen = output_end - output_begin
            output_processed[batch_idx][:seqlen] = output_ids[batch_idx][0][output_begin:output_end]
            # Alternative way of getting output
            # seqlen = min(max_seqlen - output_begin, self.gen_kwargs["max_new_tokens"])
            # output_processed[batch_idx][:seqlen] = output_ids[batch_idx][0][output_begin:output_begin+seqlen]

        return output_processed


class TRTTester:
    """
    Wrapper class to encapsulate the TRT tester util functions.
    """

    def __init__(self,
                 engine_file: str,
                 batch_size: int,
                 precision: str,
                 pyt_ckpt_path: str,
                 num_beams: Optional[int] = 1,
                 use_dla: Optional[bool] = False,
                 skip_engine_build: Optional[bool] = False,
                 engine_build_only: Optional[bool] = False,
                 decoding_step: Optional[bool] = False,
                 verbose: Optional[bool] = False,
                 ):
        """
        Test GPT model through the TRT path.
        """
        self.batch_size = batch_size
        self.verbose = verbose
        self.engine_file = engine_file
        self.cache_file = G_CNNDAILYMAIL_CALIBRATION_CACHE_PATH
        self.precision = precision
        self.pyt_ckpt_path = pyt_ckpt_path
        self.decoding_step = decoding_step

        self.gen_kwargs = {
            "early_stopping": True,
            "max_new_tokens": G_GPTJ6B_MAX_OUTPUT_SEQLEN,
            "min_new_tokens": 30,
            "num_beams": num_beams,
        }

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

        # Initiate tokenizer, use right padding for TRT-LLM
        self.tokenizer = prepare_tokenizer(self.pyt_ckpt_path, "right")

        if not skip_engine_build:
            self.create_trt_engine()
        else:
            if not os.path.exists(engine_file):
                raise RuntimeError(
                    f"Cannot find engine file {engine_file}. Please supply the onnx file or engine file.")

        # Create runner wrapper from the engine file
        self.runner = TRTLLMRunner(
            self.engine_file,
            self.batch_size,
            self.tokenizer,
            self.gen_kwargs,
            self.decoding_step,
            self.verbose
        )

    def create_trt_engine(self):
        # Build the engine by calling TRT-LLM.
        engine_dir = Path(os.path.dirname(self.engine_file))
        engine_dir.mkdir(parents=True, exist_ok=True)

        if not os.path.exists("build/TRTLLM/examples/gptj/build.py"):
            raise RuntimeError(f"TRTLLM not found under build/TRTLLM, please run make clone_trt_llm")
        build_cmd = [
            "python", "build/TRTLLM/examples/gptj/build.py", "--dtype=float16",
            "--log_level=verbose", "--enable_context_fmha",
            "--use_gpt_attention_plugin=float16",
            "--use_layernorm_plugin=float16", "--max_batch_size=32",
            f"--max_input_len={G_GPTJ6B_MAX_INPUT_SEQLEN}",
            f"--max_output_len={G_GPTJ6B_MAX_OUTPUT_SEQLEN}", f"--vocab_size={G_GPTJ6B_VOCAB_SIZE}",
            f"--max_beam_width={self.gen_kwargs['num_beams']}",
            f"--output_dir={engine_dir}",
            f"--model_dir={self.pyt_ckpt_path}",
        ]
        logging.info(f"Building engine in {engine_dir}, command: {' '.join(build_cmd)}")
        tik = time.time()
        ret = subprocess.run(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if ret.returncode != 0:
            raise RuntimeError(f"Engine build fails! stderr: {ret.stderr}")
        tok = time.time()

        logging.info(f"Engine built complete and took {tok-tik}s.")

        if engine_build_only:
            logging.info(f"--engine_build_only specified, exiting...")
            exit(0)

    def run_inference(self, num_samples):
        """
        Perform the actual inference and calculate ROUGE accuracy
        """
        sources, targets = preprocess_cnndailymail()

        # Start batch inferencing
        batch_idx = 0
        preds = []
        total_time = 0.0
        for start_idx in range(0, num_samples, self.batch_size):
            # Print Progress
            if batch_idx % 20 == 0:
                logging.info(
                    f"Processing batch: {batch_idx} image: {start_idx}/{num_samples}")

            start_time = time.time()
            # Tokenize a batch and record the seqlen info
            end_idx = min(start_idx + self.batch_size, num_samples)
            input_batch = self.tokenizer.batch_encode_plus(sources[start_idx:end_idx], return_tensors="pt",
                                                           padding=True, truncation=True,
                                                           max_length=G_GPTJ6B_MAX_INPUT_SEQLEN)

            input_ids = input_batch.input_ids.to(torch.int32).cuda()
            attention_mask = input_batch.attention_mask.numpy().astype(np.int32)
            input_real_seqlen = torch.from_numpy(np.sum(attention_mask, axis=1).astype(np.int32)).cuda()

            # If the remove_input_padding is enabled, remove the input padding and concatenate them
            if self.runner.model_config.remove_input_padding:
                concat_input_ids = torch.zeros([1, torch.sum(input_real_seqlen)], dtype=torch.int32).cuda()
                idx = 0
                for i, seqlen in enumerate(input_real_seqlen):
                    concat_input_ids[0][idx:idx + seqlen] = input_ids[i][:seqlen]
                    idx += seqlen
                input_ids = concat_input_ids

            logging.debug(
                f"input_batch shape: {input_ids.shape}, mask shape: {attention_mask.shape} input_real_seqlen: {input_real_seqlen}")

            # Input shape:
            # input batch: (BS, max_seq_len),
            # input_real_seqlen (BS)
            output_ids = self.runner(
                [input_ids, input_real_seqlen]
            )

            duration = time.time() - start_time
            logging.info(
                f"Batch {batch_idx} >>> inference time: {duration:.2f}s")
            total_time += duration

            # Decode the output
            top_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            logging.debug(f"output_texts:\n{top_texts}")
            preds.extend(output_ids)
            batch_idx += 1

        logging.info(
            f"Total inference time for {num_samples} samples: {total_time:.2f}s")

        # De-tokenize the ids into text
        logging.info(f"Decoding tokenized ids into text...")
        if self.decoding_step:
            print(preds)
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.decoding_step:
            print(preds)
        results = calculate_rouge_score(preds, targets)

        return results


class PytorchTester:
    """
    Pytorch reference tester
    """

    def __init__(
        self,
        pyt_ckpt_path: str,
        batch_size: Optional[int] = 1,
        precision: Optional[str] = "bf16",
        max_input_seq_len: Optional[int] = 1600,
        num_beams: Optional[int] = 1,
        decoding_step: Optional[bool] = False,
    ):
        self.device = torch.device("cuda:0")
        self.batch_size = batch_size
        self.pyt_ckpt_path = pyt_ckpt_path
        self.max_input_seq_len = max_input_seq_len
        self.decoding_step = decoding_step

        # Set torch dtype and mixed precision flag.
        self.amp_enabled = True
        if precision == "bf16":
            self.amp_dtype = torch.bfloat16
        elif precision == "fp16":
            self.amp_dtype = torch.float16
        elif precision == "fp32":
            self.amp_enabled = False
            self.amp_dtype = torch.float32
        else:
            raise NotImplementedError(f"Unknown dtype {precision}")

        logging.info(
            f"Loading GPTJ-6B tokenizer and checkpoint from {self.pyt_ckpt_path}")
        self.tokenizer = prepare_tokenizer(self.pyt_ckpt_path)
        self.model_kwargs = {
            "torch_dtype": self.amp_dtype
        }
        self.gen_kwargs = {
            "early_stopping": True,
            "max_new_tokens": G_GPTJ6B_MAX_OUTPUT_SEQLEN,
            "min_new_tokens": 30,
            "num_beams": num_beams,
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            self.pyt_ckpt_path, **self.model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.model = self.model.to(memory_format=torch.channels_last)

    def run_inference(self, num_samples):
        """
        Perform the inference steps in pytorch. Note the auto-regressive part is implicit for pytorch
        AutoModelForCausalLM.

        Pytorch maximum batch size is 16 on A100, costing 74GB device memory
        """
        # Read from cnn daily mail and pre-process the data
        with open(G_CNNDAILYMAIL_VALSET_PATH, 'r') as fh:
            list_data_dict = json.load(fh)

        sources, targets = preprocess_cnndailymail()

        # Loop through all the CNN dailymail inputs
        time_stat_dict = {'encoding': 0.0,
                          'inference': 0.0, 'decoding': 0.0, 'total': 0.0}
        preds = []
        batch_idx = 0
        for start_idx in range(0, num_samples, self.batch_size):
            # Print Progress
            if batch_idx % 20 == 0:
                logging.info(
                    f"Processing batch: {batch_idx} sample: {start_idx}/{num_samples}")

            end_idx = min(start_idx + self.batch_size, num_samples)

            start_time = time.time()

            # Padding behavior:
            #   1) pad short seq to the maximum within the batch
            #   2) Truncate long seq to 1919
            input_batch = self.tokenizer.batch_encode_plus(
                sources[start_idx:end_idx],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=G_GPTJ6B_MAX_INPUT_SEQLEN
            )

            encoding_time = time.time()

            for t in input_batch:
                if torch.is_tensor(input_batch[t]):
                    input_batch[t] = input_batch[t].to(self.device)

            # Record the input padded seqlen so we know where the output starts from.
            input_batch_lengths = [x.shape[0] for x in input_batch.input_ids]

            with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=self.amp_enabled, dtype=self.amp_dtype if self.amp_enabled else None):
                output_batch = self.model.generate(
                    **input_batch,
                    **self.gen_kwargs,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            inference_time = time.time()

            # Truncate the input portion of the outputs
            output_batch_response_only = []
            for data, source_len in zip(output_batch, input_batch_lengths):
                output_batch_response_only.append(data[source_len:])

            # Decode the output into text, and append to the output list
            # print(f"output_batch_response_only: {output_batch_response_only}")
            output_text = self.tokenizer.batch_decode(
                output_batch_response_only, skip_special_tokens=True)

            decoding_time = time.time()

            # Collect time for parts
            encoding_duration = encoding_time - start_time
            inference_duration = inference_time - encoding_time
            decoding_duration = decoding_time - inference_time
            total_duration = decoding_time - start_time
            time_stat_dict['encoding'] += encoding_duration
            time_stat_dict['inference'] += inference_duration
            time_stat_dict['decoding'] += decoding_duration
            time_stat_dict['total'] += total_duration
            logging.info(f"Batch {batch_idx} >>> encoding: {encoding_duration:.2f}s, infer: {inference_duration:.2f}s, " +
                         f"decoding: {decoding_duration:.2f}s, total: {total_duration:.2f}s")

            # Break down the generate() cycle into inference loops for debugging purpose.
            if self.decoding_step:
                if self.gen_kwargs['num_beams'] > 1:
                    raise NotImplementedError(f"Num_beams > 1 is not supported for per-step decoding. Exiting...")
                # Make a copy of the gen_kargs so they are not contaminated
                copy_model_kwargs = deepcopy(self.model_kwargs)
                input_ids = input_batch.input_ids
                copy_model_kwargs["attention_mask"] = input_batch.attention_mask
                # Track seqs that are still running
                unfinished_sequence = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
                eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)
                pred_text = ["" for i in range(input_ids.shape[0])]
                for i in range(self.gen_kwargs["max_new_tokens"]):
                    model_inputs = self.model.prepare_inputs_for_generation(
                        input_ids=input_ids,
                        **copy_model_kwargs
                    )
                    # outputs is a CausalLMOutputWithPast type, with following fields:
                    # loss, logits, past_key_values, hidden_states, attentions.
                    # logits and past_key_values are useful in inference.
                    outputs = self.model(**model_inputs)
                    next_token_logits = outputs.logits[:, -1, :]

                    # If the min token length is not met, force generate new tokens by
                    # changing the logits of EOS to -inf.
                    # See https://github.com/huggingface/transformers/blob/f49a3453caa6fe606bb31c571423f72264152fce/src/transformers/generation/logits_process.py#L161
                    if i < self.gen_kwargs["min_new_tokens"]:
                        next_token_logits[:, self.tokenizer.eos_token_id] = -float('inf')

                    # Debug top 5 scores
                    top_5_scores, top_5_ids = torch.topk(input=next_token_logits, k=5, dim=-1)
                    logging.debug(f"step {i}: top_id {top_5_ids} top_score: {top_5_scores}")

                    # use greedy for debugging
                    # Pad tokens for sequences that are already finished
                    top_scores, top_ids = torch.topk(input=next_token_logits, k=1, dim=-1)
                    next_tokens = top_ids.squeeze(-1) * unfinished_sequence + self.tokenizer.pad_token_id * (1 - unfinished_sequence)
                    unfinished_sequence = unfinished_sequence.mul(
                        next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    )

                    input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                    copy_model_kwargs = self.model._update_model_kwargs_for_generation(
                        outputs, copy_model_kwargs,
                    )
                    top_texts = self.tokenizer.batch_decode(next_tokens, skip_special_tokens=True)
                    # logging.debug(f"step {i}: top_ids: {top_ids}, top_scores: {top_scores}, unfinished: {unfinished_sequence}, top_texts: {top_texts}")

                    for idx, seq in enumerate(pred_text):
                        if unfinished_sequence[idx]:
                            pred_text[idx] += top_texts[idx]
                    # Break if all sequences reach the end
                    if unfinished_sequence.max() == 0:
                        break

                logging.info(f"Per-step generation is equal to gen(): {pred_text == output_text}")

            logging.debug(f"output_text of batch {batch_idx}: {output_text}")
            preds.extend(output_text)
            batch_idx += 1

        logging.info(f"time_stat_dict: {time_stat_dict}")

        results = calculate_rouge_score(preds, targets)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine_file",
                        help="Specify where the GPTJ6B engine file is",
                        default="build/TRTLLM/examples/gptj/gptj-engine/gptj_float16_tp1_rank0.engine",
                        required=False)
    parser.add_argument("--pyt_ckpt_path",
                        help="Specify where the PyTorch checkpoint path is",
                        default="build/models/GPTJ-6B/checkpoint-final")
    parser.add_argument("--batch_size",
                        help="batch size. 80GB can run a maximum of BS=8 for FP32 greedy",
                        type=int,
                        default=1)
    parser.add_argument("--max_input_seq_len",
                        help="Maximum number of input sequence length",
                        type=int,
                        default=1919)
    parser.add_argument("--num_beams",
                        help="The maximum beam width of the decoding op.",
                        type=int,
                        default=1)
    parser.add_argument("--num_samples",
                        help="Number of samples to run. We have 13368 in total for cnn-dailymail validation set",
                        type=int,
                        default=13368)
    parser.add_argument("--torch_precision",
                        help="Run Pytorch in the specified precision",
                        choices=("fp32", "fp16", "bf16"),
                        default="bf16")
    parser.add_argument("--trt_precision",
                        help="Run TensorRT in the specified precision",
                        choices=("fp32", "fp16", "int8", "fp8"),
                        default="fp32")
    parser.add_argument("--use_dla",
                        help="Use DLA instead of gpu",
                        action="store_true")
    parser.add_argument("--skip_engine_build",
                        help="Skip the TRT engine build phase if possible.",
                        action="store_true")
    parser.add_argument("--engine_build_only",
                        help="Build the engine and skip the testing part",
                        action="store_true")
    parser.add_argument("--pytorch",
                        help="whether to run pytorch inference",
                        action="store_true")
    parser.add_argument("--decoding_step",
                        help="Enable to step into the generation cycle of the model.",
                        action="store_true")
    parser.add_argument("--verbose",
                        help="verbose output",
                        action="store_true")
    args = parser.parse_args()

    # Pytorch Tester
    if args.pytorch:
        logging.info(
            f"Running Accuracy test for Pytorch reference implementation.")
        if not os.path.exists(args.pyt_ckpt_path):
            raise RuntimeError(
                f"Cannot access {args.pyt_ckpt_path}. Please download the model or mount the scratch path.")
        pt_tester = PytorchTester(
            args.pyt_ckpt_path,
            args.batch_size,
            args.torch_precision,
            args.max_input_seq_len,
            args.num_beams,
            args.decoding_step,
        )
        rouge = pt_tester.run_inference(args.num_samples)
        logging.info(f"Pytorch ROUGE Score: {rouge}")
    else:
        # TRT Tester
        logging.info(
            f"Running accuracy test for GPTJ6B using {args.engine_file} ...")
        tester = TRTTester(
            args.engine_file,
            args.batch_size,
            args.trt_precision,
            args.pyt_ckpt_path,
            args.num_beams,
            args.use_dla,
            args.skip_engine_build,
            args.engine_build_only,
            args.decoding_step,
            args.verbose)
        rouge = tester.run_inference(args.num_samples)
        logging.info(f"TRT ROUGE Score: {rouge}")

    # TRT
    # To run the TRT tester:
    # python3 -m code.gptj.tensorrt.infer --engine_file /work/build/engines/DGX-H100_H100-SXM-80GBx1/gptj/Offline/gptj-Offline-gpu-b32-fp16.custom_k_99_9_MaxP.plan --num_samples=8 --batch_size=8  --num_beams=4 --skip_engine_build
    # Torch
    # To run the pytorch tester:
    # python3 -m code.gptj.tensorrt.infer --pytorch --batch_size=2 --num_samples=8
    # To check per-step torch output
    # VERBOSE=1 python3 -m code.gptj.tensorrt.infer --pytorch --batch_size=2 --num_samples=2 --decoding_step


if __name__ == "__main__":
    main()
