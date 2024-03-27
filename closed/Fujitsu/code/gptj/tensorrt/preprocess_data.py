#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to preprocess the data for BERT."""

import argparse
import json
import os

import numpy as np
from code.common import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

G_GPTJ6B_MAX_INPUT_SEQLEN = 1919

G_PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)


def prepare_tokenizer(checkpoint_path, padding_side="left"):
    """
    Prepare the tokenizer for the cnn dailymail
    """
    logging.info(f"Initializing tokenizer from {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=G_GPTJ6B_MAX_INPUT_SEQLEN,
        padding_side=padding_side,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def preprocess_cnndailymail_prompt(cnn_val_json_path):
    # Load from CNN dailymail
    with open(cnn_val_json_path, 'r') as fh:
        list_data_dict = json.load(fh)

    sources = [G_PROMPT_INPUT.format_map(
        example) for example in list_data_dict]
    targets = [f"{example['output']}" for example in list_data_dict]

    logging.info(
        f"Loaded {len(sources)} samples from {cnn_val_json_path}")
    return sources, targets


def preprocess_cnndailymail_gptj6b(data_dir, model_dir, preprocessed_data_dir):
    cnn_val_json_path = os.path.join(
        data_dir, "cnn-daily-mail", "cnn_eval.json")
    output_dir = os.path.join(preprocessed_data_dir,
                              "cnn_dailymail_tokenized_gptj")
    ckpt_path = os.path.join(model_dir, "GPTJ-6B", "checkpoint-final")
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Creating GPT tokenizer...")
    tokenizer = prepare_tokenizer(ckpt_path, padding_side="right")
    logging.info("Done creating tokenizer.")

    logging.info("Reading CNN dailymail examples...")
    sources, targets = preprocess_cnndailymail_prompt(cnn_val_json_path)
    data_len = len(sources)
    logging.info(f"Done reading {data_len} CNN dailymail examples.")

    # Converting input strings to tokenized id.
    # 6/7/2023: Note that TRT-LLM has "masked_tokens" and "attention mask" which are opposite to each other at the moment
    # All inputs will be padded to 1919
    logging.info(f"Converting {data_len} articles to tokens...")
    input_batch = tokenizer.batch_encode_plus(
        sources, return_tensors="pt",
        padding='max_length', truncation=True,
        max_length=G_GPTJ6B_MAX_INPUT_SEQLEN
    )

    input_ids = input_batch.input_ids.numpy().astype(np.int32)
    attention_mask = input_batch.attention_mask.numpy().astype(np.int32)
    masked_tokens = 1 - attention_mask
    input_real_seqlen = np.sum(attention_mask, axis=1).astype(np.int32)
    print(
        f"Shape check: input_id: {input_ids.shape} attention_mask: {attention_mask.shape} input_lengths: {input_real_seqlen.shape}")
    logging.info("Done converting articles to tokens.")

    logging.info(
        f"Saving tokenized id, masks, and input lengths to {output_dir} ...")
    np.save(os.path.join(output_dir, "input_ids_padded.npy"), input_ids)
    np.save(os.path.join(output_dir, "attention_mask.npy"), attention_mask)
    np.save(os.path.join(output_dir, "masked_tokens.npy"), masked_tokens)
    np.save(os.path.join(output_dir, "input_lengths.npy"), input_real_seqlen)

    logging.info("Done saving preprocessed data.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", "-d",
        help="Directory containing the input data.",
        default="build/data"
    )
    parser.add_argument(
        "--model_dir", "-m",
        help="Directory containing the models.",
        default="build/models"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Output directory for the preprocessed data.",
        default="build/preprocessed_data"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    preprocessed_data_dir = args.preprocessed_data_dir

    preprocess_cnndailymail_gptj6b(data_dir, model_dir, preprocessed_data_dir)

    print("Done!")


if __name__ == '__main__':
    main()
