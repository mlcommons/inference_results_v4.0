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

"""Script to preprocess the data for SDXL."""

import argparse
import json
import numpy as np
import pandas as pd
import urllib.request

from pathlib import Path
from importlib import import_module
from code.common import logging

make_tokenizer = import_module("code.stable-diffusion-xl.tensorrt.network").make_tokenizer
EmbeddingDims = import_module("code.stable-diffusion-xl.tensorrt.utilities").EmbeddingDims


def prepare_tokenizer(checkpoint_path):
    """
    Prepare the tokenizer for the SDXL
    """
    logging.info(f"Initializing tokenizer from {checkpoint_path}")
    tokenizer = make_tokenizer(checkpoint_path)

    return tokenizer


def prepare_mscoco2014_prompts(sdxl_val_json_path):
    # Load from MSCOCO 2014 dailymail
    with open(sdxl_val_json_path, 'r') as f:
        list_data_dict = json.load(f)

    # Annotation example: {'image_id': 203564, 'id': 37, 'caption': 'A bicycle replica with a clock as the front wheel.'}
    annotations = pd.DataFrame(list_data_dict['annotations'])
    annotations['caption'] = annotations['caption'].apply(lambda x: x.replace('\n', '').strip())

    # Keep a single captions per image
    annotations = annotations.drop_duplicates(subset=["image_id"], keep="first")

    # Sort by id
    annotations = annotations.sort_values(by=["id"])

    # Prompts
    prompts = annotations['caption']

    logging.info(f"Loaded {len(prompts)} samples from {sdxl_val_json_path}")
    return prompts


def prepare_mscoco2014_prompts_5k(sdxl_val_5k_path):
    # Load from MSCOCO 2014 sampled subset
    annotations = pd.read_csv(sdxl_val_5k_path, sep='\t')

    # Prompts
    prompts = annotations['caption']
    print(prompts)

    logging.info(f"Loaded {len(prompts)} samples from {sdxl_val_5k_path}")
    return prompts


def encode_prompts(tokenizer, prompts):
    # Tokenize prompt
    text_input_ids = tokenizer.batch_encode_plus(
        prompts,
        padding="max_length",
        max_length=EmbeddingDims.PROMPT_LEN,
        truncation=True,
        return_tensors="np",
    ).input_ids.astype(np.int32)

    return text_input_ids


def preprocess_mscoco2014_sdxl(data_dir, model_dir, preprocessed_data_dir):
    mscoco_val_5k_path = Path(data_dir, "coco/SDXL/captions_5k_final.tsv")
    output_dir = Path(preprocessed_data_dir, "coco2014-tokenized-sdxl/5k_dataset_final")
    ckpt_path_clip1 = Path(model_dir, "SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/tokenizer")
    ckpt_path_clip2 = Path(model_dir, "SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/tokenizer_2")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Creating SDXL tokenizer...")
    tokenizer_clip1 = prepare_tokenizer(ckpt_path_clip1)
    tokenizer_clip2 = prepare_tokenizer(ckpt_path_clip2)
    logging.info("Done creating tokenizer.")

    logging.info("Reading COCO2014 captions from {mscoco_val_5k_path}...")
    prompts = prepare_mscoco2014_prompts_5k(mscoco_val_5k_path)
    data_len = len(prompts)
    logging.info(f"Done reading {data_len} COCO2014 captions.")

    # Converting input strings to tokenized id.
    # All inputs will be padded to 77
    logging.info(f"Converting {data_len} captions to tokens...")
    input_ids_clip1 = encode_prompts(tokenizer_clip1, prompts)
    input_ids_clip2 = encode_prompts(tokenizer_clip2, prompts)

    logging.info(f"Shape check: input_ids_clip1: {input_ids_clip1.shape} input_ids_clip2: {input_ids_clip2.shape}")
    logging.info("Done converting captions to tokens.")

    logging.info(
        f"Saving tokenized ids from CLIP1 and CLIP2 to {output_dir} ...")
    np.save(Path(output_dir, "prompt_ids_clip1_padded_5k.npy"), input_ids_clip1)
    np.save(Path(output_dir, "prompt_ids_clip2_padded_5k.npy"), input_ids_clip2)

    negative_prompts = ["normal quality, low quality, worst quality, low res, blurry, nsfw, nude"]
    input_negative_ids_clip1 = encode_prompts(tokenizer_clip1, negative_prompts)
    input_negative_ids_clip2 = encode_prompts(tokenizer_clip2, negative_prompts)

    input_negative_ids_clip1 = np.tile(input_negative_ids_clip1, [input_ids_clip1.shape[0], 1])
    input_negative_ids_clip2 = np.tile(input_negative_ids_clip2, [input_ids_clip2.shape[0], 1])

    logging.info(f"Shape check: input_negative_ids_clip1: {input_negative_ids_clip1.shape} input_negative_ids_clip2: {input_negative_ids_clip2.shape}")
    logging.info("Done converting negative prompt to tokens.")

    logging.info(
        f"Saving tokenized negative prompt from CLIP1 and CLIP2 to {output_dir} ...")
    np.save(Path(output_dir, "negative_prompt_ids_clip1_padded_5k.npy"), input_negative_ids_clip1)
    np.save(Path(output_dir, "negative_prompt_ids_clip2_padded_5k.npy"), input_negative_ids_clip2)

    logging.info("Done saving preprocessed data.")


def download_fixed_latent(preprocessed_data_dir):
    logging.info("Downloading SDXL pre generated initial noise latent...")

    output_fpath = Path(preprocessed_data_dir, "coco2014-tokenized-sdxl/5k_dataset_final/latents.pt")
    urllib.request.urlretrieve("https://github.com/mlcommons/inference/raw/master/text_to_image/tools/latents.pt", output_fpath)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", "-d",
        help="Directory containing the input data.",
        default="build/data/"
    )
    parser.add_argument(
        "--model_dir", "-m",
        help="Directory containing the models.",
        default="build/models/"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Output directory for the preprocessed data.",
        default="build/preprocessed_data/"
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    model_dir = args.model_dir
    preprocessed_data_dir = args.preprocessed_data_dir

    preprocess_mscoco2014_sdxl(data_dir, model_dir, preprocessed_data_dir)

    download_fixed_latent(preprocessed_data_dir)

    logging.info("SDXL Data Preprocessing Is Done!")


if __name__ == '__main__':
    main()
