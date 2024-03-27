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

import os
import logging
import torch

from pathlib import Path
from importlib import import_module
from diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel
)
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer
)

# dash in stable-diffusion-xl breaks traditional way of module import
EmbeddingDims = import_module("code.stable-diffusion-xl.tensorrt.utilities").EmbeddingDims
PipelineConfig = import_module("code.stable-diffusion-xl.tensorrt.utilities").PipelineConfig


class AbstractModel:
    def __init__(self,
                 max_batch_size: int = 16,
                 text_maxlen: int = EmbeddingDims.PROMPT_LEN,
                 embedding_dim: int = EmbeddingDims.CLIP,
                 precision: str = 'fp16',
                 device: str = 'cuda',
                 verbose: bool = True,
                 ):

        self.name = self.__class__.__name__
        self.device = device
        self.verbose = verbose

        self.precision = precision
        assert self.precision in ['fp32', 'fp16', 'int8'], f"Invalid model precision: {self.precision}"

        self.min_batch = 1
        self.max_batch = max_batch_size

        self.text_maxlen = text_maxlen
        self.embedding_dim = embedding_dim

    def _actual_check_dims(self, batch_size):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch, f"batch_size out of range of ({self.min_batch}, {self.max_batch})"

    def get_model(self, pytorch_ckpt_path: os.PathLike):
        raise NotImplementedError

    def export_onnx(self):
        raise NotImplementedError

    def get_input_names(self):
        raise NotImplementedError

    def get_output_names(self):
        raise NotImplementedError

    def get_onnx_output_names(self):
        raise NotImplementedError

    def get_dynamic_axes(self):
        raise NotImplementedError

    def get_sample_input(self, batch_size):
        raise NotImplementedError

    def get_input_profile(self, batch_size):
        raise NotImplementedError

    def get_shape_dict(self, batch_size):
        raise NotImplementedError

    def check_dims(func):
        def inner(self, batch_size, *args, **kwargs):
            self._actual_check_dims(batch_size, )
            return func(self, batch_size, *args, **kwargs)
        return inner


class CLIP(AbstractModel):
    def __init__(self,
                 max_batch_size: int,
                 embedding_dim: int = EmbeddingDims.CLIP,
                 precision: str = 'fp16',
                 device: str = 'cuda',
                 verbose: bool = True,
                 ):
        super().__init__(max_batch_size=max_batch_size, embedding_dim=embedding_dim, precision=precision, device=device, verbose=verbose)

    def get_model(self, pytorch_ckpt_path: os.PathLike):
        Path(pytorch_ckpt_path).resolve(strict=True)
        logging.info(f"[I] Load CLIP pytorch model from: {pytorch_ckpt_path}")
        model = CLIPTextModel.from_pretrained(pytorch_ckpt_path).to(self.device)
        return model

    def get_input_names(self):
        return ['input_ids']

    def get_output_names(self):
        return ['text_embeddings', 'hidden_states']

    def get_onnx_output_names(self):
        return ['text_embeddings', 'hidden_states_output']

    def get_dynamic_axes(self):
        return {
            'input_ids': {0: '2B'},
            'text_embeddings': {0: '2B'},
            'hidden_states': {0: '2B'}
        }

    @AbstractModel.check_dims
    def get_input_profile(self, batch_size):
        return {
            'input_ids': [(2 * self.min_batch, self.text_maxlen), (2 * batch_size, self.text_maxlen), (2 * batch_size, self.text_maxlen)]
        }

    @AbstractModel.check_dims
    def get_shape_dict(self, batch_size):
        tensor_shape_dict = {
            'input_ids': (2 * batch_size, self.text_maxlen),
            'text_embeddings': (2 * batch_size, self.text_maxlen, self.embedding_dim),
            'hidden_states': (2 * batch_size, self.text_maxlen, self.embedding_dim),
        }
        return tensor_shape_dict

    @AbstractModel.check_dims
    def get_sample_input(self, batch_size):
        return torch.zeros(2 * batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)


class CLIPWithProj(CLIP):
    def __init__(self,
                 max_batch_size: int,
                 embedding_dim: int = EmbeddingDims.CLIP_PROJ,
                 precision: str = 'fp16',
                 device: str = 'cuda',
                 verbose: bool = True,
                 ):
        super().__init__(max_batch_size=max_batch_size, embedding_dim=embedding_dim, precision=precision, device=device, verbose=verbose)

    def get_model(self, pytorch_ckpt_path):
        Path(pytorch_ckpt_path).resolve(strict=True)
        logging.info(f"[I] Load CLIP with Proj pytorch model from: {pytorch_ckpt_path}")
        model = CLIPTextModelWithProjection.from_pretrained(pytorch_ckpt_path).to(self.device)
        return model

    @AbstractModel.check_dims
    def get_shape_dict(self, batch_size):
        tensor_shape_dict = {
            'input_ids': (2 * batch_size, self.text_maxlen),
            'text_embeddings': (2 * batch_size, self.embedding_dim),
            'hidden_states': (2 * batch_size, self.text_maxlen, self.embedding_dim),
        }
        return tensor_shape_dict


class UNetXL(AbstractModel):
    def __init__(self,
                 max_batch_size: int,
                 embedding_dim: int = EmbeddingDims.UNETXL,
                 unet_dim: int = 4,
                 time_dim: int = 6,
                 precision: str = 'int8',
                 device: str = 'cuda',
                 verbose: bool = True,
                 ):
        super().__init__(max_batch_size=max_batch_size, embedding_dim=embedding_dim, precision=precision, device=device, verbose=verbose)
        self.subfolder = 'unet'
        self.unet_dim = unet_dim
        self.time_dim = time_dim

    def get_model(self, pytorch_ckpt_path: os.PathLike):
        Path(pytorch_ckpt_path).resolve(strict=True)
        logging.info(f"Load UNet pytorch model from: {pytorch_ckpt_path}")
        model = UNet2DConditionModel.from_pretrained(pytorch_ckpt_path).to(self.device)
        model.set_default_attn_processor()
        return model

    def get_input_names(self):
        return ['sample', 'timestep', 'encoder_hidden_states', 'text_embeds', 'time_ids']

    def get_output_names(self):
        return ['latent']

    def get_onnx_output_names(self):
        return self.get_output_names()

    def get_dynamic_axes(self):
        return {
            'sample': {0: '2B', 2: 'H', 3: 'W'},
            'encoder_hidden_states': {0: '2B'},
            'latent': {0: '2B', 2: 'H', 3: 'W'},
            'text_embeds': {0: '2B'},
            'time_ids': {0: '2B'}
        }

    @AbstractModel.check_dims
    def get_input_profile(self, batch_size):
        latent_height = PipelineConfig.LATENT_SIZE
        latent_width = PipelineConfig.LATENT_SIZE
        return {
            'sample': [(2 * self.min_batch, self.unet_dim, latent_height, latent_width), (2 * batch_size, self.unet_dim, latent_height, latent_width), (2 * batch_size, self.unet_dim, latent_height, latent_width)],
            'encoder_hidden_states': [(2 * self.min_batch, self.text_maxlen, self.embedding_dim), (2 * batch_size, self.text_maxlen, self.embedding_dim), (2 * batch_size, self.text_maxlen, self.embedding_dim)],
            'text_embeds': [(2 * self.min_batch, EmbeddingDims.CLIP_PROJ), (2 * batch_size, EmbeddingDims.CLIP_PROJ), (2 * batch_size, EmbeddingDims.CLIP_PROJ)],
            'time_ids': [(2 * self.min_batch, self.time_dim), (2 * batch_size, self.time_dim), (2 * batch_size, self.time_dim)],
        }

    @AbstractModel.check_dims
    def get_shape_dict(self, batch_size):
        latent_height = PipelineConfig.LATENT_SIZE
        latent_width = PipelineConfig.LATENT_SIZE
        return {
            'sample': (2 * batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (2 * batch_size, self.text_maxlen, self.embedding_dim),
            'text_embeds': (2 * batch_size, EmbeddingDims.CLIP_PROJ),
            'time_ids': (2 * batch_size, self.time_dim),
            'timestep': (1, ),
            'latent': (2 * batch_size, 4, latent_height, latent_width),
        }

    @AbstractModel.check_dims
    def get_sample_input(self, batch_size):
        dtype = torch.float16 if self.precision == 'fp16' else torch.float32
        latent_height = PipelineConfig.LATENT_SIZE
        latent_width = PipelineConfig.LATENT_SIZE
        return (
            torch.randn(2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),  # sample
            torch.tensor([1.], dtype=torch.float32, device=self.device),  # timestep
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),  # encoder_hidden_states
            {
                'added_cond_kwargs': {
                    'text_embeds': torch.randn(2 * batch_size, EmbeddingDims.CLIP_PROJ, dtype=dtype, device=self.device),
                    'time_ids': torch.randn(2 * batch_size, self.time_dim, dtype=dtype, device=self.device)
                }
            }
        )


class VAE(AbstractModel):
    def __init__(self,
                 max_batch_size: int,
                 precision: str = 'fp32',
                 device: str = 'cuda',
                 verbose: bool = True,
                 ):
        super().__init__(max_batch_size=max_batch_size, precision=precision, device=device, verbose=verbose)

    def get_model(self, pytorch_ckpt_path: os.PathLike):
        Path(pytorch_ckpt_path).resolve(strict=True)
        logging.info(f"[I] Load VAE decoder pytorch model from: {pytorch_ckpt_path}")
        model = AutoencoderKL.from_pretrained(pytorch_ckpt_path).to(self.device)
        model.forward = model.decode
        return model

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
        return ['images']

    def get_onnx_output_names(self):
        return self.get_output_names()

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    @AbstractModel.check_dims
    def get_input_profile(self, batch_size):
        latent_height = PipelineConfig.LATENT_SIZE
        latent_width = PipelineConfig.LATENT_SIZE
        return {
            'latent': [(self.min_batch, 4, latent_height, latent_width), (batch_size, 4, latent_height, latent_width), (batch_size, 4, latent_height, latent_width)]
        }

    @AbstractModel.check_dims
    def get_shape_dict(self, batch_size):
        latent_height = PipelineConfig.LATENT_SIZE
        latent_width = PipelineConfig.LATENT_SIZE
        image_height = PipelineConfig.IMAGE_SIZE
        image_width = PipelineConfig.IMAGE_SIZE
        return {
            'latent': (batch_size, 4, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }

    @AbstractModel.check_dims
    def get_sample_input(self, batch_size):
        latent_height = PipelineConfig.LATENT_SIZE
        latent_width = PipelineConfig.LATENT_SIZE
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)


def make_tokenizer(pytorch_ckpt_path):
    logging.info(f"[I] Load tokenizer pytorch model from: {pytorch_ckpt_path}")
    model = CLIPTokenizer.from_pretrained(pytorch_ckpt_path)
    return model
