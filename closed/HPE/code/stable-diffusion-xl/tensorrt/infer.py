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


__doc__ = """Scripts that take a set of SDXL engines and SDXL inputs, infer the output and test the accuracy"""

import argparse
import numpy as np
import pandas as pd
import torch

from PIL import Image
from pathlib import Path
from cuda import cudart
from importlib import import_module
from diffusers import (
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
)

from code.common import logging
from code.common.systems.system_list import DETECTED_SYSTEM
from code.common.systems.system_list import SystemClassifications
from build.inference.text_to_image.tools.clip.clip_encoder import CLIPEncoder
from build.inference.text_to_image.tools.fid.fid_score import compute_fid

# dash in stable-diffusion-xl breaks traditional way of module import
Dataset = import_module("code.stable-diffusion-xl.tensorrt.dataset").Dataset
PipelineConfig = import_module("code.stable-diffusion-xl.tensorrt.utilities").PipelineConfig
calculate_max_engine_device_memory = import_module("code.stable-diffusion-xl.tensorrt.utilities").calculate_max_engine_device_memory
torch_to_image = import_module("code.stable-diffusion-xl.tensorrt.utilities").torch_to_image
CUASSERT = import_module("code.stable-diffusion-xl.tensorrt.utilities").CUASSERT
CLIP = import_module("code.stable-diffusion-xl.tensorrt.network").CLIP
CLIPWithProj = import_module("code.stable-diffusion-xl.tensorrt.network").CLIPWithProj
UNetXL = import_module("code.stable-diffusion-xl.tensorrt.network").UNetXL
VAE = import_module("code.stable-diffusion-xl.tensorrt.network").VAE
sdxl_scheduler = import_module("code.stable-diffusion-xl.tensorrt.scheduler")
SDXLEngine = import_module("code.stable-diffusion-xl.tensorrt.backend").SDXLEngine
SDXLBufferManager = import_module("code.stable-diffusion-xl.tensorrt.backend").SDXLBufferManager

G_START_IND = 0

# FIX ME: update to use mlcommons code


class CoCoAccuracyTester:
    """
    Post processing to calculate FID and CLIP
    ref: https://github.com/mlcommons/inference/blob/master/text_to_image/tools/accuracy_coco.py
    """

    def __init__(self, raw_captions, statistics_path, device="cuda"):
        self.raw_captions = pd.read_csv(raw_captions, sep='\t')
        self.statistics_path = statistics_path.as_posix()

        self.device = device if torch.cuda.is_available() else "cpu"
        self.clip = CLIPEncoder(device=self.device)

    def preprocess_image(self, img_dir, file_name):
        img = Image.open(img_dir + "/" + file_name)
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        tensor = torch.Tensor(np.asarray(img).transpose([2, 0, 1])).to(torch.uint8)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor.unsqueeze(0)

    def report_accuracy(self, generated_images):
        logging.info("Accumulating results")
        clip_scores = []
        result_list = []
        for idx, generated_img in enumerate(generated_images):
            # Scale [0, 1] to [0, 255]
            generated_img = (generated_img * 255).clamp(0, 255).permute(1, 2, 0).round().to(torch.uint8).cpu().numpy()
            result_list.append(generated_img)
            # Load Ground Truth
            caption = self.raw_captions["caption"][idx]
            clip_scores.append(
                100 * self.clip.get_clip_score(caption, Image.fromarray(generated_img)).item()
            )

        fid_score = compute_fid(result_list, self.statistics_path, self.device)
        clip_score = np.mean(clip_scores)

        logging.info(f"[FID] {fid_score}, [CLIP] {clip_score}")


class BaseTester:
    def __init__(self,
                 preprocessed_data_dir: str,
                 batch_size: int,
                 num_samples: int,
                 latent_dtype: str,
                 denoising_steps: int,
                 seed: int = None,
                 verbose: bool = False,
                 debug: bool = False):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.denoising_steps = denoising_steps
        self.seed = seed
        self.verbose = verbose
        self.debug = debug

        if self.seed:
            logging.warning(f"Using generator seed {self.seed} instead of official pregenerated noise latent!")

        # initialize inputs
        self.prompt_tokens_clip1 = np.load(Path(preprocessed_data_dir, "prompt_ids_clip1_padded_5k.npy"))
        self.prompt_tokens_clip2 = np.load(Path(preprocessed_data_dir, "prompt_ids_clip2_padded_5k.npy"))
        self.negative_prompt_tokens_clip1 = np.load(Path(preprocessed_data_dir, "negative_prompt_ids_clip1_padded_5k.npy"))
        self.negative_prompt_tokens_clip2 = np.load(Path(preprocessed_data_dir, "negative_prompt_ids_clip2_padded_5k.npy"))
        if latent_dtype == "fp32":
            self.latent_dtype = torch.float32
        elif latent_dtype == "fp16":
            self.latent_dtype = torch.float16
        else:
            raise ValueError(f"Not of supported latent dtype '{latent_dtype}'")

        self.init_noise_latent = torch.load(Path(preprocessed_data_dir, "latents.pt"))

        # output placeholder
        self.ids = np.load(Path(preprocessed_data_dir, "ids.npy"))
        self.images = []

    def _verbose_info(self, msg):
        if self.verbose:
            logging.info(msg)

    def _debug_info(self, msg):
        if self.debug:
            logging.info(msg)

    def report_accuracy(self):
        NotImplementedError

    def save_images(self, image_dir):
        image_dir = Path(image_dir)
        image_dir.mkdir(exist_ok=True, parents=True)
        for i, image in enumerate(self.images):
            precision = f'{self.latent_dtype}'[-7:]
            output_id = self.ids[i + G_START_IND]
            output_path = Path(image_dir, f"{output_id}.png")
            self._verbose_info(f"Saving {self.__class__.__name__} {precision} output to {output_path}")
            if (torch.is_tensor(image)):
                image = torch_to_image(image)
            image.save(output_path)


class TorchTester(BaseTester):
    def __init__(self,
                 pytorch_dir: str,
                 preprocessed_data_dir: str,
                 batch_size: int,
                 num_samples: int,
                 latent_dtype: str,
                 denoising_steps: int,
                 seed: int = None,
                 verbose: bool = False,
                 debug: bool = False):
        super().__init__(
            preprocessed_data_dir,
            batch_size,
            num_samples,
            latent_dtype,
            denoising_steps,
            seed,
            verbose,
            debug)
        self.device = "cuda"
        self.pytorch_dir = pytorch_dir

        # initialize HF pipeline
        self.scheduler = EulerDiscreteScheduler.from_pretrained(Path(self.pytorch_dir, latent_dtype, "scheduler"))
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            Path(self.pytorch_dir, latent_dtype),
            scheduler=self.scheduler,
            add_watermarker=False,
            variant="fp16" if self.latent_dtype == torch.float16 else None,
            torch_dtype=self.latent_dtype,
        ).to(self.device)

        self.init_noise_latent = self.init_noise_latent.to(self.latent_dtype).to(self.device)

    def save_images(self, image_dir):
        return super().save_images(image_dir)

    def encode_tokens(self, prompt_tokens_clip1, prompt_tokens_clip2, negative_prompt_tokens_clip1, negative_prompt_tokens_clip2):
        text_encoders = (
            [self.pipeline.text_encoder, self.pipeline.text_encoder_2]
        )

        # Encode prompt ids
        prompt_embeds_list = []
        prompt_tokens_list = [torch.tensor(prompt_tokens_clip1), torch.tensor(prompt_tokens_clip2)]
        for prompt_tokens, text_encoder in zip(
            prompt_tokens_list, text_encoders
        ):
            prompt_embeds = text_encoder(
                prompt_tokens.to(self.device),
                output_hidden_states=True
            )
            # We are only ALWAYS interested in the pooled output of the 2nd text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # Encode negative prompt ids
        negative_prompt_tokens_list = [torch.tensor(negative_prompt_tokens_clip1), torch.tensor(negative_prompt_tokens_clip2)]

        negative_prompt_embeds_list = []
        for negative_prompt_tokens, text_encoder in zip(
            negative_prompt_tokens_list, text_encoders
        ):
            negative_prompt_embeds = text_encoder(
                negative_prompt_tokens.to(self.device),
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the 2nd text encoder
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def generate_images(self):
        with torch.no_grad():
            for i in range(0, self.num_samples, self.batch_size):
                actual_batch_size = self.batch_size if self.num_samples - i > self.batch_size else self.num_samples - i
                self._verbose_info(f"Running inference on batch {i}, with batch size {actual_batch_size}")
                prompt_tokens_clip1 = self.prompt_tokens_clip1[i: i + actual_batch_size]
                prompt_tokens_clip2 = self.prompt_tokens_clip2[i: i + actual_batch_size]
                negative_prompt_tokens_clip1 = self.negative_prompt_tokens_clip1[i: i + actual_batch_size]
                negative_prompt_tokens_clip2 = self.negative_prompt_tokens_clip2[i: i + actual_batch_size]
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.encode_tokens(
                    prompt_tokens_clip1,
                    prompt_tokens_clip2,
                    negative_prompt_tokens_clip1,
                    negative_prompt_tokens_clip2
                )
                if self.debug:
                    self._debug_info(f'torch prompt_embeds clip1_hidden_states: {prompt_embeds[:,:,:768].shape} {prompt_embeds[:,:,:768]}')
                    self._debug_info(f'torch prompt_embeds clip2_hidden_states: {prompt_embeds[:,:,768:].shape} {prompt_embeds[:,:,768:]}')
                    torch.save(prompt_embeds[:, :, :768].cpu(), f'./build/test-infer-hf-fp32/prompt_embeds_clip1_hidden_states_{i}.pt')
                    torch.save(prompt_embeds[:, :, 768:].cpu(), f'./build/test-infer-hf-fp32/prompt_embeds_clip2_hidden_states_{i}.pt')
                    self._debug_info(f'torch negative_prompt_embeds clip1_hidden_states: {negative_prompt_embeds[:,:,:768].shape} {negative_prompt_embeds[:,:,:768]}')
                    self._debug_info(f'torch negative_prompt_embeds clip2_hidden_states: {negative_prompt_embeds[:,:,768:].shape} {negative_prompt_embeds[:,:,768:]}')
                    torch.save(negative_prompt_embeds[:, :, :768].cpu(), f'./build/test-infer-hf-fp32/negative_prompt_embeds_clip1_hidden_states_{i}.pt')
                    torch.save(negative_prompt_embeds[:, :, 768:].cpu(), f'./build/test-infer-hf-fp32/negative_prompt_embeds_clip2_hidden_states_{i}.pt')
                    self._debug_info(f'torch pooled_prompt_embeds: {pooled_prompt_embeds.shape} {pooled_prompt_embeds}')
                    torch.save(pooled_prompt_embeds.cpu(), f'./build/test-infer-hf-fp32/pooled_prompt_embeds_{i}.pt')
                    self._debug_info(f'torch negative_pooled_prompt_embeds: {negative_pooled_prompt_embeds.shape} {negative_pooled_prompt_embeds}')
                    torch.save(negative_pooled_prompt_embeds.cpu(), f'./build/test-infer-hf-fp32/negative_pooled_prompt_embeds_{i}.pt')
                output_images = self.pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    guidance_scale=PipelineConfig.GUIDANCE,
                    num_inference_steps=self.denoising_steps,
                    output_type="pt",
                    latents=None if self.seed else self.init_noise_latent.tile((actual_batch_size, 1, 1, 1)),
                    generator=torch.Generator(device=self.device).manual_seed(
                        self.seed
                    ) if self.seed else None,
                )
                self.images += output_images.images.cpu()


class TRTTester(BaseTester):
    def __init__(self,
                 engine_dir: str,
                 preprocessed_data_dir: str,
                 batch_size: int,
                 num_samples: int,
                 unet_precision: str,
                 latent_dtype: str,
                 denoising_steps: int,
                 use_graphs: bool = False,
                 seed: int = None,
                 verbose: bool = False,
                 debug: bool = False):
        super().__init__(
            preprocessed_data_dir,
            batch_size,
            num_samples,
            latent_dtype,
            denoising_steps,
            seed,
            verbose,
            debug)
        self.device = "cuda"
        self.engine_dir = engine_dir
        self.use_graphs = use_graphs

        # Pipeline components
        self.add_time_ids = torch.tensor(
            [PipelineConfig.IMAGE_SIZE, PipelineConfig.IMAGE_SIZE, 0, 0, PipelineConfig.IMAGE_SIZE, PipelineConfig.IMAGE_SIZE],
            dtype=self.latent_dtype,
            device=self.device)
        self.scheduler = sdxl_scheduler.EulerDiscreteScheduler()

        clip_with_proj_precision = 'fp32' if SystemClassifications.is_orin() else 'fp16'

        self.models = {
            'clip1': CLIP(max_batch_size=batch_size, precision='fp16', device=self.device),
            'clip2': CLIPWithProj(max_batch_size=batch_size, precision=clip_with_proj_precision, device=self.device),
            'unet': UNetXL(max_batch_size=batch_size, precision=unet_precision, device=self.device),
            'vae': VAE(max_batch_size=batch_size, precision='fp32', device=self.device),
        }
        self.engines = {}
        self.buffers = None
        # Use fp32 latent for TRT
        self.init_noise_latent = self.init_noise_latent.to(self.device)

        # Runtime components
        self.context_memory = None
        self.infer_stream = CUASSERT(cudart.cudaStreamCreate())  # we run all cuda stream functions on a single stream to simply the tester

        # Initialize
        self._initialize()

    def _initialize(self, scenario: str = "Offline"):
        # Initialize scheduler
        self.scheduler.set_timesteps(self.denoising_steps)

        # Initialize engines
        for name, model in self.models.items():
            engine_name = f"stable-diffusion-xl-{model.name}-{scenario}-gpu-b{self.batch_size}-{model.precision}.custom_k_99_MaxP.plan"
            engine_path = Path(self.engine_dir, engine_name)
            self.engines[name] = SDXLEngine(engine_name=name, engine_path=engine_path)

        # Initialize engine runtime
        max_device_memory = calculate_max_engine_device_memory(self.engines)
        shared_device_memory = CUASSERT(cudart.cudaMalloc(max_device_memory))
        self.context_memory = shared_device_memory

        for engine in self.engines.values():
            self._verbose_info(f"Activating engine: {engine.engine_path}")
            engine.activate(self.context_memory)

        # Initialize buffers
        self.buffers = SDXLBufferManager(self.engines)
        self.buffers.initialize()

        if self.use_graphs:
            # NOTE(vir): only enable graphs for unet for now
            self.engines['unet'].enable_cuda_graphs(self.buffers)

    def _transfer_to_clip_buffer(self, prompt_tokens_clip1, prompt_tokens_clip2, negative_prompt_tokens_clip1, negative_prompt_tokens_clip2):
        # TODO: yihengz compare cuda copy speed against torch.tensor()
        # cudart.cudaMemcpy(self.buffers['clip1'].get_tensor('input_ids').data_ptr(),
        #                   prompt_tokens_clip1.ctypes.data,
        #                   prompt_tokens_clip1.size * prompt_tokens_clip1.itemsize,
        #                   cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # [ negative prompt, prompt ]
        concat_prompt_clip1 = torch.concat([torch.tensor(negative_prompt_tokens_clip1), torch.tensor(prompt_tokens_clip1)], dim=0).to(self.device)
        concat_prompt_clip2 = torch.concat([torch.tensor(negative_prompt_tokens_clip2), torch.tensor(prompt_tokens_clip2)], dim=0).to(self.device)
        self.buffers['clip1_input_ids'] = concat_prompt_clip1
        self.buffers['clip2_input_ids'] = concat_prompt_clip2

    def _save_buffer_to_images(self, actual_batch_size):
        # TODO: yihengz check if we actually need sync the stream
        CUASSERT(cudart.cudaStreamSynchronize(self.infer_stream))  # make sure VAE kernel execution are finished

        # Normalize TRT output
        vae_outputs = (self.buffers['vae_images'].detach().clone().cpu() + 1) * 0.5
        self.images += list(vae_outputs[:actual_batch_size])

    def _get_time_ids(self, actual_batch_size):
        return self.add_time_ids.repeat(actual_batch_size * 2, 1).to(torch.float16)

    def _encode_tokens(self, actual_batch_size):
        clip_models = ['clip1', 'clip2']
        for clip in clip_models:
            for tensor_name, tensor_shape in self.models[clip].get_shape_dict(actual_batch_size).items():
                self.engines[clip].stage_tensor(tensor_name, self.buffers[f'{clip}_{tensor_name}'], tensor_shape)
            self.engines[clip].infer(self.infer_stream)

    def _denoise_latent(self, actual_batch_size):
        # Prepare predetermined input tensors
        if self.seed:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            latents_shape = self.models['unet'].get_shape_dict(actual_batch_size)['sample']
            latents = torch.randn(latents_shape, device=self.device, dtype=self.latent_dtype, generator=generator)  # TODO modify dtype after we switch to quantized engine
        else:
            latents = self.init_noise_latent
            latents = torch.concat([latents] * actual_batch_size)
        latents = latents * self.scheduler.init_noise_sigma()
        encoder_hidden_states = torch.concat([
            self.buffers['clip1_hidden_states'],
            self.buffers['clip2_hidden_states'].to(self.latent_dtype)
        ], dim=-1)
        text_embeds = self.buffers['clip2_text_embeddings'].to(self.latent_dtype)

        self.buffers['unet_encoder_hidden_states'] = encoder_hidden_states
        self.buffers['unet_text_embeds'] = text_embeds
        self.buffers['unet_time_ids'] = self._get_time_ids(actual_batch_size)

        for step_index, timestep in enumerate(self.scheduler.timesteps):
            # Expand the latents because we have prompt and negative prompt guidance
            latents_expanded = self.scheduler.scale_model_input(torch.concat([latents] * 2), step_index, timestep)

            # Prepare runtime dependent input tensors
            self.buffers['unet_sample'] = latents_expanded.to(self.latent_dtype)
            self.buffers['unet_timestep'] = timestep.to(self.latent_dtype).to("cuda")

            for tensor_name, tensor_shape in self.models['unet'].get_shape_dict(actual_batch_size).items():
                self.engines['unet'].stage_tensor(tensor_name, self.buffers[f'unet_{tensor_name}'], tensor_shape)

            self.engines['unet'].infer(self.infer_stream, batch_size=actual_batch_size)

            # TODO: yihengz check if we actually need sync the stream
            CUASSERT(cudart.cudaStreamSynchronize(self.infer_stream))  # make sure Unet kernel execution are finished

            # Perform guidance
            noise_pred = self.buffers['unet_latent']

            noise_pred_negative_prompt = noise_pred[0:actual_batch_size]  # negative prompt in batch dimension [0:BS]
            noise_pred_prompt = noise_pred[actual_batch_size:actual_batch_size * 2]  # prompt in batch dimension [BS:]

            noise_pred = noise_pred_negative_prompt + PipelineConfig.GUIDANCE * (noise_pred_prompt - noise_pred_negative_prompt)

            latents = self.scheduler.step(noise_pred, latents, step_index)

        latents = 1. / PipelineConfig.VAE_SCALING_FACTOR * latents
        # Transfer the Unet output to vae buffer
        self.buffers['vae_latent'] = latents

    def _decode_latent(self, actual_batch_size):
        # self._verbose_info(f"Decoding latent")
        for tensor_name, tensor_shape in self.models['vae'].get_shape_dict(actual_batch_size).items():
            self.engines['vae'].stage_tensor(tensor_name, self.buffers[f'vae_{tensor_name}'], tensor_shape)
        self.engines['vae'].infer(self.infer_stream)

    def save_images(self, image_dir):
        return super().save_images(image_dir)

    def generate_images(self):
        for i in range(0 + G_START_IND, self.num_samples + G_START_IND, self.batch_size):
            actual_batch_size = self.batch_size if self.num_samples + G_START_IND - i > self.batch_size else self.num_samples + G_START_IND - i
            self._verbose_info(f"Running inference on batch {i}, with batch size {actual_batch_size}")
            # self._debug_info(f"Memory info: {get_gpu_memory()}")
            prompt_tokens_clip1 = self.prompt_tokens_clip1[i: i + actual_batch_size]
            prompt_tokens_clip2 = self.prompt_tokens_clip2[i: i + actual_batch_size]
            negative_prompt_tokens_clip1 = self.negative_prompt_tokens_clip1[i: i + actual_batch_size]
            negative_prompt_tokens_clip2 = self.negative_prompt_tokens_clip2[i: i + actual_batch_size]

            self._transfer_to_clip_buffer(
                prompt_tokens_clip1,
                prompt_tokens_clip2,
                negative_prompt_tokens_clip1,
                negative_prompt_tokens_clip2
            )

            self._encode_tokens(actual_batch_size)
            self._denoise_latent(actual_batch_size)  # runs self.denoising_steps inside
            self._decode_latent(actual_batch_size)

            self._save_buffer_to_images(actual_batch_size)


def main():
    # To run the TRT tester:
    # python3 -m code.stable-diffusion-xl.tensorrt.infer --engine-dir=/work/build/engines/DGX-H100_H100-SXM-80GBx1/stable-diffusion-xl/Offline/ --batch-size=1 --num-samples=1 --latent-dtype=fp16
    # To run the Torch tester:
    # python3 -m code.stable-diffusion-xl.tensorrt.infer --pytorch-dir=/work/build/models/SDXL/pytorch_models/ --batch-size=2 --num-samples=4 --latent-dtype=fp16
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir",
                        help="Specify where the SDXL engine files are")
    parser.add_argument("--pytorch-dir",
                        help="Specify where the SDXL pretrained checkpoint files are",
                        default="build/models/SDXL/pytorch_models/")
    parser.add_argument("--preprocessed-data-dir",
                        help="Specify where the SDXL proprocessed input data files are",
                        default="build/preprocessed_data/coco2014-tokenized-sdxl/5k_dataset_final/")
    parser.add_argument("--validation-data-dir",
                        help="Specify where the SDXL proprocessed vailidation data files are",
                        default="build/data/coco/SDXL/coco2014/")
    parser.add_argument("--batch-size",
                        help="batch size",
                        type=int,
                        default=1)
    parser.add_argument("--num-samples",
                        help="Number of samples to run. We have 24781 in total for openImages",
                        type=int,
                        default=1)
    parser.add_argument("--unet-precision",
                        help="Precision of the Unet engine",
                        type=str,
                        default="int8")
    parser.add_argument("--latent-dtype",
                        help="Datatype of the Unet input latent",
                        type=str,
                        default="fp32")
    parser.add_argument("--denoising-step",
                        help="Steps of the denoising scheduler",
                        type=int,
                        default=PipelineConfig.STEPS)
    parser.add_argument("--image-dir",
                        help="Specify where to dump generated images")
    parser.add_argument("--use-graphs",
                        help="Enable use of CUDA Graphs",
                        action="store_true")
    parser.add_argument("--seed",
                        help="Generator seed for the SDXL pipeline",
                        type=int)
    parser.add_argument("--verbose",
                        help="verbose output",
                        action="store_true")
    parser.add_argument("--debug",
                        help="debug output",
                        action="store_true")
    args = parser.parse_args()

    if args.num_samples != 5000:
        logging.warning("Full accuracy test must be run with --num-samples=5000. Please dump images with partial run!")
    tester = None
    if args.engine_dir:
        # TRT Tester
        logging.info(f"Running accuracy test with TensorRT engines")
        tester = TRTTester(args.engine_dir,
                           args.preprocessed_data_dir,
                           args.batch_size,
                           args.num_samples,
                           args.unet_precision,
                           args.latent_dtype,
                           args.denoising_step,
                           args.use_graphs,
                           args.seed,
                           args.verbose,
                           args.debug)
    else:
        # Torch Tester
        logging.info(f"Running accuracy test with Pytorch reference implementation")
        tester = TorchTester(args.pytorch_dir,
                             args.preprocessed_data_dir,
                             args.batch_size,
                             args.num_samples,
                             args.latent_dtype,
                             args.denoising_step,
                             args.seed,
                             args.verbose,
                             args.debug)
    tester.generate_images()
    # if args.num_samples == 5000:
    #     accuracy_reported = CoCoAccuracyTester(Path(args.validation_data_dir, "captions/captions_source.tsv"), Path("build/preprocessed_data/coco2014-tokenized-sdxl/5k_dataset_final/val2014.npz"))
    #     accuracy_reported.report_accuracy(tester.images)
    if args.image_dir:
        tester.save_images(args.image_dir)


if __name__ == "__main__":
    main()
