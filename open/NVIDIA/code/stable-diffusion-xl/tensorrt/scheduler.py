# Copyright 2023 Katherine Crowson and The HuggingFace Team. All rights reserved.
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

import torch
import numpy as np

# TODO double check HF default config
# {
#   "_class_name": "EulerDiscreteScheduler",
#   "_diffusers_version": "0.21.2",
#   "beta_end": 0.012,
#   "beta_schedule": "scaled_linear",
#   "beta_start": 0.00085,
#   "clip_sample": false,
#   "interpolation_type": "linear",
#   "num_train_timesteps": 1000,
#   "prediction_type": "epsilon",
#   "sample_max_value": 1.0,
#   "set_alpha_to_one": false,
#   "skip_prk_steps": true,
#   "steps_offset": 1,
#   "timestep_spacing": "leading",
#   "trained_betas": null,
#   "use_karras_sigmas": false
# }


class EulerDiscreteScheduler():
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        device='cuda',
        prediction_type: str = "epsilon",
        steps_offset: int = 1,
    ):
        # TODO yihengz: double check HF default parameters
        self.betas = (
            torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.is_scale_input_called = False

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type

    def set_timesteps(self, num_inference_steps: int):
        # "timestep_spacing": "leading"
        self.num_inference_steps = num_inference_steps

        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
        timesteps += self.steps_offset

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=self.device)
        self.timesteps = torch.from_numpy(timesteps).to(device=self.device)

    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        # "timestep_spacing": "leading"
        return (self.sigmas.max() ** 2 + 1) ** 0.5

    def initialize_latents(
        self,
        batch_size: int,
        unet_channels: int = 4,
        latent_height: int = 128,
        latent_width: int = 128,
        generator: torch.Generator = None
    ) -> torch.FloatTensor:
        latents_dtype = torch.float32  # TODO yihengz: check if we should use fp16
        latents_shape = (batch_size, unet_channels, latent_height, latent_width)
        latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype, generator=generator)
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma()
        return latents

    def scale_model_input(
        self, sample: torch.FloatTensor, step_index, timestep
    ) -> torch.FloatTensor:
        sigma = self.sigmas[step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)

        self.is_scale_input_called = True
        return sample

    def step(
        self, model_output, sample, step_index,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator = None,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output: The direct output from learned diffusion model.
            sample: A current instance of a sample created by the diffusion process.
            timestep: The current discrete timestep in the diffusion chain.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0): Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*): A random number generator.
        """

        sigma = self.sigmas[step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        device = model_output.device
        noise = torch.randn(model_output.shape, dtype=model_output.dtype, device=device, generator=generator).to(
            device
        )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if self.prediction_type == "original_sample" or self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif self.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat

        dt = self.sigmas[step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt

        return prev_sample

    def __len__(self):
        return self.num_train_timesteps
