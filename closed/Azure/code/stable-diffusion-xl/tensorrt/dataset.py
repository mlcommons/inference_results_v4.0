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
import logging
import torch
import numpy as np

from pathlib import Path


class Dataset:
    def __init__(self, tensor_path):
        self.prompt_tokens_clip1 = torch.tensor(np.load(Path(tensor_path, "prompt_ids_clip1_padded_5k.npy"))).pin_memory()
        self.prompt_tokens_clip2 = torch.tensor(np.load(Path(tensor_path, "prompt_ids_clip2_padded_5k.npy"))).pin_memory()
        self.negative_prompt_tokens_clip1 = torch.tensor(np.load(Path(tensor_path, "negative_prompt_ids_clip1_padded_5k.npy"))).pin_memory()
        self.negative_prompt_tokens_clip2 = torch.tensor(np.load(Path(tensor_path, "negative_prompt_ids_clip2_padded_5k.npy"))).pin_memory()
        self.init_noise_latent = torch.load(Path(tensor_path, "latents.pt"))

        self.caption_count = self.prompt_tokens_clip1.shape[0]

    def load_query_samples(self, sample_list):
        pass

    def unload_query_samples(self, sample_list):
        pass
