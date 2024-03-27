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

import tqdm
import numpy as np


def gen_frequency_data(
    ds_dir: str = '/home/mlperf_inf_dlrmv2/criteo/day23/fp32/day_23_sparse_multi_hot_unpacked/',
    out_file: str = '/home/mlperf_inf_dlrmv2/criteo/day23/row_frequencies.npy'
):
    """Gather frequency data from training dataset and save to npy file."""
    print(f'Computing row_frequencies data from: {ds_dir}')

    mega = {}
    for file in tqdm.tqdm(os.listdir(ds_dir)):
        path = os.path.join(ds_dir, file)
        table = np.load(path)

        unique, freq = np.unique(table, return_counts=True)
        mega[path] = np.stack([unique, freq], axis=-1).astype(np.int32)

        # sorted_by_frequency = np.argsort(freq)[::-1]
        # sorted = np.stack([unique[sorted_by_frequency], freq[sorted_by_frequency]], axis=-1)
        # mega[path] = sorted

    row_frequencies = np.vstack(list(mega.values())).reshape(-1).astype(np.int32)
    with open(out_file, 'wb') as f:
        np.save(f, row_frequencies)

    print(f'Saved row_frequencies data to: {out_file}')


if __name__ == "__main__":
    gen_frequency_data()
