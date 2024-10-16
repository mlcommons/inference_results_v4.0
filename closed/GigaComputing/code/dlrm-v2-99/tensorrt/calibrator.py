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

from pathlib import Path

import tensorrt as trt
import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoprimaryctx

from .criteo import CriteoDay23Dataset, CRITEO_SYNTH_MULTIHOT_SIZES


class DLRMv2Calibrator(trt.IInt8EntropyCalibrator2):
    """Calibrator for DLRMv2 benchmark."""

    def __init__(self,
                 data_dir,
                 calib_batch_size=256,
                 calib_max_batches=500,
                 force_calibration=False,
                 cache_file="code/dlrm-v2/tensorrt/calibrator.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.calib_batch_size = calib_batch_size
        self.calib_max_batches = calib_max_batches

        num_samples = calib_batch_size * calib_max_batches
        assert num_samples == 128000, "Calibration should be indices 89137319-89265318 inclusive"

        self.force_calibration = force_calibration
        self.current_idx = 0
        self.cache_file = Path(cache_file)

        dense_input_size = self.calib_batch_size * 13 * 4
        sparse_input_size = self.calib_batch_size * sum(CRITEO_SYNTH_MULTIHOT_SIZES) * 4
        self.device_input_numeric = cuda.mem_alloc(dense_input_size)
        self.device_input_cat = cuda.mem_alloc(sparse_input_size)

        self.lower_bound = 89137319
        self.upper_bound = 89265318

        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if not self.force_calibration and self.cache_file.exists():
            with self.cache_file.open(mode="rb") as f:
                self.cache = f.read()

        else:
            self.cache = None

            # Only load the dataset if we need to calibrate, since this is time intensive.
            self.ds = CriteoDay23Dataset(data_dir)

    def get_batch_size(self):
        return self.calib_batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_idx < self.calib_max_batches:
            # Get a slice of the dataset
            s = self.lower_bound + self.current_idx * self.calib_batch_size
            e = self.lower_bound + (self.current_idx + 1) * self.calib_batch_size
            print(f"Running calib batch {self.current_idx} with indices {s}:{e}")
            assert s < self.upper_bound and e <= self.upper_bound + 1

            batch = self.ds.get_batch(indices=np.arange(s, e))
            dense_input = np.ascontiguousarray(batch["dense"], dtype=np.float32)
            sparse_input = np.ascontiguousarray(np.hstack(batch["sparse"]), dtype=np.int32)

            cuda.memcpy_htod(self.device_input_numeric, dense_input)
            cuda.memcpy_htod(self.device_input_cat, sparse_input)

            self.current_idx += 1
            return [int(self.device_input_numeric), int(self.device_input_cat)]
        else:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        return self.cache

    def write_calibration_cache(self, cache):
        with self.cache_file.open(mode="wb") as f:
            f.write(cache)

    def clear_cache(self):
        self.cache = None

    def __del__(self):
        self.device_input_numeric.free()
        self.device_input_cat.free()
