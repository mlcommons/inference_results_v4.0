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


import ctypes
import numpy as np
import os
import pycuda.autoprimaryctx
import pycuda.driver as cuda
import tensorrt as trt
import time

from glob import glob
from packaging import version

from code.common import logging
from code.common.constants import TRT_LOGGER
from typing import Optional


# WAR: Numpy 1.24.0+ has removed np.bool as a datatype. TensorRT 8.6.0.6 (MLPINF v3.0 release version) has
# not captured this, but was fixed in 8.6.1.
# https://gitlab-master.nvidia.com/TensorRT/TensorRT/-/commit/3a6e5deeef2848bba6e633390dabbf90e47121f0
if version.parse(trt.__version__) < version.parse("8.6.1"):
    # From TensorRT source code python/packaging/bindings_wheel/tensorrt/__init__.py:
    typemap = {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int8: np.int8,
        trt.int32: np.int32,
        trt.bool: np.bool_,
        trt.uint8: np.uint8,
    }
    nptype = typemap.get
else:
    nptype = trt.nptype


class HostDeviceMem(object):
    def __init__(self, host, device, tensor_name):
        self.host = host
        self.device = device
        self.tensor_name = tensor_name


def allocate_buffers(engine, profile_id):
    """Allocate device memory for I/O bindings of engine and return them."""
    d_inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    if engine.has_implicit_batch_dimension:
        max_batch_size = engine.max_batch_size
    else:
        shape = engine.get_binding_shape(0)
        if -1 in list(shape):
            batch_dim = list(shape).index(-1)
            max_batch_size = engine.get_profile_shape(0, 0)[2][batch_dim]
        else:
            max_batch_size = shape[0]
    nb_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    bindings = [0 for i in range(engine.num_bindings)]
    for binding in range(profile_id * nb_bindings_per_profile, (profile_id + 1) * nb_bindings_per_profile):
        logging.info("Binding {:}".format(binding))
        dtype = engine.get_binding_dtype(binding)
        format = engine.get_binding_format(binding)
        shape = engine.get_binding_shape(binding)
        if format in [trt.TensorFormat.CHW4, trt.TensorFormat.DLA_HWC4]:
            shape[-3] = ((shape[-3] - 1) // 4 + 1) * 4
        elif format in [trt.TensorFormat.CHW32]:
            shape[-3] = ((shape[-3] - 1) // 32 + 1) * 32
        elif format == trt.TensorFormat.DHWC8:
            shape[-4] = ((shape[-4] - 1) // 8 + 1) * 8
        elif format == trt.TensorFormat.CDHW32:
            shape[-4] = ((shape[-4] - 1) // 32 + 1) * 32
        if not engine.has_implicit_batch_dimension:
            if -1 in list(shape):
                batch_dim = list(shape).index(-1)
                shape[batch_dim] = max_batch_size
            size = trt.volume(shape)
        else:
            size = trt.volume(shape) * max_batch_size
        # Allocate device buffers
        device_mem = cuda.mem_alloc(size * dtype.itemsize)
        # Append device buffer to device bindings.
        bindings[binding] = int(device_mem)
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            d_inputs.append(device_mem)
        else:
            host_mem = cuda.pagelocked_empty(size, nptype(dtype))
            outputs.append(HostDeviceMem(host_mem, device_mem, None))
    return d_inputs, outputs, bindings, stream


def allocate_buffers_v2(engine: trt.ICudaEngine,
                        profile_id: int):
    """
    Allocate device memory based on the engine tensor I/O.
    Use TensorRT 8.6 API instead of the deprecated binding syntax.

    Returns:
        input_tensor_address_map (dic[name -> pycuda.driver.DeviceAllocation]): name - address pair for the Input Tensor
        outputs (List[HostDeviceMem]): output host-device address pair
        stream (pycuda.driver.Stream): the stream to synchronize the events
    """
    num_io_tensors = engine.num_io_tensors
    input_tensor_address_map = {}

    d_inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    if engine.has_implicit_batch_dimension:
        max_batch_size = engine.max_batch_size
    else:
        # NOTE: We are assuming that input idx 0 has dynamic shape, and the only dynamic dim is the batch dim,
        # which might not be true.
        tensor_name = engine.get_tensor_name(0)
        shape = engine.get_tensor_shape(tensor_name)
        if -1 in list(shape):
            batch_dim = list(shape).index(-1)
            max_batch_size = engine.get_profile_shape(profile_id, tensor_name)[2][batch_dim]
        else:
            max_batch_size = shape[0]

    for tensor_idx in range(num_io_tensors):
        tensor_name = engine.get_tensor_name(tensor_idx)
        tensor_io_type = engine.get_tensor_mode(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        tensor_format = engine.get_tensor_format(tensor_name)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        logging.info(f"Tensor idx: {tensor_idx}, Tensor name: {tensor_name}, iotype: {tensor_io_type}, dtype: {tensor_dtype}, format: {tensor_format}, shape: {tensor_shape}")

        if tensor_format in [trt.TensorFormat.CHW4, trt.TensorFormat.DLA_HWC4]:
            tensor_shape[-3] = ((tensor_shape[-3] - 1) // 4 + 1) * 4
        elif tensor_format in [trt.TensorFormat.CHW32]:
            tensor_shape[-3] = ((tensor_shape[-3] - 1) // 32 + 1) * 32
        elif tensor_format == trt.TensorFormat.DHWC8:
            tensor_shape[-4] = ((tensor_shape[-4] - 1) // 8 + 1) * 8
        elif tensor_format == trt.TensorFormat.CDHW32:
            tensor_shape[-4] = ((tensor_shape[-4] - 1) // 32 + 1) * 32
        if not engine.has_implicit_batch_dimension:
            if -1 in list(tensor_shape):
                batch_dim = list(tensor_shape).index(-1)
                tensor_shape[batch_dim] = max_batch_size
            size = trt.volume(tensor_shape)
        else:
            size = trt.volume(tensor_shape) * max_batch_size

        logging.info(f"Padded shape: {tensor_shape}")

        # Allocate device buffers
        device_mem = cuda.mem_alloc(size * tensor_dtype.itemsize)

        # Append to the appropriate list.
        if tensor_io_type == trt.TensorIOMode.OUTPUT:
            host_mem = cuda.pagelocked_empty(size, nptype(tensor_dtype))
            outputs.append(HostDeviceMem(host_mem, device_mem, tensor_name))
        elif tensor_io_type == trt.TensorIOMode.INPUT:
            input_tensor_address_map[tensor_name] = device_mem

    return input_tensor_address_map, outputs, stream


def get_input_format(engine):
    return engine.get_binding_dtype(0), engine.get_binding_format(0)


class EngineRunner:
    """Enable running inference through an engine on each call."""

    def __init__(self, engine_file, verbose=False, plugins=None, profile_id=0):
        """Load engine from file, allocate device memory for its bindings and create execution context."""

        self.engine_file = engine_file
        self.verbose = verbose
        if not os.path.exists(engine_file):
            raise ValueError("File {:} does not exist".format(engine_file))

        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        if plugins is not None:
            for plugin in plugins:
                ctypes.CDLL(plugin)
        self.engine = self.load_engine(engine_file)

        if profile_id < 0:
            profile_id = self.engine.num_optimization_profiles + profile_id

        self.input_tensor_map, self.outputs, self.stream = allocate_buffers_v2(self.engine, profile_id)
        self.context = self.engine.create_execution_context()

        # Set context tensor address (equivalent to set binding in old syntax)
        for name, addr in self.input_tensor_map.items():
            self.context.set_tensor_address(name, int(addr))
        for o in self.outputs:
            self.context.set_tensor_address(o.tensor_name, int(o.device))

        if profile_id > 0:
            self.context.active_optimization_profile = profile_id

    def load_engine(self, src_path):
        """Deserialize engine file to an engine and return it."""

        TRT_LOGGER.min_severity = trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO
        with open(src_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        return engine

    def __call__(self, inputs, batch_size=1):
        """Use host inputs to run inference on device and return back results to host."""

        # Copy input data to device bindings of context.
        profile_id = self.context.active_optimization_profile
        assert len(inputs) == len(self.input_tensor_map), \
            f"Feeding {len(inputs)} inputs, but the engine only has {len(self.input_tensor_map)} allocations."
        input_device_addrs = [self.input_tensor_map[self.engine.get_tensor_name(idx)] for idx in range(len(inputs))]
        [cuda.memcpy_htod_async(input_device_addr, inp, self.stream) for (input_device_addr, inp) in zip(input_device_addrs, inputs)]

        # Run inference.
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            for tensor_idx in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(tensor_idx)
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    input_shape = self.engine.get_tensor_shape(tensor_name)
                    if -1 in list(input_shape):
                        input_shape[0] = batch_size
                        self.context.set_input_shape(tensor_name, input_shape)
            self.context.execute_async_v3(self.stream.handle)

        # Copy output device buffers back to host.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

        # Synchronize the stream
        self.stream.synchronize()

        # Return only the host outputs.
        return [out.host for out in self.outputs]

    def __del__(self):
        # Clean up everything.
        with self.engine, self.context:
            [address.free() for name, address in self.input_tensor_map.items()]
            [out.device.free() for out in self.outputs]
            del self.stream
