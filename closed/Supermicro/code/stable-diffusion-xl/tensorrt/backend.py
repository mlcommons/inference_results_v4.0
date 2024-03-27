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

from __future__ import annotations

import threading
import queue
import array
import os
import time

import numpy as np
import tensorrt as trt
import torch

from pathlib import Path
from typing import Dict, List, Union
from importlib import import_module

from cuda import cudart
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from code.common import logging, run_command

# dash in stable-diffusion-xl breaks traditional way of module import
Dataset = import_module("code.stable-diffusion-xl.tensorrt.dataset").Dataset
PipelineConfig = import_module("code.stable-diffusion-xl.tensorrt.utilities").PipelineConfig
numpy_to_torch_dtype_dict = import_module("code.stable-diffusion-xl.tensorrt.utilities").numpy_to_torch_dtype_dict
calculate_max_engine_device_memory = import_module("code.stable-diffusion-xl.tensorrt.utilities").calculate_max_engine_device_memory
nvtx_profile_start = import_module("code.stable-diffusion-xl.tensorrt.utilities").nvtx_profile_start
nvtx_profile_stop = import_module("code.stable-diffusion-xl.tensorrt.utilities").nvtx_profile_stop
CUASSERT = import_module("code.stable-diffusion-xl.tensorrt.utilities").CUASSERT
CLIP = import_module("code.stable-diffusion-xl.tensorrt.network").CLIP
CLIPWithProj = import_module("code.stable-diffusion-xl.tensorrt.network").CLIPWithProj
UNetXL = import_module("code.stable-diffusion-xl.tensorrt.network").UNetXL
VAE = import_module("code.stable-diffusion-xl.tensorrt.network").VAE
sdxl_scheduler = import_module("code.stable-diffusion-xl.tensorrt.scheduler")

try:
    import mlperf_loadgen as lg
except:
    logging.warning("Loadgen Python bindings are not installed. Installing Loadgen Python bindings!")
    run_command("make build_loadgen")
    import mlperf_loadgen as lg


class SDXLEngine():
    """
    Sub-Engine/Network within SDXL pipeline.
    reads engine file, loads and activates execution context
    """

    def __init__(
        self,
        engine_name: str,
        engine_path: os.PathLike,
    ):
        self.engine_name = engine_name
        self.engine_path = engine_path

        logging.info(f"Loading TensorRT engine: {self.engine_path}.")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, device_memory: int):
        self.context = self.engine.create_execution_context_without_device_memory()
        self.context.device_memory = device_memory

        # NOTE(vir): need to call enable_cuda_graphs to switch to graph mode
        self.use_graphs = False

    def enable_cuda_graphs(self, buffers: SDXLBufferManager, stream: int = 2):
        '''
        enable cuda graphs for SDXLEngine.
        will capture graphs for all valid batch sizes.
        all subsequent calls to infer will now use cuda-graphs

        assumptions:
            - assumes activate() has been called already
            - buffers are staged already
        '''

        assert self.context is not None, "need to activate engine first"

        self.use_graphs = True
        self.cuda_graphs = {}
        logging.info(f'Enabling cuda graphs for {self.engine_name}')

        # get tensor names
        names = [self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)]
        num_inputs = sum([1 for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT])
        input_names, output_names = names[:num_inputs], names[num_inputs:]

        # get opt profiles
        input_profiles = [list(self.engine.get_tensor_profile_shape(name, 0)) for name in input_names]
        max_shapes = [list(profile[-1]) for profile in input_profiles]
        min_shapes = [list(profile[0]) for profile in input_profiles]
        opt_shapes = [list(profile[1]) for profile in input_profiles]

        # set engine BS bounds
        self.max_batch_size = max([shape[0] for shape in max_shapes])
        self.min_batch_size = max([shape[0] for shape in min_shapes])
        self.opt_batch_size = max([shape[0] for shape in opt_shapes])

        # helper func. flatten out input dict
        def yield_inputs(table: Union[torch.Tensor, dict, list]):
            for entry in table:
                if type(entry) is torch.Tensor:
                    yield entry

                elif type(entry) is dict:
                    for sub_entry in yield_inputs(list(entry.values())):
                        yield sub_entry

                elif type(entry) is list:
                    for sub_entry in entry:
                        yield sub_entry

                else:
                    assert False, "table not supported"

        # capture graph for each even in BS: [2 ... max_batch_size]
        for actual_batch_size in range(self.min_batch_size, self.max_batch_size + 1, 2):
            batch_size = actual_batch_size // 2

            # create and stage sample input
            sample_inputs = yield_inputs(UNetXL(max_batch_size=batch_size, precision='int8').get_sample_input(batch_size))
            for name, buffer in zip(input_names, sample_inputs):
                full_name = f'{self.engine_name}_{name}'
                buffers[full_name] = buffer

            # first run after reshape
            noerror = self.context.execute_async_v3(stream)
            if not noerror:
                raise ValueError(f"ERROR: inference failed.")

            # capture graph
            CUASSERT(cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
            self.context.execute_async_v3(stream)
            graph = CUASSERT(cudart.cudaStreamEndCapture(stream))
            self.cuda_graphs[actual_batch_size] = CUASSERT(cudart.cudaGraphInstantiate(graph, 0))

            # test first run of graph
            CUASSERT(cudart.cudaGraphLaunch(self.cuda_graphs[actual_batch_size], stream))
            logging.info(f'captured graph for BS={batch_size}')

    def stage_tensor(self, name: str, buffer: torch.Tensor, shape_override: List[int] = None):
        if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            assert self.context.set_input_shape(name, shape_override or list(buffer.shape))

        assert self.context.set_tensor_address(name, buffer.data_ptr())

    def infer(self, stream: Union[int, cudart.cudaStream_t], batch_size: int = None):
        if self.use_graphs:
            actual_batch_size = self.opt_batch_size if batch_size is None else batch_size * 2
            assert self.min_batch_size <= actual_batch_size <= self.max_batch_size

            # run using appropriate cuda graph
            CUASSERT(cudart.cudaGraphLaunch(self.cuda_graphs[actual_batch_size], stream))

        else:
            # run without cuda graph
            noerror = self.context.execute_async_v3(stream)
            if not noerror:
                raise ValueError(f"ERROR: {self.engine_path} inference failed.")

        return True


class SDXLBufferManager:
    """
    Buffer Manager for sdxl pipeline.
    manages sdxl engine buffers
    """

    def __init__(self, engines_dict: Dict[str, SDXLEngine], device: str = 'cuda'):
        self.engines_dict = engines_dict
        self.device = device

        self.buffers: Dict[str, torch.Tensor] = {}

        # inputs [
        #     'CLIP_input_ids',
        #     'CLIPWithProj_input_ids',
        #     'UNetXl_sample',
        #     'UNetXl_timestep',
        #     'UNetXl_encoder_hidden_states',
        #     'UNetXl_text_embeds',
        #     'UNetXl_time_ids',
        #     'UNetXl_latent',
        # ]
        self.input_tensors: Dict[str, List[int]] = {}

        # outputs [
        #     'CLIP_hidden_states',
        #     'CLIP_text_embeddings',
        #     'CLIPWithProj_hidden_states',
        #     'CLIPWithProj_text_embeddings',
        #     'UNetXl_latent',
        #     'VAE_images',
        # ]
        self.output_tensors: Dict[str, List[int]] = {}

    def to(self, device: str):
        assert device in ['cpu', 'cuda']
        self.device = device

        # put buffers on device
        for name, buffer in self.buffers.items():
            buffer.to(self.device)

            engine = self.engines_dict[name.split('_')[0]]
            tensor_name = '_'.join(name.split('_')[1:])

            engine.stage_tensor(tensor_name, buffer)

    def initialize(self, shape_dict: Dict[str, List[int]] = {}):
        # allocate all buffers, bookkeep shapes and setup with contexts
        for network_name, sdxl_engine in self.engines_dict.items():
            trt_engine = sdxl_engine.engine

            # [-1]: max opt profile [0]: batch dimension
            max_batch_size = trt_engine.get_tensor_profile_shape(trt_engine[0], 0)[-1][0]
            for idx in range(trt_engine.num_io_tensors):
                tensor_name = trt_engine[idx]
                full_name = f'{network_name}_{tensor_name}'

                tensor_shape = list(shape_dict.get(full_name, trt_engine.get_tensor_shape(tensor_name)))
                # set dynamic dimension
                if tensor_shape[0] == -1:
                    tensor_shape[0] = max_batch_size

                dtype = numpy_to_torch_dtype_dict[trt.nptype(trt_engine.get_tensor_dtype(tensor_name))]

                # NOTE(vir): use torch as storage/allocation backend for now
                self.buffers[full_name] = torch.empty(tensor_shape, dtype=dtype).to(device=self.device)
                sdxl_engine.stage_tensor(tensor_name, self.buffers[full_name])

                if trt_engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self.input_tensors[full_name] = tensor_shape

                if trt_engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                    self.output_tensors[full_name] = tensor_shape

    def get_dummy_feed_dict(self):
        feed_dict = {
            name: torch.rand(shape)
            for name, shape in self.input_tensors.items()
        }

        return feed_dict

    def stage_buffers(self, feed_dict: Dict[str, torch.Tensor] = None):
        feed_dict = feed_dict or self.get_dummy_feed_dict()
        [self.__setitem__(name, buf) for name, buf in feed_dict.items()]

    def get_outputs(self):
        return {output: self.buffers[output] for output in self.output_tensors.keys()}

    def get_input_names(self):
        return list(self.input_tensors.keys())

    def get_output_names(self):
        return list(self.output_tensors.keys())

    def __getitem__(self, buffer_name: str):
        assert buffer_name in self.buffers, f"invalid buffer identifier, no such buffer: {buffer_name}"
        return self.buffers[buffer_name]

    def __setitem__(self, buffer_name: str, tensor: torch.Tensor):
        assert buffer_name in self.buffers, f"invalid buffer identifier, can't add new buffer: {buffer_name}"

        max_batch_size = self.buffers[buffer_name].shape[0]
        actual_batch_size = 1 if len(tensor.shape) < 2 else tensor.shape[0]
        assert max_batch_size >= actual_batch_size, f"BS={actual_batch_size} must be <={max_batch_size} for {buffer_name}"

        # copy submatrix in
        self.buffers[buffer_name][:actual_batch_size].copy_(tensor)

        engine, tensor_name = self.engines_dict[buffer_name.split('_')[0]], '_'.join(buffer_name.split('_')[1:])
        tensor_mode = engine.engine.get_tensor_mode(tensor_name)

        # capture shape changes
        if tensor_mode == trt.TensorIOMode.INPUT:
            self.input_tensors[buffer_name] = list(tensor.shape)

        if tensor_mode == trt.TensorIOMode.OUTPUT:
            self.output_tensors[buffer_name] = list(tensor.shape)


# TODO: yihengz clean up test classes
class SDXLEngineTester():
    """
    Load sdxl subnets and setup test pipeline.
    """

    def __init__(
        self,
        engine_path: os.PathLike,
        engine_files: Dict[str, str],
    ):
        self.engines = {
            name: SDXLEngine(
                engine_name=name,
                engine_path=Path(engine_path) / file
            )
            for name, file in engine_files.items()
        }
        self.context_memory = None

    def initialize(self):
        # initialize runtime
        max_device_memory = calculate_max_engine_device_memory(self.engines)
        _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.context_memory = shared_device_memory

        for engine in self.engines.values():
            engine.activate(device_memory=self.context_memory)

    def infer(self, stream: int = 1, batch_size: int = None):
        # run inference on all sub-engines
        assert all([engine.infer(stream, batch_size) for engine in self.engines.values()])
        return True


class TestPipeline:
    def __init__(
        self,
        engine_path: os.PathLike = '/work/build/engines',
        engine_files: Dict[str, str] = {
            'CLIP': 'stable-diffusion-xl-CLIP-Offline-gpu-b1-fp16.custom_k_99_MaxP.plan',
            'CLIPWithProj': 'stable-diffusion-xl-CLIPWithProj-Offline-gpu-b1-fp16.custom_k_99_MaxP.plan',
            'UNetXl': 'stable-diffusion-xl-UNetXL-Offline-gpu-b1-fp16.custom_k_99_MaxP.plan',
            'VAE': 'stable-diffusion-xl-VAE-Offline-gpu-b1-fp32.custom_k_99_MaxP.plan',
        }
    ):
        self.test_engine = SDXLEngineTester(engine_path, engine_files)
        self.buffers = SDXLBufferManager(self.test_engine.engines)

        # initialize on construct
        # allocate memory and setup for execution
        self.test_engine.initialize()
        self.buffers.initialize()

        # enable cuda graphs in unet for test pipeline
        self.test_engine.engines['UNetXl'].enable_cuda_graphs(self.buffers)

    def infer(self, feed_dict=None):
        self.buffers.stage_buffers(feed_dict)
        assert self.test_engine.infer()
        return self.buffers.get_outputs()


class SDXLResponse:
    def __init__(self,
                 sample_ids,
                 generated_images,
                 results_ready):
        self.sample_ids = sample_ids
        self.generated_images = generated_images
        self.results_ready = results_ready


class SDXLCopyStream:
    def __init__(self, device_id, gpu_batch_size):
        CUASSERT(cudart.cudaSetDevice(device_id))
        self.stream = CUASSERT(cudart.cudaStreamCreate())
        self.h2d_event = CUASSERT(cudart.cudaEventCreateWithFlags(cudart.cudaEventDefault | cudart.cudaEventDisableTiming))
        self.d2h_event = CUASSERT(cudart.cudaEventCreateWithFlags(cudart.cudaEventDefault | cudart.cudaEventDisableTiming))
        self.vae_outputs = torch.zeros((gpu_batch_size, PipelineConfig.IMAGE_SIZE, PipelineConfig.IMAGE_SIZE, 3), dtype=torch.uint8)  # output cpu buffer

    def save_buffer_to_cpu_images(self, vae_ouput_buffer):
        # Normalize TRT (output + 1) * 0.5
        # Post process following the reference: https://github.com/mlcommons/inference/blob/master/text_to_image/coco.py
        vae_output_post_processed = ((vae_ouput_buffer + 1) * 255 * 0.5).round().permute(0, 2, 3, 1).to(torch.uint8).contiguous()

        cudart.cudaMemcpyAsync(self.vae_outputs.data_ptr(),
                               vae_output_post_processed.data_ptr(),
                               vae_output_post_processed.nelement() * vae_output_post_processed.element_size(),
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                               self.stream)

    def record_h2d_event(self):
        CUASSERT(cudart.cudaEventRecord(self.h2d_event, self.stream))

    def record_d2h_event(self):
        CUASSERT(cudart.cudaEventRecord(self.d2h_event, self.stream))

    def make_infer_await_h2d(self, infer_stream):
        CUASSERT(cudart.cudaStreamWaitEvent(infer_stream, self.h2d_event, 0))

    def await_infer_done(self, infer_done):
        CUASSERT(cudart.cudaStreamWaitEvent(self.stream, infer_done, 0))


class SDXLCore:
    def __init__(self,
                 device_id: int,
                 dataset: Dataset,
                 gpu_engine_files: List[str],
                 gpu_batch_size: int,
                 gpu_copy_streams: int = 1,  # TODO copy stream number limit to 1
                 use_graphs: bool = False,
                 verbose: bool = False,
                 verbose_nvtx: bool = False):

        CUASSERT(cudart.cudaSetDevice(device_id))
        torch.autograd.set_grad_enabled(False)
        self.device = "cuda"
        self.device_id = device_id
        self.engine_files = gpu_engine_files
        self.gpu_batch_size = gpu_batch_size
        self.use_graphs = use_graphs
        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx

        self._verbose_info(f"[Device {self.device_id}] Initializing")

        # NVTX components
        if self.verbose_nvtx:
            self.nvtx_markers = {}

        # Dataset
        self.dataset = dataset
        self.total_samples = 0

        # Pipeline components
        self.models = {
            'clip1': CLIP(max_batch_size=gpu_batch_size, device=self.device),
            'clip2': CLIPWithProj(max_batch_size=gpu_batch_size, device=self.device),
            'unet': UNetXL(max_batch_size=gpu_batch_size, device=self.device),
            'vae': VAE(max_batch_size=gpu_batch_size, device=self.device),
        }
        self.engines = {}
        self.buffers = None
        self.scheduler = sdxl_scheduler.EulerDiscreteScheduler()
        self.latent_dtype = torch.float16

        # Runtime components
        self.context_memory = None
        self.infer_stream = CUASSERT(cudart.cudaStreamCreate())
        self.infer_done = CUASSERT(cudart.cudaEventCreateWithFlags(cudart.cudaEventDefault | cudart.cudaEventDisableTiming))
        self.copy_stream = SDXLCopyStream(device_id, gpu_batch_size)

        # QSR components
        self.response_queue = queue.Queue()
        self.response_thread = threading.Thread(target=self._process_response, args=(), daemon=True)
        # self.start_inference = threading.Condition()

        # Initialize scheduler
        self.scheduler.set_timesteps(PipelineConfig.STEPS)

        # Initialize engines
        for i, name in enumerate(self.models.keys()):
            self.engines[name] = SDXLEngine(engine_name=name, engine_path=self.engine_files[i])

        # Initialize engine runtime
        max_device_memory = calculate_max_engine_device_memory(self.engines)
        shared_device_memory = CUASSERT(cudart.cudaMalloc(max_device_memory))
        self.context_memory = shared_device_memory

        for engine in self.engines.values():
            self._verbose_info(f"Activating engine: {engine.engine_path}")
            engine.activate(self.context_memory)

        # Initialize buffers
        self.buffers = SDXLBufferManager(self.engines, device=self.device)
        self.buffers.initialize()
        self.add_time_ids = torch.tensor(
            [PipelineConfig.IMAGE_SIZE, PipelineConfig.IMAGE_SIZE, 0, 0, PipelineConfig.IMAGE_SIZE, PipelineConfig.IMAGE_SIZE],
            dtype=torch.float16,
            device=self.device).repeat(gpu_batch_size * 2, 1)
        self.init_noise_latent = self.dataset.init_noise_latent.to(self.device)
        self.init_noise_latent = torch.concat([self.init_noise_latent] * gpu_batch_size) * self.scheduler.init_noise_sigma()

        # Initialize cuda graphs
        if self.use_graphs:
            self.engines['unet'].enable_cuda_graphs(self.buffers)

        # Initialize QSR thread
        self.response_thread.start()

    def __del__(self):
        # exit all threads
        self.response_queue.put(None)
        self.response_queue.join()
        self.response_thread.join()

    def _verbose_info(self, msg):
        if self.verbose:
            logging.info(msg)

    def _process_response(self):
        while True:
            response = self.response_queue.get()
            if response is None:
                # None in the queue indicates the parent want us to exit
                self.response_queue.task_done()
                break
            qsr = []
            actual_batch_size = len(response.sample_ids)
            CUASSERT(cudart.cudaEventSynchronize(response.results_ready))
            self._verbose_info(f"[Device {self.device_id}] Reporting back {actual_batch_size} samples")

            if self.verbose_nvtx:
                nvtx_profile_start("report_qsl", self.nvtx_markers, color='yellow')

            for idx, sample_id in enumerate(response.sample_ids):
                qsr.append(lg.QuerySampleResponse(sample_id,
                                                  response.generated_images[idx].data_ptr(),
                                                  response.generated_images[idx].nelement() * response.generated_images[idx].element_size()))

            lg.QuerySamplesComplete(qsr)
            self.total_samples += actual_batch_size
            self.response_queue.task_done()
            if self.verbose_nvtx:
                nvtx_profile_stop("report_qsl", self.nvtx_markers)

    def _transfer_to_clip_buffer(self, prompt_tokens_clip1, prompt_tokens_clip2, negative_prompt_tokens_clip1, negative_prompt_tokens_clip2):
        # TODO: yihengz support copy stream
        # cudart.cudaMemcpy(self.buffers['clip1'].get_tensor('input_ids').data_ptr(),
        #                   prompt_tokens_clip1.ctypes.data,
        #                   prompt_tokens_clip1.size * prompt_tokens_clip1.itemsize,
        #                   cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # [ negative prompt, prompt ]
        concat_prompt_clip1 = torch.concat([negative_prompt_tokens_clip1, prompt_tokens_clip1], dim=0)
        concat_prompt_clip2 = torch.concat([negative_prompt_tokens_clip2, prompt_tokens_clip2], dim=0)
        self.buffers['clip1_input_ids'] = concat_prompt_clip1
        self.buffers['clip2_input_ids'] = concat_prompt_clip2

    def _encode_tokens(self, actual_batch_size):
        clip_models = ['clip1', 'clip2']
        for clip in clip_models:
            for tensor_name, tensor_shape in self.models[clip].get_shape_dict(actual_batch_size).items():
                self.engines[clip].stage_tensor(tensor_name, self.buffers[f'{clip}_{tensor_name}'], tensor_shape)
            self.engines[clip].infer(self.infer_stream)

    def _denoise_latent(self, actual_batch_size):
        # Prepare predetermined input tensors
        if self.verbose_nvtx:
            nvtx_profile_start("prepar_denoise", self.nvtx_markers, color='yellow')
        latents = self.init_noise_latent[:actual_batch_size]
        encoder_hidden_states = torch.concat([
            self.buffers['clip1_hidden_states'],
            self.buffers['clip2_hidden_states'].to(self.latent_dtype)
        ], dim=-1)
        text_embeds = self.buffers['clip2_text_embeddings'].to(self.latent_dtype)

        self.buffers['unet_encoder_hidden_states'] = encoder_hidden_states
        self.buffers['unet_text_embeds'] = text_embeds
        self.buffers['unet_time_ids'] = self.add_time_ids[:actual_batch_size * 2]

        if self.verbose_nvtx:
            nvtx_profile_stop("prepar_denoise", self.nvtx_markers)

        for step_index, timestep in enumerate(self.scheduler.timesteps):
            if self.verbose_nvtx:
                nvtx_profile_start("stage_denoise", self.nvtx_markers, color='pink')
            # Expand the latents because we have prompt and negative prompt guidance
            latents_expanded = self.scheduler.scale_model_input(torch.concat([latents] * 2), step_index, timestep)

            # Prepare runtime dependent input tensors
            self.buffers['unet_sample'] = latents_expanded.to(self.latent_dtype)
            self.buffers['unet_timestep'] = timestep.to(self.latent_dtype).to("cuda")

            for tensor_name, tensor_shape in self.models['unet'].get_shape_dict(actual_batch_size).items():
                self.engines['unet'].stage_tensor(tensor_name, self.buffers[f'unet_{tensor_name}'], tensor_shape)

            if self.verbose_nvtx:
                nvtx_profile_stop("stage_denoise", self.nvtx_markers)
                nvtx_profile_start("denoise_infer", self.nvtx_markers, color='green')

            self.engines['unet'].infer(self.infer_stream, batch_size=actual_batch_size)

            # TODO: yihengz check if we actually need sync the stream
            CUASSERT(cudart.cudaStreamSynchronize(self.infer_stream))  # make sure Unet kernel execution are finished

            if self.verbose_nvtx:
                nvtx_profile_stop("denoise_infer", self.nvtx_markers)
                nvtx_profile_start("scheduler", self.nvtx_markers, color='pink')
            # Perform guidance
            noise_pred = self.buffers['unet_latent']

            noise_pred_negative_prompt = noise_pred[0:actual_batch_size]  # negative prompt in batch dimension [0:BS]
            noise_pred_prompt = noise_pred[actual_batch_size:actual_batch_size * 2]  # prompt in batch dimension [BS:]

            noise_pred = noise_pred_negative_prompt + PipelineConfig.GUIDANCE * (noise_pred_prompt - noise_pred_negative_prompt)

            latents = self.scheduler.step(noise_pred, latents, step_index)
            if self.verbose_nvtx:
                nvtx_profile_stop("scheduler", self.nvtx_markers)

        latents = 1. / PipelineConfig.VAE_SCALING_FACTOR * latents
        # Transfer the Unet output to vae buffer
        self.buffers['vae_latent'] = latents

    def _decode_latent(self, actual_batch_size):
        if self.verbose_nvtx:
            nvtx_profile_start("vae_decode", self.nvtx_markers, color='blue')
        for tensor_name, tensor_shape in self.models['vae'].get_shape_dict(actual_batch_size).items():
            self.engines['vae'].stage_tensor(tensor_name, self.buffers[f'vae_{tensor_name}'], tensor_shape)
        self.engines['vae'].infer(self.infer_stream)
        CUASSERT(cudart.cudaEventRecord(self.infer_done, self.infer_stream))
        if self.verbose_nvtx:
            nvtx_profile_stop("vae_decode", self.nvtx_markers)

    def _save_buffer_to_images(self):
        if self.verbose_nvtx:
            nvtx_profile_start("post_process", self.nvtx_markers, color='yellow')
        self.copy_stream.await_infer_done(self.infer_done)
        self.copy_stream.save_buffer_to_cpu_images(self.buffers['vae_images'])
        self.copy_stream.record_d2h_event()
        if self.verbose_nvtx:
            nvtx_profile_stop("post_process", self.nvtx_markers)

    def generate_images(self, samples):
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        if self.verbose_nvtx:
            nvtx_profile_start("read_tokens", self.nvtx_markers, color='yellow')
        actual_batch_size = len(samples)
        sample_indices = [q.index for q in samples]
        sample_ids = [q.id for q in samples]
        self._verbose_info(f"[Device {self.device_id}] Running inference on sample {sample_indices} with batch size {actual_batch_size}")

        # TODO add copy stream support
        prompt_tokens_clip1 = self.dataset.prompt_tokens_clip1[sample_indices, :].to(self.device)
        prompt_tokens_clip2 = self.dataset.prompt_tokens_clip2[sample_indices, :].to(self.device)
        negative_prompt_tokens_clip1 = self.dataset.negative_prompt_tokens_clip1[sample_indices, :].to(self.device)
        negative_prompt_tokens_clip2 = self.dataset.negative_prompt_tokens_clip2[sample_indices, :].to(self.device)

        if self.verbose_nvtx:
            nvtx_profile_stop("read_tokens", self.nvtx_markers)
            nvtx_profile_start("stage_clip_buffers", self.nvtx_markers, color='pink')
        self._transfer_to_clip_buffer(
            prompt_tokens_clip1,
            prompt_tokens_clip2,
            negative_prompt_tokens_clip1,
            negative_prompt_tokens_clip2
        )
        if self.verbose_nvtx:
            nvtx_profile_stop("stage_clip_buffers", self.nvtx_markers)
            # nvtx_profile_start("generate_images", self.nvtx_markers)

        self._encode_tokens(actual_batch_size)
        self._denoise_latent(actual_batch_size)  # runs self.denoising_steps inside
        self._decode_latent(actual_batch_size)

        self._save_buffer_to_images()

        # Report back to loadgen use sample_ids
        response = SDXLResponse(sample_ids=sample_ids,
                                generated_images=self.copy_stream.vae_outputs,
                                results_ready=self.copy_stream.d2h_event)
        self.response_queue.put(response)

    def warm_up(self, warm_up_iters):
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        self._verbose_info(f"[Device {self.device_id}] Running warm up with batch size {self.gpu_batch_size}x{warm_up_iters}")

        for _ in range(warm_up_iters):
            prompt_tokens_clip1 = self.dataset.prompt_tokens_clip1[:self.gpu_batch_size, :].to(self.device)
            prompt_tokens_clip2 = self.dataset.prompt_tokens_clip2[:self.gpu_batch_size, :].to(self.device)
            negative_prompt_tokens_clip1 = self.dataset.negative_prompt_tokens_clip1[:self.gpu_batch_size, :].to(self.device)
            negative_prompt_tokens_clip2 = self.dataset.negative_prompt_tokens_clip2[:self.gpu_batch_size, :].to(self.device)

            self._transfer_to_clip_buffer(
                prompt_tokens_clip1,
                prompt_tokens_clip2,
                negative_prompt_tokens_clip1,
                negative_prompt_tokens_clip2
            )

            self._encode_tokens(self.gpu_batch_size)
            self._denoise_latent(self.gpu_batch_size)
            self._decode_latent(self.gpu_batch_size)

            self._save_buffer_to_images()


class SDXLServer:
    def __init__(self,
                 devices: List[int],
                 dataset: Dataset,
                 gpu_engine_files: List[str],
                 gpu_batch_size: int,
                 gpu_inference_streams: int = 1,  # TODO support multiple SDXLCore per device
                 gpu_copy_streams: int = 1,  # TODO copy stream number limit to 1
                 use_graphs: bool = False,
                 verbose: bool = False,
                 verbose_nvtx: bool = False,
                 enable_batcher: bool = False,
                 batch_timeout_threashold: float = -1):

        self.devices = devices
        self.gpu_batch_size = gpu_batch_size
        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx
        self.enable_batcher = enable_batcher and batch_timeout_threashold > 0

        # NVTX components
        if self.verbose_nvtx:
            self.nvtx_markers = {}

        # Server components
        self.sample_queue = queue.Queue()  # sample sync queue
        self.sample_count = 0
        self.sdxl_cores = {}
        self.core_threads = []

        # Initialize the cores
        for device_id in self.devices:
            self.sdxl_cores[device_id] = SDXLCore(device_id=device_id,
                                                  dataset=dataset,
                                                  gpu_engine_files=gpu_engine_files,
                                                  gpu_batch_size=self.gpu_batch_size,
                                                  gpu_copy_streams=gpu_copy_streams,
                                                  use_graphs=use_graphs,
                                                  verbose=self.verbose,
                                                  verbose_nvtx=self.verbose_nvtx)

        # Start the cores
        for device_id in self.devices:
            thread = threading.Thread(target=self.process_samples, args=(device_id,))
            # thread.daemon = True
            self.core_threads.append(thread)
            thread.start()

        if self.enable_batcher:
            self.batcher_threshold = batch_timeout_threashold  # maximum seconds to form a batch
            self.batcher_queue = queue.SimpleQueue()  # batcher sync queue
            self.batcher_thread = threading.Thread(target=self.batch_samples, args=())
            self.batcher_thread.start()

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def _verbose_info(self, msg):
        if self.verbose:
            logging.info(msg)

    def warm_up(self):
        for device_id in self.devices:
            self.sdxl_cores[device_id].warm_up(warm_up_iters=2)

    def process_samples(self, device_id):
        while True:
            samples = self.sample_queue.get()
            if samples is None:
                # None in the queue indicates the SUT want us to exit
                self.sample_queue.task_done()
                break
            self.sdxl_cores[device_id].generate_images(samples)
            self.sample_queue.task_done()

    def batch_samples(self):
        batched_samples = self.batcher_queue.get()
        timeout_stamp = time.time()
        while True:
            if len(batched_samples) != 0 and (len(batched_samples) >= self.gpu_batch_size or time.time() - timeout_stamp >= self.batcher_threshold):  # max batch or time limit exceed
                self._verbose_info(f"Formed batch of {len(batched_samples[:self.gpu_batch_size])} samples")
                self.sample_queue.put(batched_samples[:self.gpu_batch_size])
                batched_samples = batched_samples[self.gpu_batch_size:]
                timeout_stamp = time.time()

            try:
                samples = self.batcher_queue.get(timeout=self.batcher_threshold)
            except queue.Empty:
                continue

            if samples is None:  # None in the queue indicates the SUT want us to exit
                break
            batched_samples += samples

    def issue_queries(self, query_samples):
        num_samples = len(query_samples)
        self._verbose_info(f"[Server] Received {num_samples} samples")
        self.sample_count += num_samples
        for i in range(0, num_samples, self.gpu_batch_size):
            # Construct batches
            actual_batch_size = self.gpu_batch_size if num_samples - i > self.gpu_batch_size else num_samples - i
            if self.enable_batcher:
                self.batcher_queue.put(query_samples[i: i + actual_batch_size])
            else:
                self.sample_queue.put(query_samples[i: i + actual_batch_size])

    def flush_queries(self):
        pass

    def finish_test(self):
        # exit all threads
        self._verbose_info(f"SUT finished!")
        logging.info(f"[Server] Received {self.sample_count} total samples")
        for _ in self.core_threads:
            self.sample_queue.put(None)
        self.sample_queue.join()
        if self.enable_batcher:
            self.batcher_queue.put(None)
            self.batcher_thread.join()
        for device_id in self.devices:
            logging.info(f"[Device {device_id}] Reported {self.sdxl_cores[device_id].total_samples} samples")
        for thread in self.core_threads:
            thread.join()


if __name__ == '__main__':
    test_model = TestPipeline(
        engine_path='/work/build/engines/H100-SXM-80GB-02x1/stable-diffusion-xl/Offline',
        engine_files={
            'CLIP': 'stable-diffusion-xl-CLIP-Offline-gpu-b8-fp16.custom_k_99_MaxP.plan',
            'CLIPWithProj': 'stable-diffusion-xl-CLIPWithProj-Offline-gpu-b8-fp16.custom_k_99_MaxP.plan',
            'UNetXl': 'stable-diffusion-xl-UNetXL-Offline-gpu-b8-int8.custom_k_99_MaxP.plan',
            'VAE': 'stable-diffusion-xl-VAE-Offline-gpu-b8-fp32.custom_k_99_MaxP.plan',
        }
    )
    outputs = test_model.infer()
