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
from importlib import import_module
from os import PathLike
from pathlib import Path

import onnx
import tempfile
import tensorrt as trt
import polygraphy.logger
from importlib import import_module

from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_network,
    modify_network_outputs,
    save_engine
)

from nvmitten.constants import Precision
from nvmitten.nvidia.builder import (TRTBuilder,
                                     MLPerfInferenceEngine,
                                     LegacyBuilder)
from nvmitten.pipeline import Operation
from nvmitten.utils import dict_get, logging

from code.common.mitten_compat import ArgDiscarder
from code.common.systems.system_list import SystemClassifications

# dash in stable-diffusion-xl breaks traditional way of module import
AbstractModel = import_module("code.stable-diffusion-xl.tensorrt.network").AbstractModel
CLIP = import_module("code.stable-diffusion-xl.tensorrt.network").CLIP
CLIPWithProj = import_module("code.stable-diffusion-xl.tensorrt.network").CLIPWithProj
UNetXL = import_module("code.stable-diffusion-xl.tensorrt.network").UNetXL
VAE = import_module("code.stable-diffusion-xl.tensorrt.network").VAE
SDXLGraphSurgeon = import_module("code.stable-diffusion-xl.tensorrt.sdxl_graphsurgeon").SDXLGraphSurgeon
polygraphy.logger.G_LOGGER.module_severity = polygraphy.logger.G_LOGGER.ERROR


class SDXLBaseBuilder(TRTBuilder,
                      MLPerfInferenceEngine):
    """Base SDXL builder class.
    """

    def __init__(self,
                 *args,
                 model: AbstractModel,
                 model_path: str,
                 batch_size: int = 1,
                 workspace_size: int = 80 << 30,
                 num_profiles: int = 1,  # Unused. Forcibly overridden to 1.
                 device_type: str = "gpu",
                 **kwargs):
        # TODO: yihengz Force num_profiles to 1 for SDXL, not sure if multiple execution context can help heavy benchmarks
        super().__init__(*args, num_profiles=1, workspace_size=workspace_size, **kwargs)

        self.model = model
        self.model_path = model_path
        # engine precision is determined by the model
        if self.model.precision == 'fp32':
            self.precision = Precision.FP32
        elif self.model.precision == 'fp16':
            self.precision = Precision.FP16
        elif self.model.precision == 'int8':
            self.precision = Precision.INT8
        else:
            raise ValueError("Unsupported model precision")
        self.batch_size = batch_size
        self.device_type = device_type

    def create_network(self, use_native_instance_norm):
        add_hidden_states = isinstance(self.model, CLIP) or isinstance(self.model, CLIPWithProj)
        sdxl_gs = SDXLGraphSurgeon(self.model_path,
                                   self.precision,
                                   self.device_type,
                                   self.model.name,
                                   add_hidden_states=add_hidden_states)

        model = sdxl_gs.create_onnx_model()

        network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        if use_native_instance_norm:
            parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)

        if model.ByteSize() >= SDXLGraphSurgeon.ONNX_LARGE_FILE_THRESHOLD:
            # onnx._serialize cannot take input proto >= 2 BG
            # We need to save proto larger than 2GB into separate files and parse from files
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                tmp_path.mkdir(exist_ok=True)
                onnx_tmp_path = tmp_path / "tmp_model.onnx"
                onnx.save_model(model,
                                str(onnx_tmp_path),
                                save_as_external_data=True,
                                all_tensors_to_one_file=True,
                                convert_attribute=False)
                success = parser.parse_from_file(str(onnx_tmp_path))
                if not success:
                    err_desc = parser.get_error(0).desc()
                    raise RuntimeError(f"Parse SDXL graphsurgeon onnx model failed! Error: {err_desc}")
        else:
            # Parse from ONNX file
            # set instance norm flag for better perf of SDXL
            success = parser.parse(onnx._serialize(model))
            if not success:
                err_desc = parser.get_error(0).desc()
                raise RuntimeError(f"Parse SDXL graphsurgeon onnx model failed! Error: {err_desc}")

        logging.info(f"Updating network outputs to {self.model.get_output_names()}")
        _, network, _ = modify_network_outputs((self.builder, network, parser), self.model.get_output_names())

        self.apply_network_io_types(network)
        return network

    def apply_network_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for network inputs and outputs to the tensorrt.INetworkDefinition.
        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        # Set input dtype
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            if self.precision == Precision.FP32:
                input_tensor.dtype = trt.float32
            elif self.precision == Precision.FP16 or self.precision == Precision.INT8:
                input_tensor.dtype = trt.float16

        # Set output dtype
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            if self.precision == Precision.FP32:
                output_tensor.dtype = trt.float32
            elif self.precision == Precision.FP16 or self.precision == Precision.INT8:
                output_tensor.dtype = trt.float16

    # Overwrites mitten function with the same signature, function parameters are unused
    def gpu_profiles(self,
                     network: trt.INetworkDefinition,
                     builder_config: trt.IBuilderConfig,
                     batch_size: int):
        profile = Profile()
        input_profile = self.model.get_input_profile(batch_size)
        for name, dims in input_profile.items():
            assert len(dims) == 3
            profile.add(name, min=dims[0], opt=dims[1], max=dims[2])
        self.profiles = [profile]

    def engine_name(self,
                    device_type: str,
                    batch_size: int,
                    precision: Precision,
                    tag: str = "default") -> str:
        """Gets the name of the engine, constructed from the device it is build for, the explicit batch size, and an
        optional tag.

        Args:
            device_type (str): The device that TRT is building the engine for. Either "gpu" or "dla".
            batch_size (int): The max batch size / explicit batch size the engine is built with.
            precision (str): The lowest precision enabled for the engine.
            tag (str): A tag to use for the engine. (Default: "default")

        Returns:
            str: The name of the engine.
        """
        if not precision:
            if hasattr(self, "precision"):
                # TODO: self.precision is currently a string, but in the case it is an AliasedNameEnum member, add support
                # to use .valstr()
                precision = self.precision
            else:
                raise ValueError("precision cannot be None if self.precision is not set.")

        name = self.benchmark.valstr()
        scenario = self.scenario.valstr()
        model = self.model.name
        precision = precision.valstr()
        return f"{name}-{model}-{scenario}-{device_type}-b{batch_size}-{precision}.{tag}.plan"

    def create_builder_config(self, *args, **kwargs) -> trt.IBuilderConfig:
        config_kwargs = {}
        # TODO：yihengz explore if we can enable cudnn/cublas for better perf
        # config_kwargs['tactic_sources'] = []
        # TODO: yihengz explore if builder_optimization_level = 5 can get better perf, disabling for making engine build time too long
        # config_kwargs['builder_optimization_level'] = 4
        config_kwargs['int8'] = self.precision == Precision.INT8
        config_kwargs['fp16'] = self.precision == Precision.FP16 or self.precision == Precision.INT8
        config_kwargs['tf32'] = self.precision == Precision.FP32
        config_kwargs['profiling_verbosity'] = trt.ProfilingVerbosity.DETAILED if self.verbose or self.verbose_nvtx else trt.ProfilingVerbosity.LAYER_NAMES_ONLY
        config_kwargs['memory_pool_limits'] = {trt.MemoryPoolType.WORKSPACE: self.workspace_size}

        self.create_profiles(network=None, builder_config=None, batch_size=self.batch_size)
        builder_config = CreateConfig(profiles=self.profiles, **config_kwargs)

        return builder_config

    def build_engine(self,
                     network: trt.INetworkDefinition,
                     builder_config: trt.IBuilderConfig,  # created inside
                     batch_size: int,  # determined by self.batch_size
                     save_to: PathLike):
        save_to = Path(save_to)
        if save_to.is_file():
            logging.warning(f"{save_to} already exists. This file will be overwritten")
        save_to.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Building TensorRT engine for {self.model_path}: {save_to}")

        engine = engine_from_network(
            (self.builder, network),
            config=builder_config,
        )

        engine_inspector = engine.create_engine_inspector()
        layer_info = engine_inspector.get_engine_information(trt.LayerInformationFormat.ONELINE)
        logging.info("========= TensorRT Engine Layer Information =========")
        logging.info(layer_info)

        # [https://nvbugs/3965323] Need to delete the engine inspector to release the refcount
        del engine_inspector

        save_engine(engine, path=save_to)

# TODO yihengz check if we can remove max_batch_size


class SDXLCLIPBuilder(SDXLBaseBuilder,
                      ArgDiscarder):
    """SDXL CLIP builder class.
    """

    def __init__(self,
                 *args,
                 batch_size: int,
                 model_path: PathLike,
                 **kwargs):
        super().__init__(*args,
                         model=CLIP(max_batch_size=batch_size, precision='fp16', device='cuda'),
                         model_path=model_path,
                         batch_size=batch_size,
                         **kwargs)

    def apply_network_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for network inputs and outputs to the tensorrt.INetworkDefinition.
        CLIP keeps int32 input (tokens)
        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        # Set output dtype
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            if self.precision == Precision.FP32:
                output_tensor.dtype = trt.float32
            elif self.precision == Precision.FP16:
                output_tensor.dtype = trt.float16


class SDXLCLIPWithProjBuilder(SDXLBaseBuilder,
                              ArgDiscarder):
    """SDXL CLIPWithProj builder class.
    """

    def __init__(self,
                 *args,
                 batch_size: int,
                 model_path: PathLike,
                 **kwargs):
        clip_with_proj_precision = 'fp32' if SystemClassifications.is_orin() else 'fp16'
        super().__init__(*args,
                         model=CLIPWithProj(max_batch_size=batch_size, precision=clip_with_proj_precision, device='cuda'),
                         model_path=model_path,
                         batch_size=batch_size,
                         **kwargs)

    def apply_network_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for network inputs and outputs to the tensorrt.INetworkDefinition.
        CLIPWithProj keeps int32 input (tokens)
        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        # Set output dtype
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            if self.precision == Precision.FP32:
                output_tensor.dtype = trt.float32
            elif self.precision == Precision.FP16:
                output_tensor.dtype = trt.float16


class SDXLUNetXLBuilder(SDXLBaseBuilder,
                        ArgDiscarder):
    """SDXL UNetXL builder class.
    """

    def __init__(self,
                 *args,
                 batch_size: int,  # *2 for prompt + negative prompt
                 precision: str,
                 model_path: PathLike,
                 **kwargs):
        super().__init__(*args,
                         model=UNetXL(max_batch_size=batch_size, precision=precision, device='cuda'),
                         model_path=model_path,
                         batch_size=batch_size,
                         **kwargs)


class SDXLVAEBuilder(SDXLBaseBuilder,
                     ArgDiscarder):
    """SDXL VAE builder class.
    """

    def __init__(self,
                 *args,
                 batch_size: int,
                 model_path: PathLike,
                 **kwargs):
        super().__init__(*args,
                         model=VAE(max_batch_size=batch_size, precision='fp32', device='cuda'),
                         model_path=model_path,
                         batch_size=batch_size,
                         **kwargs)


class SDXLEngineBuilderOp(Operation, ArgDiscarder):
    @classmethod
    def immediate_dependencies(cls):
        return None

    def __init__(self,
                 *args,
                 # TODO: Legacy value - Remove after refactor is done.
                 precision: str,  # config precision only sets the UNetXL precision
                 config_ver: str = "default",
                 model_path: str = "build/models/SDXL/",
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.config_ver = config_ver

        # onnx.load() has issue reading weights separately under the same path as model file if input path is PosixPath, use str instead
        clip_path = model_path + "onnx_models/clip1/model.onnx"
        self.clip_builder = SDXLCLIPBuilder(*args, model_path=clip_path, **kwargs)

        # onnx.load() has issue reading weights separately under the same path as model file if input path is PosixPath, use str instead
        clip_proj_path = model_path + "onnx_models/clip2/model.onnx"
        self.clip_proj_builder = SDXLCLIPWithProjBuilder(*args, model_path=clip_proj_path, **kwargs)

        # onnx.load() has issue reading weights separately under the same path as model file if input path is PosixPath, use str instead
        if precision == 'int8':
            unetxl_path = model_path + "ammo_models/unetxl.int8/unet.onnx"
        elif precision == 'fp16':
            unetxl_path = model_path + "onnx_models/unetxl/model.onnx"
        else:
            raise ValueError("Unsupported UNetXL precision")
        self.unetxl_builder = SDXLUNetXLBuilder(*args, precision=precision, model_path=unetxl_path, **kwargs)

        # onnx.load() has issue reading weights separately under the same path as model file if input path is PosixPath, use str instead
        vae_path = model_path + "onnx_models/vae/model.onnx"
        self.vae_builder = SDXLVAEBuilder(*args, model_path=vae_path, **kwargs)

    def run(self, scratch_space, dependency_outputs):
        # Use default engine_dir
        engine_dir = self.clip_builder.engine_dir(scratch_space)

        # Build each engine separately
        logging.info(f"Building CLIP1 from {self.clip_builder.model_path}")
        clip_network = self.clip_builder.create_network(use_native_instance_norm=False)
        clip_engine_name = self.clip_builder.engine_name(self.clip_builder.device_type,
                                                         self.clip_builder.batch_size,
                                                         self.clip_builder.precision,
                                                         tag=self.config_ver)
        self.clip_builder.build_engine(network=clip_network,
                                       builder_config=self.clip_builder.create_builder_config(),
                                       batch_size=None,  # determined by clip_builder.batch_size
                                       save_to=engine_dir / clip_engine_name)

        logging.info(f"Building CLIP2 from {self.clip_proj_builder.model_path}")
        clip_proj_network = self.clip_proj_builder.create_network(use_native_instance_norm=False)
        clip_proj_engine_name = self.clip_proj_builder.engine_name(self.clip_proj_builder.device_type,
                                                                   self.clip_proj_builder.batch_size,
                                                                   self.clip_proj_builder.precision,
                                                                   tag=self.config_ver)
        self.clip_proj_builder.build_engine(network=clip_proj_network,
                                            builder_config=self.clip_proj_builder.create_builder_config(),
                                            batch_size=None,  # determined by clip_proj_builder.batch_size
                                            save_to=engine_dir / clip_proj_engine_name)

        logging.info(f"Building UNetXL from {self.unetxl_builder.model_path}")
        unetxl_network = self.unetxl_builder.create_network(use_native_instance_norm=False)
        unetxl_engine_name = self.unetxl_builder.engine_name(self.unetxl_builder.device_type,
                                                             self.unetxl_builder.batch_size,
                                                             self.unetxl_builder.precision,
                                                             tag=self.config_ver)
        self.unetxl_builder.build_engine(network=unetxl_network,
                                         builder_config=self.unetxl_builder.create_builder_config(),
                                         batch_size=None,  # determined by unetxl_builder.batch_size
                                         save_to=engine_dir / unetxl_engine_name)

        logging.info(f"Building VAE from {self.vae_builder.model_path}")
        vae_network = self.vae_builder.create_network(use_native_instance_norm=True)
        vae_engine_name = self.vae_builder.engine_name(self.vae_builder.device_type,
                                                       self.vae_builder.batch_size,
                                                       self.vae_builder.precision,
                                                       tag=self.config_ver)
        self.vae_builder.build_engine(network=vae_network,
                                      builder_config=self.vae_builder.create_builder_config(),
                                      batch_size=None,  # determined by vae_builder.batch_size
                                      save_to=engine_dir / vae_engine_name)


class SDXL(LegacyBuilder):
    def __init__(self, args):
        super().__init__(SDXLEngineBuilderOp(**args))
