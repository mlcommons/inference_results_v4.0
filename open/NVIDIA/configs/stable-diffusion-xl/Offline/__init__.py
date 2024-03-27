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

import os
import sys
sys.path.insert(0, os.getcwd())
from importlib import import_module

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
GPUBaseConfig = import_module("configs.stable-diffusion-xl").GPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    precision = "int8"

    use_graphs = False
    gpu_inference_streams = 1
    gpu_copy_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_96GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1
    gpu_batch_size = 8
    offline_expected_qps = 1.8
    use_graphs = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    gpu_batch_size = 8
    offline_expected_qps = 1.64
    use_graphs = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = 13.2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_SXM_80GBx8_MaxQ(H100_SXM_80GBx8):
    offline_expected_qps = 9
    power_limit = 350


# PCIE for testing only
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GBx1
    gpu_batch_size = 8
    offline_expected_qps = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1
    gpu_batch_size = 8
    offline_expected_qps = 1.7
    use_graphs = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8(H200_SXM_141GBx1):
    system = KnownSystem.H200_SXM_141GBx8
    offline_expected_qps = 13.7


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40Sx1(OfflineGPUBaseConfig):
    system = KnownSystem.L40Sx1
    gpu_batch_size = 1
    offline_expected_qps = 0.6
    use_graphs = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40Sx8(L40Sx1):
    system = KnownSystem.L40Sx8
    offline_expected_qps = 4.8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin(OfflineGPUBaseConfig):
    system = KnownSystem.Orin
    precision = "fp16"
    workspace_size = 60000000000
    gpu_batch_size = 1
    offline_expected_qps = 0.07
    use_graphs = True


################## open division configs start from here ########################


class OfflineGPUOpenBaseConfig(OfflineGPUBaseConfig):
    use_graphs = True
    gpu_batch_size = 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, ModelOpt.DeepCache)
class H100_SXM_80GBx1(OfflineGPUOpenBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    offline_expected_qps = 1.64
    sdxl_shallow_unet_steps = 1
    open_model_path = "build/open-models/SDXL-DeepCache/"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, ModelOpt.DeepCache)
class H100_SXM_80GBx8(OfflineGPUOpenBaseConfig):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = 1.64 * 8
    sdxl_shallow_unet_steps = 1
    open_model_path = "build/open-models/SDXL-DeepCache/"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, ModelOpt.DeepCachePruned)
class H100_SXM_80GBx1(OfflineGPUOpenBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    offline_expected_qps = 4
    sdxl_shallow_unet_steps = 1
    open_model_path = "build/open-models/SDXL-DeepCachePruned/"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, ModelOpt.DeepCachePruned)
class H100_SXM_80GBx8(OfflineGPUOpenBaseConfig):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = 4 * 8
    sdxl_shallow_unet_steps = 1
    open_model_path = "build/open-models/SDXL-DeepCachePruned/"
