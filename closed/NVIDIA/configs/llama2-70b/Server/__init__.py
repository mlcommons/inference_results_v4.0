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

ParentConfig = import_module("configs.llama2-70b")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 2
    pipeline_parallelism = 1
    precision = "fp16"
    enable_sort = False
    kvcache_free_gpu_mem_frac = 0.90


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_144GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.GH200_144GB_ARMx1
    gpu_batch_size = 1024
    use_fp8 = True
    enable_sort = False
    tensor_parallelism = 1
    server_target_qps = 12.4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GH200_144GB_aarch64x1_HighAccuracy(GH200_144GB_aarch64x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_TP2x1(ServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx2
    gpu_batch_size = 1024
    use_fp8 = True
    tensor_parallelism = 2
    server_target_qps = 18.4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_TP2x2(H100_SXM_80GB_TP2x1):
    system = KnownSystem.H100_SXM_80GBx4
    server_target_qps = 18.4 * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_TP2x4(H100_SXM_80GB_TP2x2):
    system = KnownSystem.H100_SXM_80GBx8
    server_target_qps = 17.8 * 4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_SXM_80GB_TP2x4_MaxQ(H100_SXM_80GB_TP2x4):
    system = KnownSystem.H100_SXM_80GBx8
    server_target_qps = 13.5 * 4
    power_limit = 450


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GB_TP2x1_HighAccuracy(H100_SXM_80GB_TP2x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GB_TP2x4_HighAccuracy(H100_SXM_80GB_TP2x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class H100_SXM_80GB_TP2x4_HighAccuracy_MaxQ(H100_SXM_80GB_TP2x4_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_NVL_94GB_TP2x1(ServerGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx2
    gpu_batch_size = 640
    use_fp8 = True
    server_target_qps = 12.5
    enable_sort = False
    tensor_parallelism = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_NVL_94GB_TP2x2(H100_NVL_94GB_TP2x1):
    system = KnownSystem.H100_NVL_94GBx4
    server_target_qps = 12.5 * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_NVL_94GB_TP2x4(H100_NVL_94GB_TP2x2):
    system = KnownSystem.H100_NVL_94GBx8
    server_target_qps = 12.5 * 4
    gpu_rank_map = "0,3&1,2&4,5&6,7"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_NVL_94GB_TP2x4_MaxQ(H100_NVL_94GB_TP2x4):
    server_target_qps = 38
    power_limit = 350


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_NVL_94GB_TP2x1_HighAccuracy(H100_NVL_94GB_TP2x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_NVL_94GB_TP2x4_HighAccuracy(H100_NVL_94GB_TP2x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class H100_NVL_94GB_TP2x4_HighAccuracy_MaxQ(H100_NVL_94GB_TP2x4_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1
    gpu_batch_size = 896
    use_fp8 = True
    server_target_qps = 11.8
    enable_sort = False
    tensor_parallelism = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx1_HighAccuracy(H200_SXM_141GBx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8(H200_SXM_141GBx1):
    system = KnownSystem.H200_SXM_141GBx8
    server_target_qps = 91.5


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx8_HighAccuracy(H200_SXM_141GBx8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_Full_SXM_141GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H200_Full_SXM_141GBx1
    gpu_batch_size = 1024
    use_fp8 = True
    server_target_qps = 12.8
    enable_sort = False
    tensor_parallelism = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_Full_SXM_141GBx1_HighAccuracy(H200_Full_SXM_141GBx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_Full_SXM_141GBx8(H200_Full_SXM_141GBx1):
    system = KnownSystem.H200_Full_SXM_141GBx8
    server_target_qps = 102


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_Full_SXM_141GBx8_HighAccuracy(H200_Full_SXM_141GBx8):
    pass
