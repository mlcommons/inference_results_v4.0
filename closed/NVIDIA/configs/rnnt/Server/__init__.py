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

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.rnnt import GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    use_graphs = True
    gpu_inference_streams = 1
    gpu_copy_streams = 1
    num_warmups = 20480
    nobatch_sorting = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_96GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1
    gpu_batch_size = 2048
    server_target_qps = 24000
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4x1(ServerGPUBaseConfig):
    system = KnownSystem.L4x1
    gpu_batch_size = 512
    server_target_qps = 3750
    audio_batch_size = 64
    audio_buffer_num_lines = 1024


class L40x1(ServerGPUBaseConfig):
    system = KnownSystem.L40x1
    gpu_batch_size = 2048
    server_target_qps = 8500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    gpu_batch_size = 2048
    server_target_qps = 22000
    audio_buffer_num_lines = 8192
    audio_batch_size = 768
    use_graphs = True  # MLPINF-1773


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GBx8
    gpu_batch_size = 2048
    server_target_qps = 185000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_SXM_80GBx8_MaxQ(H100_SXM_80GBx8):
    power_limit = 300
    server_target_qps = 15500 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GBx1
    gpu_batch_size = 2048
    server_target_qps = 15000
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True  # MLPINF-1773


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    system = KnownSystem.H100_PCIe_80GBx8
    gpu_batch_size = 2048
    server_target_qps = 12500 * 8
    audio_batch_size = 512


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    power_limit = 200
    server_target_qps = 88000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GB_ARMx1
    gpu_batch_size = 2048
    server_target_qps = 15000
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True  # MLPINF-1773


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x4(H100_PCIe_80GB_aarch64x1):
    system = KnownSystem.H100_PCIe_80GB_ARMx4
    gpu_batch_size = 2048
    server_target_qps = 60000
    audio_batch_size = 512


class A100_PCIe_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx1
    gpu_batch_size = 2048
    server_target_qps = 11100


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(A100_PCIe_80GBx1):
    system = KnownSystem.A100_PCIe_80GBx8
    server_target_qps = 90000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    server_target_qps = 75000
    power_limit = 200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_MIG_1x1g_10gb
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 2
    gpu_batch_size = 1024
    num_warmups = 64
    server_target_qps = 650
    max_seq_length = 64


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    server_target_qps = 600


class A100_PCIe_80GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx1
    gpu_batch_size = 2048
    server_target_qps = 11100


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(A100_PCIe_80GB_aarch64x1):
    system = KnownSystem.A100_PCIe_80GB_ARMx4
    server_target_qps = 43000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    server_target_qps = 37500
    power_limit = 200


class A100_PCIe_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_ARMx1
    gpu_batch_size = 2048
    server_target_qps = 11100


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4(A100_PCIe_aarch64x1):
    system = KnownSystem.A100_PCIe_40GB_ARMx4
    dali_pipeline_depth = 1
    server_target_qps = 42500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    server_target_qps = 38500
    power_limit = 200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 2
    gpu_batch_size = 1024
    num_warmups = 64
    server_target_qps = 1350
    start_from_device = True
    max_seq_length = 64


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 1320


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARM_MIG_1x1g_10gb
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 2
    gpu_batch_size = 1024
    num_warmups = 64
    server_target_qps = 1300
    max_seq_length = 64


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx1
    gpu_batch_size = 1792
    server_target_qps = 12750


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8(A100_SXM_80GB_aarch64x1):
    system = KnownSystem.A100_SXM_80GB_ARMx8
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    server_num_issue_query_threads = 0
    server_target_qps = 104000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GB_aarch64x8_MaxQ(A100_SXM_80GB_aarch64x8):
    server_target_qps = 75000
    power_limit = 250


class A100_SXM_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx1
    gpu_batch_size = 1792
    server_target_qps = 12750
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(A100_SXM_80GBx1):
    system = KnownSystem.A100_SXM_80GBx8
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    server_num_issue_query_threads = 0
    server_target_qps = 104000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    server_target_qps = 88000
    power_limit = 275


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(ServerGPUBaseConfig):
    system = KnownSystem.A2x2
    audio_buffer_num_lines = 512
    dali_pipeline_depth = 1
    gpu_copy_streams = 4
    num_warmups = 32
    gpu_batch_size = 256
    audio_batch_size = 32
    server_target_qps = 1305


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(ServerGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    audio_batch_size = 32
    audio_buffer_num_lines = 512
    audio_fp16_input = None
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    gpu_batch_size = 256
    num_warmups = 32
    nobatch_sorting = None
    server_target_qps = 1100
    workspace_size = 1610612736


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 950


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(ServerGPUBaseConfig):
    system = KnownSystem.A30x8
    gpu_batch_size = 1792
    server_target_qps = 37000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x8_MaxQ(A30x8):
    server_target_qps = 43500.0
    power_limit = 200
