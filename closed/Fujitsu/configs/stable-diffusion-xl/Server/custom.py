# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX16(ServerGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx16

    gpu_batch_size = 1
    sdxl_batcher_time_limit = 0
    server_target_qps = 10.2
    use_graphs = True
    min_query_count = 16*800

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX8(ServerGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx8

    gpu_batch_size = 1
    sdxl_batcher_time_limit = 0
    #server_target_qps = 4.9
    server_target_qps = 4.2
    use_graphs = True
    min_query_count = 8*800

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class CDI_L40SX8_MaxQ(CDI_L40SX8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(H100_SXM_80GBx8):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    server_target_qps = 12.5 / 2
