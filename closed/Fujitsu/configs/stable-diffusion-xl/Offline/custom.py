# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX16(OfflineGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx16
    gpu_batch_size = 1
    use_graphs = True
    offline_expected_qps = 9.6

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX8(OfflineGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx8
    gpu_batch_size = 1
    use_graphs = True
    offline_expected_qps = 4.8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class CDI_L40SX8_MaxQ(CDI_L40SX8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    #offline_expected_qps = 12.9 / 2
    gpu_batch_size = 8
    offline_expected_qps = 1.6 * 4
    use_graphs = False    
