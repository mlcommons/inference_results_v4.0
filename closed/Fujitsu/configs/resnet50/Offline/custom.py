# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40SX16(OfflineGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx16

    gpu_batch_size = 64
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 170000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_L40Sx8(OfflineGPUBaseConfig):
    system = KnownSystem.CDI_L40Sx8

    gpu_batch_size = 64
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 170000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class CDI_L40Sx8_MaxQ(CDI_L40Sx8):
    pass

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2560M7_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.GX2560M7_H100_SXM_80GBx4
    gpu_batch_size = 2048
    offline_expected_qps = 90000*4
    start_from_device = True
