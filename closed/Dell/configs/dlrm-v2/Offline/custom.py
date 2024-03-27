# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = 600_000
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 70000*4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4_HighAccuracy(XE8640_H100_SXM_80GBx4):
    offline_expected_qps = 44000*4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    offline_expected_qps = 604493
    numa_config = "0,1,2,3:0-55,112-167&4,5,6,7:56-111,168-223"
    gpu_batch_size = 241141
    embedding_weights_on_gpu_part: float = 1.0

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    offline_expected_qps = 404493
    numa_config = "0,1,2,3:0-55,112-167&4,5,6,7:56-111,168-223"
    gpu_batch_size = 600_000
    embedding_weights_on_gpu_part: float = 1.0


