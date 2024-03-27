# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC_H100_V5(ServerGPUBaseConfig):
    system = KnownSystem.NC_H100_v5
    gpu_batch_size = 128
    use_fp8 = True
    server_target_qps = 42

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC_H100_V5_HighAccuracy(NC_H100_V5):
    precision = "fp16"


