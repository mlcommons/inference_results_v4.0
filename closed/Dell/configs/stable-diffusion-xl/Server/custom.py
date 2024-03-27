# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_L40Sx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_L40Sx4
    gpu_batch_size = 2
    server_target_qps = 2.64
    sdxl_batcher_time_limit = 0
    use_graphs = True
    min_query_count = 6*800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7615_L40Sx2(ServerGPUBaseConfig):
    system = KnownSystem.R7615_L40Sx2
    gpu_batch_size = 1
    server_target_qps = 1.244
    sdxl_batcher_time_limit = 0
    use_graphs = True
    min_query_count = 6*800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = 8
    server_target_qps = 1.55
    sdxl_batcher_time_limit = 3
    use_graphs = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBX4
    gpu_batch_size = 6
    server_target_qps = 6.49
    sdxl_batcher_time_limit = 4
    use_graphs = False  # disable to meet latency constraint for x1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_batch_size = 8
    server_target_qps = 13.26
    sdxl_batcher_time_limit = 8
    use_graphs = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_batch_size = 8
    sdxl_batcher_time_limit = 5
    use_graphs = True
    server_target_qps = 9
    power_limit = 350

