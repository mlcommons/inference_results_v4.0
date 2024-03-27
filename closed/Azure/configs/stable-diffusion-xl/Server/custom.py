# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC_H100_V5(ServerGPUBaseConfig):
    system = KnownSystem.NC_H100_v5
    gpu_batch_size = 8
    server_target_qps = 2.3
    use_graphs = False  # disable to meet latency constraint for x1
    min_query_count = 6 * 800
    sdxl_batcher_time_limit = 0

