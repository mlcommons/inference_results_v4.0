# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = 51200 * 2
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps =258750
    server_num_issue_query_threads = 4
    numa_config = "0-1:0-47,96-143&2-3:48-95,144-191"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4_HighAccuracy(XE8640_H100_SXM_80GBx4):
    server_target_qps =170000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_batch_size = 51200
    server_target_qps = 509000
    server_num_issue_query_threads = 8
    numa_config = "0,1,2,3:0-55,112-167&4,5,6,7:56-111,168-223"
    embedding_weights_on_gpu_part: float = 1.0

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    gpu_batch_size = 51200
    server_target_qps = 331300
    server_num_issue_query_threads = 8
    numa_config = "0,1,2,3:0-55,112-167&4,5,6,7:56-111,168-223"
    embedding_weights_on_gpu_part: float = 1.0


