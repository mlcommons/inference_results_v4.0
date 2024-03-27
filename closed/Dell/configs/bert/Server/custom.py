# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIe_80GBx4
    gpu_batch_size: int = 77
    active_sms: int = 60
    gpu_copy_streams: int = 4
    gpu_inference_streams: int = 2
    graphs_max_seqlen: int = 200
    server_num_issue_query_threads: int = 1
    server_target_qps: int = 11860
    soft_drop: float = 1.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIe_80GBx4_HighAccuracy(R750xa_A100_PCIe_80GBx4):
   precision = "fp16"
    server_target_qps = 5800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760_L40SX2(ServerGPUBaseConfig):
    system = KnownSystem.R760_L40Sx2
    gpu_batch_size = 48
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 6700
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    workspace_size = 7000000000000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760_L40SX2_HighAccuracy(R760_L40SX2):
    precision = "fp16"
    server_target_qps = R760_L40SX2.server_target_qps / 2
    use_fp8 = True
    use_graphs = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_L40Sx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_L40Sx4
    gpu_batch_size: int = 16
    active_sms: int = 60
    gpu_copy_streams: int = 4
    gpu_inference_streams: int = 2
    graphs_max_seqlen: int = 200
    server_num_issue_query_threads: int = 1
    server_target_qps: int = 13700
    soft_drop: float = 1.0
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760xa_L40Sx4_HighAccuracy(R760xa_L40Sx4):
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 6920
    gpu_batch_size = 48
    use_graphs = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7615_L40Sx2(ServerGPUBaseConfig):
    system = KnownSystem.R7615_L40Sx2
    gpu_batch_size: int = 16
    active_sms: int = 60
    gpu_copy_streams: int = 4
    gpu_inference_streams: int = 2
    graphs_max_seqlen: int = 200
    server_num_issue_query_threads: int = 1
    server_target_qps: int = 6670
    soft_drop: float = 1.0
    use_small_tile_gemm_plugin = True
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R7615_L40Sx2_HighAccuracy(R7615_L40Sx2):
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 3325
    gpu_batch_size = 48
    use_graphs = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = 204
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    server_target_qps = 28559
    server_num_issue_query_threads = 1
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4_High_Accuracy(XE8640_H100_SXM_80GBx4):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 170
    server_target_qps = 25390


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBX4
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = 120
    gpu_copy_streams = 3
    gpu_inference_streams = 2
    server_target_qps = 28343
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4_HighAccuracy(XE9640_H100_SXM_80GBX4):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 145
    server_target_qps = 24850

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_inference_streams = 2
    gpu_copy_streams = 4
    server_target_qps = 57100
    use_small_tile_gemm_plugin = False
    use_graphs = False
    gpu_batch_size = 171
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 292
    server_target_qps = 51200
    use_small_tile_gemm_plugin = False
    gpu_copy_streams = 6
    gpu_inference_streams = 1
    server_num_issue_query_threads = 1
    workspace_size = 7516192768

