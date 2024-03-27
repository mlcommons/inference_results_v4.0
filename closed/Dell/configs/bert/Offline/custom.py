# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIe_80GBx4
    gpu_batch_size: int = 1024
    gemm_plugin_fairshare_cache_size: int = 120
    offline_expected_qps: int = 15000
    use_small_tile_gemm_plugin: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIe_80GBx4_HighAccuracy(R750xa_A100_PCIe_80GBx4):
    precision = "fp16"
    offline_expected_qps = 7500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760_L40SX2(OfflineGPUBaseConfig):
    system = KnownSystem.R760_L40Sx2
    use_small_tile_gemm_plugin = True
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    gpu_batch_size = 32
    offline_expected_qps = 6800
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760_L40SX2_HighAccuracy(R760_L40SX2):
    precision = "fp16"
    offline_expected_qps = 6800
    use_fp8 = True
    use_graphs = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_L40Sx4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_L40Sx4
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 32
    offline_expected_qps = 13600
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760xa_L40Sx4_HighAccuracy(R760xa_L40Sx4):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 13200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7615_L40Sx2(OfflineGPUBaseConfig):
    system = KnownSystem.R7615_L40Sx2
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 32
    offline_expected_qps = 6800
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R7615_L40Sx2_HighAccuracy(R7615_L40Sx2):
    precision = "fp16"
    gpu_batch_size = 16
    use_fp8 = True
    use_graphs = False
    offline_expected_qps = 3500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 1346
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 37600
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4_HighAccuracy(XE8640_H100_SXM_80GBx4):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 992
    offline_expected_qps = 32800

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBX4
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 1313
    gpu_copy_streams = 2
    gpu_inference_streams = 3
    offline_expected_qps = 39394


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4_HighAccuracy(XE9640_H100_SXM_80GBX4):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 1369
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 34333
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    offline_expected_qps = 74500
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 1280
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    precision = "fp16"
    offline_expected_qps = 64600
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 1024

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR7620_L4x1_HighAccuracy(XR7620_L4x1):
    precision = "fp16"
    use_fp8 = True
    gpu_batch_size = 16
    offline_expected_qps = 640
    gpu_inference_streams = 1
    energy_aware_kernels = False
    gpu_copy_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR8620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 13
    offline_expected_qps = 1060
    workspace_size = 7516192768

