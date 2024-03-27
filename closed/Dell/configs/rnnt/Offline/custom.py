# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIe_80GBx4
    gpu_batch_size = 2048
    offline_expected_qps = 60000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760_L40X2(OfflineGPUBaseConfig):
    system = KnownSystem.r760_l40x2
    gpu_batch_size = 1296
    offline_expected_qps = 23000
    use_graphs = True
    start_from_device = True
    disable_encoder_plugin: bool = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_L40Sx4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_L40Sx4
    gpu_batch_size = 1024
    offline_expected_qps = 62550


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7615_L40Sx2(OfflineGPUBaseConfig):
    system = KnownSystem.R7615_L40Sx2
    gpu_batch_size = 1024
    offline_expected_qps = 31275


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    start_from_device = True
    gpu_batch_size = 2048
    offline_expected_qps = 92000
    audio_batch_size = 1024
    audio_buffer_num_lines = 8192
    use_graphs = True
    disable_encoder_plugin = False
    gpu_inference_streams = 2
    gpu_copy_streams = 6

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBX4
    gpu_batch_size = 2165
    audio_batch_size = 1024
    audio_buffer_num_lines = 8192
    use_graphs = True
    disable_encoder_plugin = False
    offline_expected_qps = 99539
    gpu_copy_streams= 3
    gpu_inference_streams= 2
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    offline_expected_qps = 192150
    gpu_batch_size = 2383
    start_from_device = False
    audio_batch_size = 1024
    audio_buffer_num_lines = 8192
    use_graphs = True
    disable_encoder_plugin = False
    workspace_size = 80000000000
    num_warmups = 512
    dali_batches_issue_ahead = 4
    dali_pipeline_depth = 4
    gpu_inference_streams = 2
    gpu_copy_streams = 5
    nobatch_sorting = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    gpu_batch_size = 512
    offline_expected_qps = 3900
    audio_batch_size = 64
    audio_buffer_num_lines = 1024


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR8620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1
    gpu_batch_size = 512
    offline_expected_qps = 3900
    audio_batch_size = 64
    audio_buffer_num_lines = 1024

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR8620_L4X1_MAXQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1_MaxQ
    gpu_batch_size = 512
    offline_expected_qps = 3900
    audio_batch_size = 64
    audio_buffer_num_lines = 1024

