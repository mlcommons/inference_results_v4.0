# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIe_80GBx4
    gpu_batch_size = 2040
    server_target_qps = 49550
    gpu_copy_streams = 1
    gpu_inference_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760_L40X2(ServerGPUBaseConfig):
    system = KnownSystem.r760_l40x2
    gpu_batch_size = 1272
    offline_expected_qps = 25000
    use_graphs = True
    start_from_device = True
    disable_encoder_plugin: bool = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_L40Sx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_L40Sx4
    gpu_batch_size = 2048
    server_target_qps = 42250
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    audio_batch_size = 512
    use_graphs = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7615_L40Sx2(ServerGPUBaseConfig):
    system = KnownSystem.R7615_L40Sx2
    gpu_batch_size = 1024
    server_target_qps = 22350
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    audio_batch_size = 512
    use_graphs = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = 2189
    server_target_qps = 96000
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True  # MLPINF-1773
    gpu_inference_streams = 2
    gpu_copy_streams = 7

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBX4
    gpu_batch_size = 1986
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True  # MLPINF-1773
    server_target_qps = 92133
    gpu_copy_streams = 5
    gpu_inference_streams = 2
    dali_pipeline_depth = 4
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_batch_size = 1728
    server_target_qps = 187300
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    dali_pipeline_depth = 4
    audio_batch_size = 512
    use_graphs = True
    audio_buffer_num_lines = 8192

