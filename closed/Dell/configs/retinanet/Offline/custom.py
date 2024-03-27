# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIe_80GBx4
    gpu_batch_size = 26
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    offline_expected_qps = 2990
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760_L40X2(OfflineGPUBaseConfig):
    system = KnownSystem.r760_l40x2
    gpu_batch_size = 2
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    offline_expected_qps = 1700
    run_infer_on_copy_streams = False
    workspace_size = 60000000000
    start_from_device = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_L40Sx4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_L40Sx4
    gpu_batch_size = 4
    gpu_copy_streams = 2
   gpu_inference_streams = 2
    offline_expected_qps = 4000
    run_infer_on_copy_streams = False
    workspace_size = 60000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7615_L40Sx2(OfflineGPUBaseConfig):
    system = KnownSystem.R7615_L40Sx2
    gpu_batch_size = 4
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 2000
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = 30
    gpu_copy_streams = 3
    gpu_inference_streams = 2
    offline_expected_qps = 6800
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBX4
    gpu_batch_size = 23
    gpu_copy_streams = 3
    gpu_inference_streams = 2
    offline_expected_qps = 7922
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    offline_expected_qps = 14100
    gpu_batch_size = 48
    gpu_copy_streams = 2
    gpu_inference_streams = 6
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    gpu_batch_size = 2
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 220
    run_infer_on_copy_streams = False
    workspace_size = 20000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR8620_L4x1(OfflineGPUBaseConfig):

    system = KnownSystem.XR8620_L4x1
    gpu_batch_size = 2
    gpu_copy_streams = 3
    gpu_inference_streams = 2
    offline_expected_qps = 255 #170
    run_infer_on_copy_streams = False
    workspace_size = 20000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR8620_L4X1_MAXQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1_MaxQ
    gpu_batch_size = 1
    gpu_copy_streams = 2
    gpu_inference_streams = 3
    offline_expected_qps = 221
    run_infer_on_copy_streams = False
    workspace_size = 20000000000
    power_limit = 60

