# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 572434000
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR7620_L4x1_HighAccuracy(XR7620_L4x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR8620_L4x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 572434000
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR8620_L4x1_HighAccuracy(XR8620_L4x1):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR8620_L4X1_MAXQ(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1_MaxQ
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 572434000
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XR8620_L4X1_MAXQ_HighAccuracy(XR8620_L4X1_MAXQ):
    pass

