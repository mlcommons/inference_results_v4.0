# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1((MultiStreamGPUBaseConfig):
    system = KnownSystem.L4x1
    multi_stream_expected_latency_ns = 40000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR8620_L4x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1
    multi_stream_expected_latency_ns = 40000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR8620_L4X1_MAXQ(MultiStreamGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1_MaxQ
    multi_stream_expected_latency_ns = 40000000

