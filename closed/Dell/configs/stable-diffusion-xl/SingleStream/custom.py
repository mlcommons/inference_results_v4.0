# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py


from . import *



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR8620_L4x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR8620_L4x1
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 274000000
    use_graphs = True
