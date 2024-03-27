# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000_E11P_H100X8(ServerGPUBaseConfig):
    system = KnownSystem.ESC8000_E11P_H100x8
    gpu_batch_size = 8
    server_target_qps = 8
    sdxl_batcher_time_limit = 5
    use_graphs = True  # disable to meet latency constraint for x1

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000_E11_L40SX8(L40Sx8):
    system = KnownSystem.ESC8000_E11_L40Sx8

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):


