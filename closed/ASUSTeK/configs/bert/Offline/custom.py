# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000_E11P_H100X8(H100_PCIe_80GBx8):
    system = KnownSystem.ESC8000_E11P_H100x8

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC8000_E11P_H100X8_HighAccuracy(H100_PCIe_80GBx8_HighAccuracy):
    system = KnownSystem.ESC8000_E11P_H100x8


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000_E11P_H100X8_Triton(ESC8000_E11P_H100X8):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: int = 0
    input_dtype: str = ''
    input_format: str = ''
    precision: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    batch_triton_requests: bool = False
    bert_opt_seqlen: int = 0
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    coalesced_tensor: bool = False
    deque_timeout_usec: int = 0
    energy_aware_kernels: bool = False
    gather_kernel_buffer_threshold: int = 0
    gemm_plugin_fairshare_cache_size: int = 0
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    graph_specs: str = ''
    graphs_max_seqlen: int = 0
    instance_group_count: int = 0
    max_queue_delay_usec: int = 0
    model_path: str = ''
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    numa_config: str = ''
    offline_expected_qps: float = 0.0
    output_pinned_memory: bool = False
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    soft_drop: float = 0.0
    use_concurrent_harness: bool = False
    use_fp8: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_small_tile_gemm_plugin: bool = False
    use_spin_wait: bool = False
    verbose_glog: int = 0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC8000_E11P_H100X8_HighAccuracy_Triton(ESC8000_E11P_H100X8_HighAccuracy):
    use_triton = True



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000_E11_L40SX8(OfflineGPUBaseConfig):
    system = KnownSystem.ESC8000_E11_L40Sx8
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 32
    offline_expected_qps = 3400*8
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC8000_E11_L40SX8_HighAccuracy(ESC8000_E11_L40SX8):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 1750*8
    gpu_batch_size = 32 
    

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000_E11_L40SX8_Triton(ESC8000_E11_L40SX8):
    use_triton = True


ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC8000_E11_L40SX8_HighAccuracy_Triton(ESC8000_E11_L40SX8_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000_E11_L4X8(OfflineGPUBaseConfig):
    system = KnownSystem.ESC8000_E11_L4x8
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 8
    offline_expected_qps = 7700
    workspace_size = 7516192768
    

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC8000_E11_L4X8_HighAccuracy(ESC8000_E11_L4X8):
    precision = "fp16"
    use_fp8 = True
    gpu_batch_size = 16 
    offline_expected_qps = 5200
    gpu_inference_streams = 1
    gpu_copy_streams = 1


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC8000_E11_L4X8_Triton(ESC8000_E11_L4X8):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC8000_E11_L4X8_HighAccuracy_Triton(ESC8000_E11_L4X8_HighAccuracy):
    use_triton = True

