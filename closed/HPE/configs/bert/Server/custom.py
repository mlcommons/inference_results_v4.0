# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx4
    #gpu_batch_size = 64
    #gpu_batch_size = 64*2
    gpu_batch_size = 64*1
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 2890.0*4
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4_HighAccuracy(HPE_ProLiant_DL380a_L40S_PCIe_48GBx4):
    precision = "fp16"
    server_target_qps = 1360*4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4_TRT(H100_PCIe_80GBx1):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_PCIe_80GBx4_TRT
    gpu_batch_size = 128
    precision = "fp16"
    server_target_qps = 3900*4
    use_fp8 = True
    soft_drop = 1.0
    server_num_issue_query_threads = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 2

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4_TRT_HighAccuracy(HPE_ProLiant_DL380a_H100_PCIe_80GBx4_TRT):
    gpu_batch_size = 128
    precision = "fp16"
    server_target_qps = 3800*4
    use_fp8 = True
    soft_drop = 1.0

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_TRT(H100_SXM_80GBx8):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8_TRT
    start_from_device = True
    #run_infer_on_copy_streams = True 
    gpu_batch_size = 128
    server_target_qps = 7150 * 8 #7200 * 8 #7100 * 8  #H100_SXM_80GBx1.server_target_qps * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_TRT_HighAccuracy(HPE_CRAY_XD670_H100_SXM_80GBX8_TRT):
    server_target_qps = 6410*8  
    start_from_device = True
    #run_infer_on_copy_streams = True
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 256 #512
    gpu_inference_streams = 1

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_TRT_Triton(HPE_CRAY_XD670_H100_SXM_80GBX8_TRT):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_TRT_HighAccuracy_Triton(HPE_CRAY_XD670_H100_SXM_80GBX8_TRT_HighAccuracy):
    use_triton = True

"""     # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
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
    output_pinned_memory: bool = False
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: int = 0
    server_target_qps_adj_factor: float = 0.0
    soft_drop: float = 0.0
    use_concurrent_harness: bool = False
    use_fp8: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_small_tile_gemm_plugin: bool = False
    use_spin_wait: bool = False
    verbose_glog: int = 0
    workspace_size: int = 0
 """


