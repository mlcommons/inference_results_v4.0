# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_TRT(ServerGPUBaseConfig):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8_TRT

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):

'''
# Optional fields:
    active_sms: int = 0
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    check_contiguity: bool = False
    coalesced_tensor: bool = False
    complete_threads: int = 0
    embedding_weights_on_gpu_part: float = 0.0
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    gpu_num_bundles: int = 0
    instance_group_count: int = 0
    max_pairs_per_staging_thread: int = 0
    mega_table_npy_file: str = ''
    mega_table_scales_npy_file: str = ''
    model_path: str = ''
    num_staging_batches: int = 0
    num_staging_threads: int = 0
    numa_config: str = ''
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    qsl_numa_override: str = ''
    reduced_precision_io: int = 0
    request_timeout_usec: int = 0
    row_frequencies_npy_filepath: str = ''
    run_infer_on_copy_streams: bool = False
    sample_partition_path: str = ''
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: int = 0
    server_target_qps_adj_factor: float = 0.0
    use_batcher_thread_per_device: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_spin_wait: bool = False
    verbose_glog: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0
'''

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_TRT_HighAccuracy(HPE_CRAY_XD670_H100_SXM_80GBX8_TRT):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_TRT_Triton(HPE_CRAY_XD670_H100_SXM_80GBX8_TRT):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):

'''
# Optional fields:
    active_sms: int = 0
    batch_triton_requests: bool = False
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    check_contiguity: bool = False
    coalesced_tensor: bool = False
    complete_threads: int = 0
    embedding_weights_on_gpu_part: float = 0.0
    gather_kernel_buffer_threshold: int = 0
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    gpu_num_bundles: int = 0
    instance_group_count: int = 0
    max_pairs_per_staging_thread: int = 0
    max_queue_delay_usec: int = 0
    mega_table_npy_file: str = ''
    mega_table_scales_npy_file: str = ''
    model_path: str = ''
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    num_staging_batches: int = 0
    num_staging_threads: int = 0
    numa_config: str = ''
    output_pinned_memory: bool = False
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    qsl_numa_override: str = ''
    reduced_precision_io: int = 0
    request_timeout_usec: int = 0
    row_frequencies_npy_filepath: str = ''
    run_infer_on_copy_streams: bool = False
    sample_partition_path: str = ''
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: int = 0
    server_target_qps_adj_factor: float = 0.0
    use_batcher_thread_per_device: bool = False
    use_concurrent_harness: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_spin_wait: bool = False
    verbose_glog: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0
'''

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_TRT_HighAccuracy_Triton(HPE_CRAY_XD670_H100_SXM_80GBX8_TRT_HighAccuracy):
    use_triton = True


