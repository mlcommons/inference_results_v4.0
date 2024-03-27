# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx4
    gpu_batch_size = 2048
    server_target_qps = 8550*4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4_TRT(H100_PCIe_80GBx1):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_PCIe_80GBx4_TRT
    gpu_batch_size = 2048 #2250
    gpu_copy_streams = 6 #7
    gpu_inference_streams = 2
    audio_batch_size = 512
    audio_buffer_num_lines = 8192
    server_target_qps = 17025*4 
    use_graphs = True
    #start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_TRT(H100_SXM_80GBx8):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8_TRT
    #start_from_device = True
    #run_infer_on_copy_streams = True
    gpu_batch_size = 1728
    server_target_qps = 180000
    audio_batch_size = 512
    audio_buffer_num_lines = 8192
    use_graphs = True
    gpu_inference_streams = 2
    gpu_copy_streams = 2
    num_warmups = 20480
    nobatch_sorting = True
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 4
    #workspace_size = 60000000000


"""     # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):

    # Optional fields:
    active_sms: int = 0
    audio_batch_size: int = 0
    audio_buffer_num_lines: int = 0
    audio_fp16_input: bool = False
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    dali_batches_issue_ahead: int = 0
    dali_pipeline_depth: int = 0
    disable_encoder_plugin: bool = False
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    instance_group_count: int = 0
    max_seq_length: int = 0
    model_path: str = ''
    nobatch_sorting: bool = False
    noenable_audio_processing: bool = False
    nopipelined_execution: bool = False
    nouse_copy_kernel: bool = False
    num_warmups: int = 0
    numa_config: str = ''
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
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_spin_wait: bool = False
    verbose_glog: int = 0
    workspace_size: int = 0
 """

