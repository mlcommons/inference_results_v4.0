# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ES200G2_L40SX2(SingleStreamGPUBaseConfig):
    system = KnownSystem.ES200G2_L40Sx2

    precision = "fp16"
    workspace_size = 60000000000
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 1370000000
    use_graphs = True
    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    #gpu_batch_size: int = 0
    #input_dtype: str = ''
    #input_format: str = ''
    #precision: str = ''
    #tensor_path: str = ''

    ## Optional fields:
    #active_sms: int = 0
    #buffer_manager_thread_count: int = 0
    #cache_file: str = ''
    #gpu_copy_streams: int = 0
    #gpu_inference_streams: int = 0
    #instance_group_count: int = 0
    #model_path: str = ''
    #numa_config: str = ''
    #performance_sample_count_override: int = 0
    #preferred_batch_size: str = ''
    #request_timeout_usec: int = 0
    #run_infer_on_copy_streams: bool = False
    #single_stream_expected_latency_ns: int = 0
    #single_stream_target_latency_percentile: float = 0.0
    #use_graphs: bool = False
    #use_jemalloc: bool = False
    #use_spin_wait: bool = False
    #verbose_glog: int = 0
    #workspace_size: int = 0


