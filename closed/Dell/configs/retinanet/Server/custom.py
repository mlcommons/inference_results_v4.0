# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIe_80GBx4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 36911
    gpu_batch_size = 24
    gpu_inference_streams = 1
    server_target_qps = 2830
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760_L40X2(ServerGPUBaseConfig):
    system = KnownSystem.r760_l40x2
    gpu_batch_size = 4
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    server_target_qps = 1585
    run_infer_on_copy_streams = False
    workspace_size = 60000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_L40Sx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_L40Sx4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 36911
    gpu_batch_size = 8
    gpu_inference_streams = 1
    server_target_qps = 3060
    workspace_size = 70000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7615_L40Sx2(ServerGPUBaseConfig):
    system = KnownSystem.R7615_L40Sx2
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 36911
    gpu_batch_size = 8
    gpu_inference_streams = 1
    server_target_qps = 1510
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    start_from_device = True
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 13
    gpu_inference_streams = 2
    server_target_qps = 6763
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBX4
    start_from_device = True
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 32008
    gpu_batch_size = 14
    gpu_inference_streams = 3
    server_target_qps = 6737
    workspace_size = 60000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    start_from_device = True
    use_deque_limit = True
    server_target_qps = 13620
    workspace_size = 60000000000
    numa_config = "0,1,2,3:0-55,112-167&4,5,6,7:56-111,168-223"
    gpu_copy_streams = 3
    deque_timeout_usec = 31592
    gpu_batch_size = 13
    gpu_inference_streams = 8

