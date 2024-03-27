# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIe_80GBx4
    use_deque_limit = True
    deque_timeout_usec = 2119
    gpu_batch_size = 276
    gpu_copy_streams = 5
    gpu_inference_streams = 1
    server_target_qps = 144600
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760_L40X2(ServerGPUBaseConfig):
    system = KnownSystem.r760_l40x2
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 66
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 88270
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_L40Sx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_L40Sx4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 72
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 179630
    use_cuda_thread_per_device = True
    use_graphs = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7615_L40Sx2(ServerGPUBaseConfig):
    system = KnownSystem.R7615_L40Sx2
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 68
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 90600
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    use_deque_limit = True
    deque_timeout_usec = 3548
    gpu_batch_size = 273
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 310274
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBX4
    gpu_batch_size = 261
    gpu_copy_streams = 3
    gpu_inference_streams = 2
    deque_timeout_usec = 3641
    server_target_qps = 303050


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    use_deque_limit = True
    deque_timeout_usec = 4182
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True
    gpu_batch_size = 391
    gpu_copy_streams = 5
    gpu_inference_streams = 1
    server_target_qps = 630000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    use_deque_limit = True
    deque_timeout_usec = 3000
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True
    gpu_batch_size = 256
    gpu_copy_streams = 4
    gpu_inference_streams = 7
    power_limit = 300
    server_target_qps = 50000 * 8

