from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40SX8(OfflineGPUBaseConfig):
    system = KnownSystem.l40sx8
    gpu_batch_size = 64
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 320000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8
    numa_config = "0,1,2,3:0-47,96-143&4,5,6,7:48-95,144-191"
