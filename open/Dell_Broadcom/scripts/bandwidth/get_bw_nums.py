# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import subprocess as sp

pcie_gen_bw_gbytepersec = {
    '1': 0.250,
    '2': 0.500,
    '3': 0.985,
    '4': 1.959,
    '5': 3.938,
    '6': 7.563,
    '7': 15.125
}


def gpu_pcie_bandwidth_calc():
    nvsmi_proc = sp.run(["nvidia-smi", "--query-gpu", "index,pci.bus_id,pcie.link.gen.gpucurrent,pcie.link.gen.gpumax,pcie.link.gen.hostmax,pcie.link.width.current,pcie.link.width.max",
                        "--format=csv,noheader"], capture_output=True)
    nvsmi_out = nvsmi_proc.stdout.decode()

    gpus = nvsmi_out.split('\n')

    current_pci_bw = 0
    max_pci_bw = 0
    for gpu in gpus:
        if gpu == "":
            continue

        gpu_info = [x.strip(' ') for x in gpu.split(',')]
        for data_point in gpu_info:
            data_point.strip(' ')

        gpu_idx, gpu_pci_id, gpu_curr_pci_ver, gpu_max_pci_ver, host_max_pci_ver, pci_current_width, pci_max_width = gpu_info

        pci_current_width = int(pci_current_width)
        pci_max_width = int(pci_max_width)

        current_gpu_pci_bw = pcie_gen_bw_gbytepersec[gpu_curr_pci_ver] * pci_current_width
        max_gpu_pci_bw = pcie_gen_bw_gbytepersec[gpu_max_pci_ver] * pci_max_width

        print(f"GPU #{gpu_idx} with PCI bus ID {gpu_pci_id} has current b/w of {current_gpu_pci_bw}GB/s, and supports upto {max_gpu_pci_bw}GB/s")
        current_pci_bw += current_gpu_pci_bw
        max_pci_bw += max_gpu_pci_bw

    print(f"Aggregated accelerator(s) to PCIe bus bandwidth is {current_pci_bw}GB/s, max supported by accelerator(s) is {max_pci_bw}GB/s")


if __name__ == "__main__":
    gpu_pcie_bandwidth_calc()
