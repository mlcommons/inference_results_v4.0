#! /usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
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

import os
import sys
import argparse
import json

import collections

trt_version = "TensorRT 9.3.0"
cuda_version = "CUDA 12.2"
cudnn_version = "cuDNN 8.9.6"
dali_version = "DALI 1.28.0"
triton_version = "Triton 23.01"
os_version = "Ubuntu 20.04.4"
hopper_driver_version = "Driver 535.129.03"
submitter = "NVIDIA"

h200_driver_version = "Driver 550.54"
gracehopper_driver_version = "Driver 535.65"
gracehopper_os_version = "Ubuntu 22.04.2"

soc_sw_version_dict = \
    {
        "orin-jetson-agx": {
            "trt": "TensorRT 9.0.1",
            "cuda": "CUDA 11.4",
            "cudnn": "cuDNN 8.5.0",
            "jetpack": "Jetpack 5.1.1",
            "os": "Jetson r35.3.1 L4T",
            "dali": "N/A"
        },
    }


def get_soc_sw_version(accelerator_name, software_name):
    if software_name not in ["trt", "cuda", "cudnn", "jetpack", "os", "dali"]:
        raise KeyError(f"No version info for {software_name}. Options: {(list(soc_sw_version_dict[list(soc_sw_version_dict)[0]].keys()))}")
    if "orin nx" in accelerator_name.lower():
        return soc_sw_version_dict["orin-nx"][software_name]
    elif "orin" in accelerator_name.lower():
        # For v2.0 submission, "orin" stands for "orin-jetson-agx"
        if "auto" not in accelerator_name.lower():
            return soc_sw_version_dict["orin-jetson-agx"][software_name]
        else:
            raise KeyError("Only Jetson is available in the Orin family now.")
    else:
        raise KeyError(f"No version info for {accelerator_name}.")


class Status:
    AVAILABLE = "available"
    PREVIEW = "preview"
    RDI = "rdi"


class Division:
    CLOSED = "closed"
    OPEN = "open"


class SystemType:
    EDGE = "edge"
    DATACENTER = "datacenter"
    BOTH = "datacenter,edge"

# List of Machines


# host_memory_configuration: Get from sudo dmidecode --type 17
Machine = collections.namedtuple("Machine", [
    "status",
    "host_processor_model_name",
    "host_processors_per_node",
    "host_processor_core_count",
    "host_memory_capacity",
    "host_storage_capacity",
    "host_storage_type",
    "accelerator_model_name",
    "accelerator_short_name",
    "mig_short_name",
    "accelerator_memory_capacity",
    "accelerator_memory_configuration",
    "hw_notes",
    "sw_notes",
    "system_id_prefix",
    "system_name_prefix",
    "host_memory_configuration",
    "host_networking",
    "host_network_card_count",
    "host_networking_topology",
    "accelerator_host_interconnect",
    "accelerator_interconnect",
    "cooling",
    "system_type_detail",
    "power_supply_details",
    "power_supply_quantity_and_rating_watts",
])

"""
Tutorial: how to get each field.
L40S = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="AMD EPYC 7742 64-Core Processor", # lscpu
    host_processors_per_node=2, # lscpu
    host_processor_core_count=64, # lscpu
    host_memory_capacity="1 TB", # free -h
    host_storage_capacity="3 TB SSD, 5 TB CIFS", # df -h /tmp
    host_storage_type="NVMe SSD, CIFS mounted disk storage",
    accelerator_model_name="NVIDIA L40S", # nvidia-smi
    accelerator_short_name="L40S",
    mig_short_name="",
    accelerator_memory_capacity="45 GB",
    accelerator_memory_configuration="GDDR6",
    hw_notes="",
    sw_notes="",
    system_id_prefix="",
    system_name_prefix="ASROCKRACK 4U8G-ROME2/4E", # csm find "partitions=mlperf-inference" -v
    host_memory_configuration="16x 64GB 36ASF8G72PZ-3G2E1", # sudo dmidecode --type 17
    host_networking="Gig Ethernet", # lspci | egrep -i --color 'network|ethernet'
    host_network_card_count="2x 10Gbe",
    host_networking_topology="Ethernet on switching network",
    accelerator_host_interconnect="PCIe Gen4 x16", # sudo lspci -vv | grep -E 'PCI bridge|LnkCap'
    accelerator_interconnect="N/A",
    cooling="Air-cooled",
    system_type_detail="N/A",
    power_supply_details="",
    power_supply_quantity_and_rating_watts="",
)
"""
# The H100-PCIe-80GBx8 machine
IPP1_2037 = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="AMD EPYC 7742 64-Core Processor",
    host_processors_per_node=2,
    host_processor_core_count=64,
    host_memory_capacity="2 TB",
    host_storage_capacity="4 TB SSD, 5 TB CIFS",
    host_storage_type="NVMe SSD, CIFS mounted disk storage",
    accelerator_model_name="NVIDIA H100-PCIe-80GB",
    accelerator_short_name="H100-PCIe-80GB",
    mig_short_name="",
    accelerator_memory_capacity="80 GB",
    accelerator_memory_configuration="HBM2e",
    hw_notes="",
    sw_notes="",
    system_id_prefix="",
    system_name_prefix="Gigabyte G482-Z54",
    host_memory_configuration="32x 64GB 36ASF8G72PZ-3G2E1",
    host_networking="Gig Ethernet",
    host_network_card_count="2x 100Gbe",
    host_networking_topology="Ethernet on switching network",
    accelerator_host_interconnect="PCIe Gen4 x16",
    accelerator_interconnect="N/A",
    cooling="Air-cooled",
    system_type_detail="N/A",
    power_supply_details="",
    power_supply_quantity_and_rating_watts="",
)
# The H100-PCIe-80GBx8 machine MaxQ
IPP1_1470 = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="AMD EPYC 7742 64-Core Processor",
    host_processors_per_node=2,
    host_processor_core_count=64,
    host_memory_capacity="2 TB",
    host_storage_capacity="4 TB SSD, 5 TB CIFS",
    host_storage_type="NVMe SSD, CIFS mounted disk storage",
    accelerator_model_name="NVIDIA H100-PCIe-80GB",
    accelerator_short_name="H100-PCIe-80GB",
    mig_short_name="",
    accelerator_memory_capacity="80 GB",
    accelerator_memory_configuration="HBM2e",
    hw_notes="",
    sw_notes="",
    system_id_prefix="",
    system_name_prefix="Gigabyte G482-Z54",
    host_memory_configuration="32x 64GB 36ASF8G72PZ-3G2E1",
    host_networking="Gig Ethernet",
    host_network_card_count="2x 100Gbe",
    host_networking_topology="Ethernet on switching network",
    accelerator_host_interconnect="PCIe Gen4 x16",
    accelerator_interconnect="N/A",
    cooling="Air-cooled",
    system_type_detail="N/A",
    power_supply_details="",
    power_supply_quantity_and_rating_watts="",
)
# H100-SXM-80GB
DGX_H100 = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="Intel(R) Xeon(R) Platinum 8480C",
    host_processors_per_node=2,
    host_processor_core_count=56,
    host_memory_capacity="2 TB",
    host_storage_capacity="2 TB SSD, 5 TB CIFS",
    host_storage_type="NVMe SSD, CIFS mounted disk storage",
    accelerator_model_name="NVIDIA H100-SXM-80GB",
    accelerator_short_name="H100-SXM-80GB",
    mig_short_name="",
    accelerator_memory_capacity="80 GB",
    accelerator_memory_configuration="HBM3",
    hw_notes="",
    sw_notes="",
    system_id_prefix="DGX-H100",
    system_name_prefix="NVIDIA DGX H100",
    host_memory_configuration="32x 64GB MTC40F2046S1RC48BA1",
    host_networking="Infiniband; Data bandwidth for GPU-PCIe: 504GB/s; PCIe-NIC: 500GB/s",
    host_network_card_count="10x 400Gbe Infiniband",
    host_networking_topology="Ethernet/Infiniband on switching network",
    accelerator_host_interconnect="N/A",
    accelerator_interconnect="18x 4th Gen NVLink, 900GB/s",
    cooling="Air-cooled",
    system_type_detail="N/A",
    power_supply_details="",
    power_supply_quantity_and_rating_watts="",
)
# G+H CG1 starship
GH200_GraceHopper_Superchip_Starship = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="NVIDIA Grace CPU",
    host_processors_per_node=1,
    host_processor_core_count=72,
    host_memory_capacity="512 GB",
    host_storage_capacity="2 TB SSD, 5 TB CIFS",
    host_storage_type="NVMe SSD, CIFS mounted disk storage",
    accelerator_model_name="NVIDIA GH200-GraceHopper-Superchip",
    accelerator_short_name="GH200-96GB_aarch64",
    mig_short_name="",
    accelerator_memory_capacity="96 GB",
    accelerator_memory_configuration="HBM3",
    hw_notes="NVIDIA MGX Reference Platform;",
    sw_notes="",
    system_id_prefix="GH200-GraceHopper-Superchip",
    system_name_prefix="NVIDIA GH200-GraceHopper-Superchip",
    host_memory_configuration="16x 16DP (32GB) LPDDR5x",
    host_networking="Ethernet; Data bandwidth for GPU-NIC is 252.06 GB/s",
    host_network_card_count="1x 10Gbe Intel Ethernet X550T",
    host_networking_topology="Ethernet/Infiniband on switching network",
    accelerator_host_interconnect="NVLink-C2C",
    accelerator_interconnect="1x 400Gbe Infiniband",
    cooling="Air-cooled",
    system_type_detail="N/A",
    power_supply_details="",
    power_supply_quantity_and_rating_watts="",
)
# H200
DGX_H200 = Machine(
    status=Status.PREVIEW,
    host_processor_model_name="Intel(R) Xeon(R) Platinum 8480C",
    host_processors_per_node=2,
    host_processor_core_count=56,
    host_memory_capacity="2 TB",
    host_storage_capacity="2 TB SSD, 5 TB CIFS",
    host_storage_type="NVMe SSD, CIFS mounted disk storage",
    accelerator_model_name="NVIDIA H200-SXM-140GB",
    accelerator_short_name="H200-SXM-140GB",
    mig_short_name="",
    accelerator_memory_capacity="140 GB",
    accelerator_memory_configuration="HBM3",
    hw_notes="",
    sw_notes="",
    system_id_prefix="DGX-H200",
    system_name_prefix="NVIDIA DGX H200",
    host_memory_configuration="32x 64GB MTC40F2046S1RC48BA1",
    host_networking="Infiniband;  Data bandwidth for GPU-PCIe: 504GB/s; PCIe-NIC: 500GB/s",
    host_network_card_count="10x 400Gbe Infiniband",
    host_networking_topology="Ethernet/Infiniband on switching network",
    accelerator_host_interconnect="N/A",
    accelerator_interconnect="18x 4th Gen NVLink, 900GB/s",
    cooling="Air-cooled",
    system_type_detail="N/A",
    power_supply_details="",
    power_supply_quantity_and_rating_watts="",
)
# L40S
L40S = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="AMD EPYC 7742 64-Core Processor",  # lscpu
    host_processors_per_node=2,
    host_processor_core_count=64,
    host_memory_capacity="1 TB",
    host_storage_capacity="3 TB SSD, 5 TB CIFS",
    host_storage_type="NVMe SSD, CIFS mounted disk storage",
    accelerator_model_name="NVIDIA L40S",
    accelerator_short_name="L40S",
    mig_short_name="",
    accelerator_memory_capacity="45 GB",
    accelerator_memory_configuration="GDDR6",
    hw_notes="",
    sw_notes="",
    system_id_prefix="",
    system_name_prefix="ASROCKRACK 4U8G-ROME2/4E",  # csm find "partitions=mlperf-inference" -v
    host_memory_configuration="16x 64GB 36ASF8G72PZ-3G2E1",  # sudo dmidecode --type 17
    host_networking="Gig Ethernet; Data bandwidth for GPU-PCIe: 252 GB/s; PCIe-NIC: 7.877GB/s",  # lspci | egrep -i --color 'network|ethernet'
    host_network_card_count="2x 10Gbe",
    host_networking_topology="Ethernet on switching network",
    accelerator_host_interconnect="PCIe Gen4 x16",  # sudo lspci -vv | grep -E 'PCI bridge|LnkCap'
    accelerator_interconnect="N/A",
    cooling="Air-cooled",
    system_type_detail="N/A",
    power_supply_details="",
    power_supply_quantity_and_rating_watts="",
)
# Orin-Jetson submission machine for MaxQ
ORIN_AGX_MAXQ = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="12-core ARM Cortex-A78AE CPU",
    host_processors_per_node=1,
    host_processor_core_count=12,
    host_memory_capacity="64 GB",
    host_storage_capacity="64 GB eMMC, 5TB CIFS",
    host_storage_type="eMMC 5.1, CIFS mounted disk storage",
    accelerator_model_name="NVIDIA Jetson AGX Orin 64G",
    accelerator_short_name="Orin",
    mig_short_name="",
    accelerator_memory_capacity="Shared with host",
    accelerator_memory_configuration="LPDDR5",
    hw_notes="GPU and both DLAs are used in resnet50 and Retinanet, in Offline scenario. WIFI module is physically removed.",
    sw_notes="",
    system_id_prefix="",
    system_name_prefix="NVIDIA Jetson AGX Orin Developer Kit 64G",
    host_memory_configuration="64GB 256-bit LPDDR5",
    host_networking="USB Ethernet (RNDIS)",
    host_network_card_count="1 Integrated",
    host_networking_topology="USB forwarded",
    accelerator_host_interconnect="N/A",
    accelerator_interconnect="N/A",
    cooling="Air-cooled",
    system_type_detail="N/A",
    power_supply_details="Anker USB-C 715 (Nano II 65W)",
    power_supply_quantity_and_rating_watts="65W",
)
# Orin-Jetson submission machine for MaxP
ORIN_AGX_MAXP = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="12-core ARM Cortex-A78AE CPU",
    host_processors_per_node=1,
    host_processor_core_count=12,
    host_memory_capacity="64 GB",
    host_storage_capacity="64 GB eMMC, 5TB CIFS",
    host_storage_type="eMMC 5.1, CIFS mounted disk storage",
    accelerator_model_name="NVIDIA Jetson AGX Orin 64G",
    accelerator_short_name="Orin",
    mig_short_name="",
    accelerator_memory_capacity="Shared with host",
    accelerator_memory_configuration="LPDDR5",
    hw_notes="GPU and both DLAs are used in resnet50 and Retinanet, in Offline scenario",
    sw_notes="",
    system_id_prefix="",
    system_name_prefix="NVIDIA Jetson AGX Orin Developer Kit 64G",
    host_memory_configuration="64GB 256-bit LPDDR5",
    host_networking="Gig Ethernet",
    host_network_card_count="1 Integrated",
    host_networking_topology="USB forwarded",
    accelerator_host_interconnect="N/A",
    accelerator_interconnect="N/A",
    cooling="Air-cooled",
    system_type_detail="N/A",
    power_supply_details="Dell USB-C 130.0W Adapter (HA130PM170)",
    power_supply_quantity_and_rating_watts="130W",
)


class System():
    def __init__(self, machine, division, system_type, gpu_count=1, mig_count=0, is_triton=False, is_soc=False, is_maxq=False, additional_config=""):
        self.attr = {
            "system_id": self._get_system_id(machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config),
            "submitter": submitter,
            "division": division,
            "system_type": system_type,
            "system_type_detail": machine.system_type_detail,
            "status": machine.status if division == Division.CLOSED else Status.RDI,
            "system_name": self._get_system_name(machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config),
            "number_of_nodes": 1,
            "host_processor_model_name": machine.host_processor_model_name,
            "host_processors_per_node": machine.host_processors_per_node,
            "host_processor_core_count": machine.host_processor_core_count,
            "host_processor_frequency": "",
            "host_processor_caches": "",
            "host_processor_interconnect": "",
            "host_memory_configuration": machine.host_memory_configuration,
            "host_memory_capacity": machine.host_memory_capacity,
            "host_storage_capacity": machine.host_storage_capacity,
            "host_storage_type": machine.host_storage_type,
            "host_networking": machine.host_networking,
            "host_network_card_count": machine.host_network_card_count,
            "host_networking_topology": machine.host_networking_topology,
            "accelerators_per_node": gpu_count,
            "accelerator_model_name": self._get_accelerator_model_name(machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config),
            "accelerator_frequency": "",
            "accelerator_host_interconnect": machine.accelerator_host_interconnect,
            "accelerator_interconnect": machine.accelerator_interconnect,
            "accelerator_interconnect_topology": "",
            "accelerator_memory_capacity": machine.accelerator_memory_capacity,
            "accelerator_memory_configuration": machine.accelerator_memory_configuration,
            "accelerator_on-chip_memories": "",
            "cooling": machine.cooling,
            "hw_notes": machine.hw_notes,
            "sw_notes": machine.sw_notes,
            "framework": self._get_framework(machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config),
            "operating_system": self._get_operating_system(machine, is_soc),
            "other_software_stack": self._get_software_stack(machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config),
            "power_management": "",
            "filesystem": "",
            "boot_firmware_version": "",
            "management_firmware_version": "",
            "other_hardware": "",
            "number_of_type_nics_installed": "",
            "nics_enabled_firmware": "",
            "nics_enabled_os": "",
            "nics_enabled_connected": "",
            "network_speed_mbit": "",
            "power_supply_quantity_and_rating_watts": machine.power_supply_quantity_and_rating_watts,
            "power_supply_details": machine.power_supply_details,
            "disk_drives": "",
            "disk_controllers": "",
        }

    def _get_system_id(self, machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config):
        return "".join([
            (machine.system_id_prefix + "_") if machine.system_id_prefix != "" else "",
            machine.accelerator_short_name,
            ("x" + str(gpu_count)) if not is_soc and mig_count == 0 else "",
            "-MIG_{:}x{:}".format(mig_count * gpu_count, machine.mig_short_name) if mig_count > 0 else "",
            "_TRT" if division == Division.CLOSED else "",
            "_Triton" if is_triton else "",
            "_MaxQ" if is_maxq else "",
            "_{:}".format(additional_config) if additional_config != "" else "",
        ])

    def _get_system_name(self, machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config):
        system_details = []
        if not is_soc:
            system_details.append("{:d}x {:}{:}".format(
                gpu_count,
                machine.accelerator_short_name,
                "-MIG-{:}x{:}".format(mig_count, machine.mig_short_name) if mig_count > 0 else ""
            ))
        if is_maxq:
            system_details.append("MaxQ")
        if division == Division.CLOSED:
            system_details.append("TensorRT")
        if is_triton:
            system_details.append("Triton")
        if additional_config != "":
            system_details.append(additional_config)
        return "{:} ({:})".format(
            machine.system_name_prefix,
            ", ".join(system_details)
        )

    def _get_accelerator_model_name(self, machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config):
        return "{:}{:}".format(
            machine.accelerator_model_name,
            " ({:d}x{:} MIG)".format(mig_count, machine.mig_short_name) if mig_count > 0 else "",
        )

    def _get_operating_system(self, machine, is_soc):
        os = "Unknown"
        if is_soc:
            os = get_soc_sw_version(machine.accelerator_model_name, "os")
        else:
            if machine.accelerator_short_name.startswith('GH'):
                os = gracehopper_os_version
            else:
                os = os_version
        return os

    def _get_framework(self, machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config):
        frameworks = []
        if is_soc:
            frameworks.append(get_soc_sw_version(machine.accelerator_model_name, "jetpack"))
        if division == Division.CLOSED:
            # Distinguish different TRT version based on the arch/model
            if is_soc:
                version = get_soc_sw_version(machine.accelerator_model_name, "trt")
            else:
                version = trt_version
            frameworks.append(version)
        frameworks.append(cuda_version if not is_soc else get_soc_sw_version(machine.accelerator_model_name, "cuda"))
        return ", ".join(frameworks)

    def _get_software_stack(self, machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config):
        frameworks = []
        if is_soc:
            frameworks.append(get_soc_sw_version(machine.accelerator_model_name, "jetpack"))
        if division == Division.CLOSED:
            # Distinguish different TRT version based on the arch/model
            if is_soc:
                version = get_soc_sw_version(machine.accelerator_model_name, "trt")
            else:
                version = trt_version
            frameworks.append(version)
        frameworks.append(cuda_version if not is_soc else get_soc_sw_version(machine.accelerator_model_name, "cuda"))
        if division == Division.CLOSED:
            frameworks.append(cudnn_version if not is_soc else get_soc_sw_version(machine.accelerator_model_name, "cudnn"))
        if not is_soc:
            # For v4.0, the h100, G+H and H200 are on different driver version.
            if machine.accelerator_short_name[0] in ['H', 'L']:
                if machine.accelerator_short_name.startswith('H200'):
                    frameworks.append(h200_driver_version)
                else:
                    frameworks.append(hopper_driver_version)
            elif machine.accelerator_short_name.startswith('GH'):
                frameworks.append(gracehopper_driver_version)
            else:
                raise NotImplementedError(f"{machine.accelerator_short_name} not an available submission systems!")
        if division == Division.CLOSED:
            frameworks.append(dali_version if not is_soc else "")
        if is_triton:
            frameworks.append(triton_version)
        return ", ".join(frameworks)

    def __getitem__(self, key):
        return self.attr[key]


submission_systems = [
    # Datacenter submissions
    #                                                        #gpu   Triton, SOC, MaxQ
    System(DGX_H100, Division.CLOSED, SystemType.DATACENTER, 8, 0, False, False),  # H100-SXM-80GBx8
    System(DGX_H100, Division.CLOSED, SystemType.DATACENTER, 1, 0, False, False),  # H100-SXM-80GBx1
    System(DGX_H100, Division.CLOSED, SystemType.DATACENTER, 2, 0, False, False),  # H100-SXM-80GBx2
    System(DGX_H100, Division.CLOSED, SystemType.DATACENTER, 8, 0, False, False, True),  # H100-SXM-80GBx8-MaxQ
    System(DGX_H200, Division.CLOSED, SystemType.DATACENTER, 1, 0, False, False),  # H200-SXM-80GBx1
    System(DGX_H200, Division.CLOSED, SystemType.DATACENTER, 8, 0, False, False),  # H200-SXM-80GBx8
    System(GH200_GraceHopper_Superchip_Starship, Division.CLOSED, SystemType.DATACENTER, 1, 0, False, False),  # Starship G+H CG1
    System(L40S, Division.CLOSED, SystemType.DATACENTER, 8, 0, False, False),  # L40Sx8

    # Edge submissions
    System(ORIN_AGX_MAXP, Division.CLOSED, SystemType.EDGE, 1, 0, False, True),  # Jetson AGX Orin MaxP

    # Both datacenter and edge
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv", "-o",
        help="Specifies the output tab-separated file for system descriptions.",
        default="systems/system_descriptions.tsv"
    )
    parser.add_argument(
        "--dry_run",
        help="Don't actually copy files, just log the actions taken.",
        action="store_true"
    )
    parser.add_argument(
        "--manual_system_json",
        help="Path to the system json that is manually to the system description table.",
        nargs='+',
        default=[]
    )
    return parser.parse_args()


def main():
    args = get_args()

    tsv_file = args.tsv

    summary = []
    for system in submission_systems:
        json_file = os.path.join("..", "..", system["division"], system["submitter"], "systems", "{:}.json".format(system["system_id"]))
        print("Generating {:}".format(json_file))
        summary.append("\t".join([str(i) for i in [
            system["system_name"],
            system["system_id"],
            system["submitter"],
            system["division"],
            system["system_type"],
            system["system_type_detail"],
            system["status"],
            system["number_of_nodes"],
            system["host_processor_model_name"],
            system["host_processors_per_node"],
            system["host_processor_core_count"],
            system["host_processor_frequency"],
            system["host_processor_caches"],
            system["host_processor_interconnect"],
            system["host_memory_configuration"],
            system["host_memory_capacity"],
            system["host_storage_capacity"],
            system["host_storage_type"],
            system["host_networking"],
            system["host_networking_topology"],
            system["accelerators_per_node"],
            system["accelerator_model_name"],
            system["accelerator_frequency"],
            system["accelerator_host_interconnect"],
            system["accelerator_interconnect"],
            system["accelerator_interconnect_topology"],
            system["accelerator_memory_capacity"],
            system["accelerator_memory_configuration"],
            system["accelerator_on-chip_memories"],
            system["cooling"],
            system["hw_notes"],
            system["framework"],
            system["operating_system"],
            system["other_software_stack"],
            system["sw_notes"],
            system["power_management"],
            system["filesystem"],
            system["boot_firmware_version"],
            system["management_firmware_version"],
            system["other_hardware"],
            system["number_of_type_nics_installed"],
            system["nics_enabled_firmware"],
            system["nics_enabled_os"],
            system["nics_enabled_connected"],
            system["network_speed_mbit"],
            system["power_supply_quantity_and_rating_watts"],
            system["power_supply_details"],
            system["disk_drives"],
            system["disk_controllers"],
        ]]))
        del system.attr["system_id"]
        if not args.dry_run:
            with open(json_file, "w") as f:
                json.dump(system.attr, f, indent=4, sort_keys=True)
        else:
            print(json.dumps(system.attr, indent=4, sort_keys=True))

    # Add the systems to the summary, reading from the json file that's manually written.
    # Note: this is added since Triton system cannot be generated using this script.
    for fpath in args.manual_system_json:
        with open(fpath, "r") as fh:
            print(f"Adding {fpath} manually to the system description table.")
            system = json.load(fh)
            # Read the system_id directly from the file name.
            system_id = fpath.split("/")[-1].split(".")[0]
            summary.append("\t".join([str(i) for i in [
                system["system_name"],
                system_id,
                system["submitter"],
                system["division"],
                system["system_type"],
                system["system_type_detail"],
                system["status"],
                system["number_of_nodes"],
                system["host_processor_model_name"],
                system["host_processors_per_node"],
                system["host_processor_core_count"],
                system["host_processor_frequency"],
                system["host_processor_caches"],
                system["host_processor_interconnect"],
                system["host_memory_configuration"],
                system["host_memory_capacity"],
                system["host_storage_capacity"],
                system["host_storage_type"],
                system["host_networking"],
                system["host_networking_topology"],
                system["accelerators_per_node"],
                system["accelerator_model_name"],
                system["accelerator_frequency"],
                system["accelerator_host_interconnect"],
                system["accelerator_interconnect"],
                system["accelerator_interconnect_topology"],
                system["accelerator_memory_capacity"],
                system["accelerator_memory_configuration"],
                system["accelerator_on-chip_memories"],
                system["cooling"],
                system["hw_notes"],
                system["framework"],
                system["operating_system"],
                system["other_software_stack"],
                system["sw_notes"],
                system["power_management"],
                system["filesystem"],
                system["boot_firmware_version"],
                system["management_firmware_version"],
                system["other_hardware"],
                system["number_of_type_nics_installed"],
                system["nics_enabled_firmware"],
                system["nics_enabled_os"],
                system["nics_enabled_connected"],
                system["network_speed_mbit"],
                system["power_supply_quantity_and_rating_watts"],
                system["power_supply_details"],
                system["disk_drives"],
                system["disk_controllers"],
            ]]))

    print("Generating system description summary to {:}".format(tsv_file))
    if not args.dry_run:
        with open(tsv_file, "w") as f:
            for item in summary:
                print(item, file=f)
    else:
        print("\n".join(summary))

    print("Done!")


if __name__ == '__main__':
    main()
