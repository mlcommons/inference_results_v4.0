# HPE Cray XD670 Gen11 System Architecture

![Architecture Diagram](hpe-cray-xd670-h100_diagram.png)

HPE's [Cray XD670 Gen11](https://www.hpe.com/us/en/compute/hpc/supercomputing/cray-exascale-supercomputer.html) system supports the [GPUDirect](https://developer.nvidia.com/gpudirect) RDMA™ technology, which allows for direct I/O from PCIe devices (e.g. a Host Channel Adapter) to GPU device memory.  Each H100 GPU in the system can be connected to Infiniband 400 NDR NICs through PCIe Gen5x16 connections.

Resnet50-Offline running on a single Cray XD670 Gen11, with INT8 input, requires to support 13GB/s for example. NVIDIA has
measured approximately 49 GB/s per GPU in our internal measurement of P2P transfers between the NIC and the GPUs.
