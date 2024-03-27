# HPE ProLiant DL380a Gen11 System Architecture

![Architecture Diagram](hpe-proliant-dl380a-h100_diagram.png)

HPE's [ProLiant DL380a Gen11](https://buy.hpe.com/us/en/compute/rack-servers/proliant-dl300-servers/proliant-dl380-server/hpe-proliant-dl380a-gen11/p/1014696168) system supports the [GPUDirect](https://developer.nvidia.com/gpudirect) RDMA™ technology, which allows for direct I/O from PCIe devices (e.g. a Host Channel Adapter) to GPU device memory.  Each H100 GPU in the system can be connected to Infiniband 400 NDR NICs through PCIe Gen5x16 connections.

Resnet50-Offline running on a single ProLiant DL380a Gen11, with INT8 input, requires to support 13GB/s for example. NVIDIA has
measured approximately 49 GB/s per GPU in our internal measurement of P2P transfers between the NIC and the GPUs.
