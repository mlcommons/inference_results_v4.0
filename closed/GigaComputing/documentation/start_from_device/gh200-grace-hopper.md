# DGX-GH200 Grace Hopper Superchip System Architecture

[NVIDIA GH200 Grace Hopper](https://resources.nvidia.com/en-us-dgx-gh200/technical-white-paper) Superchip system supports the [GPUDirect](https://developer.nvidia.com/gpudirect) RDMA™ technology, which allows for direct I/O from PCIe devices (e.g. a Host Channel Adapter) to GPU device memory. The H100 GPU in the system is connected to a Grace CPU using NVIDIA's NVLINK C2C interface with data rates upto 450GB/s per direction. The Grace CPU supports up to 4 PCIe Gen5 x16 interfaces and can therefore support up to 4x NVIDIA Infiniband 400Gb NDR NICs. Using GPUDirect RDMA™, the H100 GPU can directly access PCIe devices attached to the Grace's PCIe links bypassing the Grace memory.

NVIDIA has measured approximately 49 GB/s unidirectional bandwidth in our internal measurement of P2P transfers between the NIC and the GPU.

Resnet50-Offline running on GH200, with INT8 input, requires to support 13GB/s.

3DUnet-Offline running on GH200, with INT8 input, requires upto 230MB/s of ingress and egress bandwidth.
