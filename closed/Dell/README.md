# MLPerf Inference v4.0 Implementations
This is a repository of Dell Technologies servers using optimized implementations for [MLPerf Inference Benchmark v4.0](https://www.mlperf.org/inference-overview/).

# Implementations
## Benchmarks
**Please refer to /closed/NVIDIA for detailed instructions for NVIDIA GPU & Triton submissions, including performace guides, and instructions on how to run with new systems.** 

**Please refer to /closed/Qualcomm for detailed instructions for Qualcomm Cloud AI 100 submissions.**

**Please refer to /closed/Intel for detailed instructions for Intel CPU submissions.**
  
The following benchmarks are part of our submission for MLPerf Inference v4.0:
- [3d-unet](code/3d-unet/tensorrt/README.md)
- [bert](code/bert/tensorrt/README.md)
- [dlrmv2](code/dlrm-v2/tensorrt/README.md)
- [gptj](code/gptj/tensorrt/README.md)
- [stable-diffusion-xl](code/stable-diffusion-xl/tensorrt/README.md)
- [llama2](code/llama2-70b/tensorrt/README.md)
- [rnnt](code/rnnt/tensorrt/README.md)
- [retinanet](code/retinanet/README.md)
- [resnet50](code/resnet50/tensorrt/README.md)

# Dell Technologies Submission Systems

The closed systems that Dell has submitted on are:
- Datacenter Systems
  - Dell PowerEdge R750xa
    - NVIDIA A100-PCIe-80GB
  - Dell PowerEdge R760
    - Intel Platinum 8592+
    - NVIDIA L40S
    - Qualcomm Cloud AI 100 Ultra
  - Dell PowerEdge R760xa
    - NVIDIA L40S
  - Dell PowerEdge R7615
    - NVIDIA L40S
  - Dell PowerEdge XE8640
    - NVIDIA H100-SXM-80GB
  - Dell PowerEdge XE9640
    - NVIDIA H100-SXM-80GB
  - Dell PowerEdge XE9680
    - NVIDIA H100-SXM-80GB
- Edge Systems
  - Dell PowerEdge XR7620
    - NVIDIA L4
  - Dell PowerEdge XR8620
    - NVIDIA L4



