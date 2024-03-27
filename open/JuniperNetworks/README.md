# MLPerf Inference v4.0 Juniper Networks Submission

This is a repository of Juniper Networks implementations for the [MLPerf](https://mlcommons.org/en/) Inference Benchmark.

---

## Objective

This benchmark aims to provide a framework to run MLPerf Inference on a multi-node GPU infrastructure.

We utilize an Optimized Ethernet fabric, leveraging RDMA over Converged Ethernet (ROCE) version 2 as the transport protocol for inter GPU communication in the data-center.

With ROCE v2, we ensure high-speed, low-latency communication between nodes, optimizing the inference workflow. Through rigorous testing and analysis, we aim to uncover insights into the system's operation and scalability potential in real-world applications.

## MLPerf Inference Models Supported

- LLAMA 2 70B

## Prerequisites

- Servers with A100 or H100 GPUs
- Optimized Ethernet Fabric for Accelerator Interconnect

## Software Dependencies

- Docker
- Slurm with Enroot + Pyxis
- TensorRT
- cuDNN
- Nvidia Container Toolkit
- TensorRT-LLM
- Pytorch
- CUDA

This README assumes a functioning SLURM setup. Setting it up is beyond the scope of this document

## Model

Follow MLCommons download procedure to get the HF Model weights
<https://github.com/mlcommons/inference/tree/master/language/llama2-70b#mlcommons-members-download>

## Dataset

Follow MLCommons download procedure to get the Orca dataset
<https://github.com/mlcommons/inference/tree/master/language/llama2-70b#get-dataset>

## Build

Building the components required to run this benchmark are split into the following steps:

- Docker
- Quantization
- Checkpoint Conversion
- TensorRT-LLM Build

### Docker Build

Refer to the Dockerfile provided under the open/docker directory to build the container need to run this benchmark

### Quantization

This benchmark uses the quantization scripts provided in the TensorRT-LLM repository(<https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization>) to quantize our models in to the following data types based on the execution platform

- INT8 (A100)
- FP8 (H100)

Sample:

```bash
python quantize.py --model_dir ./tmp/llama/70B \
                --dtype float16 \
                --qformat fp8 \
                --kv_cache_dtype fp8 \
                --output_dir ./tllm_checkpoint_8gpu_tp8_pp2 \
                --calib_size 512 \
                --tp_size 8
                --pp_size 2

```

### Convert Checkpoints to TRT-LLM Format

The MLPerf Inference model weights provided are based on the Hugging Face platform. To make the weights compatible to use with TensorRT-LLM, convert them to FT format using the scripts provided in the TensorRT-LLM repository (<https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llama/convert_checkpoint.py>)

Sample:

```bash
python convert_checkpoint.py --meta_ckpt_dir ./tmp/llama/70B/ \
                            --output_dir ./tllm_checkpoint_8gpu_tp8_pp2 \
                            --dtype float16 \
                            --tp_size 8 \
                            --pp_size 2
```

### TRT-LLM Build

TensorRT-LLM requires TensorRT engines to generate an inference. Build the engines using the ```trtllm-build``` tool configuring Tensor Parallelism,Pipeline Parallelism and other parameters based on your setup. A full list of configurable parameters can be found in the TensorRT-LLM repository.

Sample:

```bash
trtllm-build --checkpoint_dir ./tllm_checkpoint_8gpu_tp8 \
            --output_dir ./tmp/llama/70B/trt_engines/8gpu_tp8_pp2 \
            --gemm_plugin float16 
```

Customize the ```build.slurm``` and ```quantize.slurm``` files provided under open/code directory for your environment

## Run Benchmark

Running the benchmark requires customization of the ```run.slurm``` script provided under the open/code/ based on the system environment

### Performance Tuning

To tune the performance of the inference generation for different scenarios, there are various parameters that can be changed to suit the use-case.

- max_new_tokens
- num_beams (Needs a rebuild of the TRT-LLM Engine with Beams)
- early_stopping
- temperature
- top_k
- top_p
- length_penalty
- repetition_penalty
- presence_penalty

#### Case Studies

To achieve a higher throughput tokens/second, the max_new_tokens(currently set to 1024 in the benchmark implementation) constraint can be removed from the generation parameters(can be found in SUT.py). An increase of 3x was observed with this change.

Similarly, a new TRT-LLM engine can be build with beams support and the num_beams parameter can be increased to improve the accuracy, but a decrease in throughput tokens/second was observed with this change.

## Network Traffic Monitoring

Before running the benchmark on a multi-node setup, ensure all the inter-node communication via ROCE V2 is functional. It can be verified using ```mpi-run```(<https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php>), ```nccl-tests```(<https://github.com/NVIDIA/nccl-tests>) or ```perftest```(<https://github.com/linux-rdma/perftest>)

Various tools can also be used to monitor Ethernet traffic over ROCE V2. Some of them are listed below:

- ```rdma statistics```
- ```pcap```
