# GPTJ readme

## Getting started

Please first download the data, model and preprocess the data folloing the steps below
```
BENCHMARKS=gptj make download_model
BENCHMARKS=gptj make download_data
BENCHMARKS=gptj make preprocess_data
```
Make sure after the 3 steps above, you have the model downloaded under `build/models/GPTJ-6B`, and preprocessed data under `build/preprocessed_data/cnn_dailymail_tokenized_gptj/`.

### Backup model download method

If the download_data is not working due to MLCommon cloud error, you can use CK as an alternative apporach. We recommend to run it on a machine with sudo access.

```
pip install cmind
cm pull repo mlcommons@ck
cm run script --tags=get,ml-model,gptj,_pytorch,_rclone -j
```

The model will be rcloned to a local directory which looks likes: `/home/<username>/CM/repos/local/cache/04dedc0feede4f18/checkpoint`. Please move the model to `build/models/GPTJ-6B`.

## Fp16 model generation for Orin GPTJ submission

NVIDIA's Orin GPTJ submission uses fp16 GPTJ model. The fp16 GPTJ model needs to be converted from HuggingFace checkpoint via TRTLLM. Please follow the steps below to generate the model.
```
# Launch MLPerf Inference container under closed/NVIDIA on Orin AGX
OUTSIDE_MLPINF_ENV=1 make prebuild DOCKER_ARGS="--security-opt systempaths=unconfined"

# Clone and compile TRTLLM on Orin AGX
make clone_trt_llm && make build_trt_llm

# Run the model convertion script
cd build/TRTLLM/examples/gptj/
python convert_checkpoint.py --model_dir <path_to_your_fp16_huggingface_checkpoint> \
                             --dtype float16 \
                             --output_dir /work/build/models/GPTJ-6B/orin-fp16-model/mlperf-gptj6b-trtllm-fp16
```

## Build and quantization preparation for datacenter GPTJ submission

The quantization needs to be performed in a separate build environment than the MLPerf container. Please follow the steps below:
```
# Make sure you are outside of the MLPerf container
cd <workdir>
git clone https://github.com/NVIDIA/TensorRT-LLM.git

# Using 2/6/2024 ToT
cd TensorRT-LLM && git checkout 0ab9d17a59c284d2de36889832fe9fc7c8697604
make -C docker build

# The default docker command will not mount extra directory. If necessary, copy the docker command and append
# -v <src_dir>:<dst:dir> to mount your own directory.
make -C docker run LOCAL_USER=1

# The following steps will be performed within TRTLLM container. Change -a=90 to your target architecture
python3 scripts/build_wheel.py -a=90 --clean --install --trt_root /usr/local/tensorrt/

# Quantize the benchmark
python examples/quantization/quantize.py --dtype=float16  --output_dir=<model_dir>/models/GPTJ-6B/fp8-quantized-ammo/GPTJ-FP8-quantized --model_dir=<model_dir>/models/checkpoint --qformat=fp8 --kv_cache_dtype=fp8

# Further tune the quantization in-place
python <mlperf_dir>/closed/NVIDIA/code/gptj/tensorrt/onnx_tune.py --fp8-scalers-path=<model_dir>/models/GPTJ-6B/fp8-quantized-ammo/GPTJ-FP8-quantized/rank0.safetensors --scaler 1.005 --index 15
```

## Build and run the benchmarks

Please follow the steps below:
```
# Enter the MLPerf container (Orin AGX uses different command to launch the container, see above)
make prebuild

# The current build only supports SM90. If you want to try SM80/89 support, please go to Makefile.build and remove the "-a=90" flag from "build_trt_llm" target.
BUILD_TRTLLM=1 make build

# Before generating the engines, please point fp8_quant_model_path in code/gptj/tensorrt/builder.py to your quantized model path.
make generate_engines RUN_ARGS="--benchmarks=gptj --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly --fast"
make run_harness RUN_ARGS="--benchmarks=gptj --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly --fast"
```

You should expect to get the following results:
```
  gptj-99.9:
     accuracy: [PASSED] ROUGE1: 43.068 (Threshold=42.944) | [PASSED] ROUGE2: 20.129 (Threshold=20.103) | [PASSED] ROUGEL: 30.022 (Threshold=29.958) | [PASSED] GEN_LEN: 4095514.000 (Threshold=3615190.200)
```
