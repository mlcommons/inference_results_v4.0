# Llama2 readme

## Getting started

Please first download the data, model and preprocess the data folloing the steps below
```
# Visit https://docs.google.com/forms/d/e/1FAIpQLSc_8VIvRmXM3I8KQaYnKf7gy27Z63BBoI_I1u02f4lw6rBp3g/viewform to sign the agreement,
# which gets you access to the dataset
# Please please the llama2 models (Llama-2-70b-chat-hf) under build/models/Llama2 and datasets (open_orca) under build/preprocessed_data
BENCHMARKS=llama2 make preprocess_data
```
Make sure after the 3 steps above, you have the model downloaded under `build/models/Llama2`, and preprocessed data under `build/preprocessed_data/open_orca/`.

## Build and quantization preparation for the Llama2

The quantization needs to be performed in a separate build environment than the MLPerf container. Please follow the steps below:
```
# Make sure you are outside of the MLPerf container
cd <workdir>
git clone https://github.com/NVIDIA/TensorRT-LLM.git
# TODO: Change to release 0.8.0 later
cd TensorRT-LLM && git checkout 3d56a445e8ebf888e78be638faf6beec0a78f3c2
make -C docker build
make -C docker run LOCAL_USER=1
# The following steps will be performed within TRTLLM container. Change -a=90 to your target architecture
python3 scripts/build_wheel.py -a=90 --clean --install --trt_root /usr/local/tensorrt/
# Quantize the benchmark
# On L40s, you might need TP4
python examples/quantization/quantize.py --dtype=float16  --output_dir=<model_dir>/models/Llama2/fp8-quantized-ammo/llama2-70b-tp2pp1-fp8 --model_dir=<model_dir>/models/Llama2/Llama-2-70b-chat-hf --qformat=fp8 --kv_cache_dtype=fp8 --tp_size 2 
```

## Build and run the benchmarks

Please follow the steps below in MLPerf container:
```
# The current build only supports SM89/SM90. If you want to try SM80 support, please go to Makefile.build and modify the "-a=90" flag from "build_trt_llm" target.
make clone_trt_llm
make build_trt_llm
BUILD_TRTLLM=1 make build_harness
# If the generate engine is complaining about model_path, please change fp8_quant_model_path field in the code/llama2-70b/tensorrt/builder.py
make generate_engines RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy"
make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly"
```

You should expect to get the following results (the detailed number might be different):
```
   accuracy: [PASSED] ROUGE1: 44.495 (Threshold=43.836) | [PASSED] ROUGE2: 22.089 (Threshold=21.689) | [PASSED] ROUGEL: 28.694 (Threshold=28.222) | [PASSED] TOKENS_PER_SAMPLE: 293.100 (Threshold=263.970)
```
