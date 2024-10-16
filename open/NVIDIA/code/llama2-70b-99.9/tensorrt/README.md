# Llama2 readme

## Getting started

Please first download the data, model and preprocess the data folloing the steps below
```
# Visit https://docs.google.com/forms/d/e/1FAIpQLSc_8VIvRmXM3I8KQaYnKf7gy27Z63BBoI_I1u02f4lw6rBp3g/viewform to sign the agreement,
# which gets you access to the dataset
# Unzip the llama dataset pickle file
gzip -d <llama_dataset>.pkl.gz
# Please place the llama2 closed division models (Llama-2-70b-chat-hf) under build/models/Llama2 and datasets (open_orca) under build/preprocessed_data
BENCHMARKS=llama2 make preprocess_data
```
Make sure after the 3 steps above, you have the closed division model downloaded under `build/models/Llama2`, and preprocessed data under `build/preprocessed_data/open_orca/`.

The NVIDIA v4.0 open Llama2-70b submission has an optimized sparse model. The model is available after signing the agreement at: https://docs.google.com/forms/d/e/1FAIpQLSc_8VIvRmXM3I8KQaYnKf7gy27Z63BBoI_I1u02f4lw6rBp3g/viewform. Please navigate to open-models > nvidia-inference-v4.0. Please note that:

This model is a sparsified version of one of the standard MLPerf inference models. It was created by NVIDIA for its open submissions to the MLPerf Inference 4.0 round, and we are making it available here to MLCommons members. These modified Llama 2 model files are subject to the same licensing terms as our Llama 2-70B base models, so they are accessible to members but should not be distributed outside of MLCommons.

AI models generate responses and outputs based on complex algorithms and machine learning techniques, and those responses or outputs may be inaccurate or indecent. By testing this model, you assume the risk of any harm caused by any response or output of the model. Please do not use any confidential information or personal data with these models.

After you receive the model, please put the model under `build/open-models/LLAMA2-70B-Sparse/FP8_quantized`.

## Optimizations

For the v4.0 submission, we use our TensorRT model optimization toolkit to generate a sparse model based on the closed division Llama2-70b model. The optmized model introduces 2:4 sparsity on all attention and MLP blocks of the Llama2-70B-chat base model. We hit all 99.9% high accuracy on the Rouge Score metrics without any finetunig. More details about the optimization please refer to NVIDIA's techinial report on MLPerf v4.0 submissions.


## Build and quantization preparation for the Llama2

The Llama2-70b optimized model is already quantized.

## Build and run the benchmarks

Please follow the steps below in MLPerf container:
```
# The current build only supports SM89/SM90. If you want to try SM80 support, please go to Makefile.build and modify the "-a=90" flag from "build_trt_llm" target.
BUILD_TRTLLM=1 make build
# Please update configs/llama2-70b to include your custom machine config before building the engine
make generate_engines RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy --model_opt=Sparse"
make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly --model_opt=Sparse"
```

You should expect to get the results similar to the following:
```
     llama2-70b-99.9:
       accuracy: [PASSED] ROUGE1: 44.486 (Threshold=44.387) | [PASSED] ROUGE2: 22.435 (Threshold=22.013) | [PASSED] ROUGEL: 29.931 (Threshold=28.588) | TOKENS_PER_SAMPLE: 235.300 (Threshold=265.005)

```
