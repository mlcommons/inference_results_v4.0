# GPTJ Benchmark

## Getting started

Please first download the data, closed division model and preprocess the data folloing the steps below
```
BENCHMARKS=gptj make download_model
BENCHMARKS=gptj make download_data
BENCHMARKS=gptj make preprocess_data
```
Make sure after the 3 steps above, you have the closed division model downloaded under `build/models/GPTJ-6B`, and preprocessed data under `build/preprocessed_data/cnn_dailymail_tokenized_gptj/`.

The NVIDIA v4.0 open GPTJ submission has an optimized pruned model. The model is available at: https://drive.google.com/drive/folders/18BWAtkM3xT2C-S0w3YUfaPf01VeTwgSQ. Please note that:

These models are pruned or sparsified versions of some of the standard MLPerf inference models. They were created by NVIDIA for its open submissions to the MLPerf Inference 4.0 round, and we are making them available here to MLCommons members. The modified GPT-J and SDXL models are available only to members and should not be distributed outside of MLCommons.

AI models generate responses and outputs based on complex algorithms and machine learning techniques, and those responses or outputs may be inaccurate or indecent. By testing this model, you assume the risk of any harm caused by any response or output of the model. Please do not use any confidential information or personal data with these models.

After you receive the model, please put the model under `build/open-models/GPTJ-6B-Pruned/FP8_quantized`.

## Optimizations

For the v4.0 submission, we use our TensorRT model optimization toolkit to generate a pruned model based on the closed division GPTJ model. The optmized model pruned out 30% of the network. We get 98.5% of Rouge Score accuracies. More details about the optimization please refer to NVIDIA's techinial report on MLPerf v4.0 submissions.

## Build and quantization preparation for GPTJ

The GPTJ optimized model is already quantized.

## Build and run the benchmarks

Please follow the steps below:
```
# Enter the MLPerf container
make prebuild
# The current build only supports SM90. If you want to try SM80/89 support, please go to Makefile.build and remove the "-a=90" flag from "build_trt_llm" target.
BUILD_TRTLLM=1 make build
# Please update configs/gptj to include your custom machine config before building the engine
make generate_engines RUN_ARGS="--benchmarks=gptj --scenarios=Offline --model_opt=Pruned"
make run_harness RUN_ARGS="--benchmarks=gptj --scenarios=Offline --test_mode=AccuracyOnly --model_opt=Pruned"
```

You should expect to get the results similar to the following:
{'rouge1':42.3672, 'rouge2': 19.8264, 'rougeL': 29.7337, 'rougeLsum': 39.5676, 'gen_len': 3863701, 'gen_num': 13368}


```
  gptj-99:
    accuracy: ROUGE1: 42.367 (Threshold=42.557) | ROUGE2: 19.826 (Threshold=19.922) | [PASSED] ROUGEL: 29.734 (Threshold=29.688) | [PASSED] GEN_LEN: 3863701.000 (Threshold=3615190.200)
```
