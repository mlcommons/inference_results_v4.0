# GPTJ readme

## Getting started

Please first download the data, model and preprocess the data folloing the steps below
```
BENCHMARKS=gptj make download_model
BENCHMARKS=gptj make download_data
BENCHMARKS=gptj make preprocess_data
```
Make sure after the 3 steps above, you have the model downloaded under `build/models/GPTJ-6B`, and preprocessed data under `build/preprocessed_data/cnn_dailymail_tokenized_gptj/`.

## Build and run the benchmarks

Please follow the steps below:
```
# The current build only supports SM90. If you want to try SM80/89 support, please go to Makefile.build and remove the "-a=90" flag from "build_trt_llm" target.
make build_trt_llm
BUILD_TRTLLM=1 make build_harness
make generate_engines RUN_ARGS="--benchmarks=gptj --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly --fast"
make run_harness RUN_ARGS="--benchmarks=gptj --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly --fast"
```

You should expect to get the following results:
```
   gptj-99.9:
     accuracy: [PASSED] ROUGE1: 43.058 (Threshold=42.944) | [PASSED] ROUGE2: 20.113 (Threshold=20.103) | [PASSED] ROUGEL: 40.231 (Threshold=29.958) | [PASSED] GEN_LEN: 4103698.000 (Threshold=3615190.200)
```
