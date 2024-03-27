# Stable Diffusion XL Benchmark


## Getting started

Please first download the data, closed division model and preprocess the data folloing the steps below
```
BENCHMARKS=stable-diffusion-xl make download_model
BENCHMARKS=stable-diffusion-xl make download_data
BENCHMARKS=stable-diffusion-xl make preprocess_data
```
The closed division PyTorch model is downloaded from the [Hugging Face snapshot](https://cloud.mlcommons.org/index.php/s/DjnCSGyNBkWA4Ro) provided by the MLCommon. The Pytorch model is subsequently processed into 4 onnx models. Make sure after the 3 steps above, you have the closed division models downloaded under `build/models/SDXL/onnx_models`, and preprocessed data under `build/preprocessed_data/coco2014-tokenized-sdxl/`.

NVIDIA has two open SDXL submissions for v4.0. One is referred to as DeepCache, another is referred to as DeepCachePruned. Each submission has its own optimized model. The models are available at: https://drive.google.com/drive/folders/18BWAtkM3xT2C-S0w3YUfaPf01VeTwgSQ. Please note that:

These models are pruned or sparsified versions of some of the standard MLPerf inference models. They were created by NVIDIA for its open submissions to the MLPerf Inference 4.0 round, and we are making them available here to MLCommons members. The modified GPT-J and SDXL models are available only to members and should not be distributed outside of MLCommons.

AI models generate responses and outputs based on complex algorithms and machine learning techniques, and those responses or outputs may be inaccurate or indecent. By testing this model, you assume the risk of any harm caused by any response or output of the model. Please do not use any confidential information or personal data with these models.

After you receive the models, please put the models under `build/open-models/SDXL-DeepCache` or `build/open-models/SDXL-DeepCachePruned` respectively.

## Optimizations

For the v4.0 submission, We use our TensorRT model optimization toolkit to generate our optimized models based on the closed division model. For the DeepCache submission, we make a shallow UNet model. Together with the original deep UNet model, we can roughly halve the computation cost during denosing while still keep the closed division required accuracies. For the DeepCachePruned submission, we replace the deep UNet model by a pruned model. It further increases the performance of the benchmark but suffering the cost of accuracy drop. More details about the optimization please refer to NVIDIA's techinial report on MLPerf v4.0 submissions.

## Build and quantization preparation for the Llama2

The SDXL optimized models are already quantized.

## Build and run the benchmarks

Please follow the steps below in MLPerf container:

```
make build
# Please update configs/stable-diffusion-xl to include your custom machine config before building the engine
# Please use --model_opt=DeepCachePruned for running the DeepCachePruned model
make generate_engines RUN_ARGS="--benchmarks=stable-diffusion-xl --scenarios=Offline --model_opt=DeepCache"
# Please use --model_opt=DeepCachePruned for running the DeepCachePruned model
make run_harness RUN_ARGS="--benchmarks=stable-diffusion-xl --scenarios=Offline --test_mode=AccuracyOnly --model_opt=DeepCache"
```

You should expect to get the following results:
```
  stable-diffusion-xl:
    accuracy: [PASSED] CLIP_SCORE: 31.713 (Valid Range=[31.686,  31.813]) | [PASSED] FID_SCORE: 23.605 (Valid Range=[23.011,  23.950])

```
