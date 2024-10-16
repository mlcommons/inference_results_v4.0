# Retinanet Benchmark

This benchmark performs object detection using the [Retinanet](https://arxiv.org/abs/1708.02002v2) network and the open-images-v6 dataset.

:warning: **IMPORTANT**: Please use [closed/NVIDIA](closed/NVIDIA) as the working directory when
running the below commands. :warning:

## Dataset

### Downloading / obtaining the dataset

The dataset used for this benchmark is [open-images-v6](https://storage.googleapis.com/openimages/web/factsfigures_v6.html). You can run `BENCHMARKS=retinanet make download_data` to download the data to `$MLPERF_SCRATCH_PATH/data`.

### Preprocessing the dataset for usage

To process the input images to INT8 NCHW format, please run `BENCHMARKS=retinanet make prepreocess_data`. The preprocessed data will be saved to `$MLPERF_SCRATCH_PATH/preprocessed_data/open-images-v6-mlperf`.

## Model

### Downloading / obtaining the model

The pytorch model *retinanet_model_10.pth* is downloaded from the [zenodo link](https://zenodo.org/record/6605272/files/retinanet_model_10.zip) provided by the MLCommon. The Pytorch model is subsequently processed into an onnx model.

Please run `BENCHMARKS=retinanet make downdload_model` for the download and post-processing.

### Optimizations

#### Plugins

The following TensorRT plugins are used to optimize RetinaNet benchmark:
- `NMS_OPT_TRT`: An optimized implementation of the Non-Maximum Suppresion kernel as a TRT plugin
- `RetinaNetNMSPVATRT`: for Orin only, an optimized implementation of the Non-Maximum Suppresion kernel run on the PVA module, as a TRT Plugin

#### Lower Precision

To further optimize performance, with minimal impact on classification accuracy, we run the computations in INT8 precision.

### Calibration

RetinaNet INT8 is calibrated on a subset of the openimages v6 validation set. The indices of this subset can be found at
`data_maps/open-images-v6-mlperf/cal_map.txt`. We use TensorRT symmetric calibration, and store the scaling factors in
`code/retinanet/tensorrt/calibrator.cache`.

## Instructions for Audits

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```
make run RUN_ARGS="--benchmarks=retinanet --scenarios=<SCENARIO> --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=retinanet --scenarios=<SCENARIO> --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.
