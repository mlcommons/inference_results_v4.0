## Setup instructions for partners

This is not automated, this is still in development and is not quite polished yet.

1. Create a directory for storing DLRMv2 files. I will let this be `/path/to/dlrmv2/files` as an example.
2. Create `criteo` and `model` subdirectories under this directory.
3. The `weights.zip` file from the [model file download instructions](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch#downloading-model-weights) should be downloaded or moved to the `/path/to/dlrmv2/files/model` directory.
4. Unzip `weights.zip`. There should now be a directory called `/path/to/dlrmv2/files/model/model_weights`.
5. After you have run the Criteo preprocessing scripts, create a directory with the path `/path/to/dlrmv2/files/criteo/day23/fp32`.
6. Copy `day_23_dense.npy` and `day_23_labels.npy` into `/path/to/dlrmv2/files/criteo/day23/fp32` from the `numpy_contiguous_shuffled` directory created by the preprocessing script.
7. Copy `day_23_sparse_multi_hot.npz` into `/path/to/dlrmv2/files/criteo/day23/fp32` from the `synthetic_multihot` directory created by the preprocessing script.
8. Manually unpack the `.npz` file with `unzip /path/to/dlrmv2/files/criteo/day23/fp32/day_23_sparse_multi_hot.npz -d /path/to/dlrmv2/files/criteo/day23/fp32/day_23_sparse_multi_hot_unpacked`
9. Copy the `day_23` file (the raw file from the original Criteo download that was extracted from `day_23.gz`, *not* the
   preprocessed one) to `/path/to/dlrmv2/files/criteo/day23/raw_data` (For clarification, the *filename* should be
   `raw_data`, it should *not* be a directory named `raw_data` with `day_23` inside). This is required for the accuracy
   run to work.
10. Launch the container with `make prebuild DOCKER_ARGS="-v /path/to/dlrmv2/files:/home/mlperf_inf_dlrmv2"` to mount the DLRMv2 files into the container at `/home/mlperf_inf_dlrmv2`.

## Instructions to run

Run the benchmark just like any other benchmark with `make generate_engines` and `make run_harness`.
