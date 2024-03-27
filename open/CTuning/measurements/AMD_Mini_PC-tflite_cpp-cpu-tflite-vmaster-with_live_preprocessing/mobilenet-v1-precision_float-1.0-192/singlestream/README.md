This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=run,mobilenet-models,_tflite,_performance-only \
	--adr.compiler.tags=gcc \
	--adr.mlperf-inference-implementation.compressed_dataset=on \
	--results_dir=/home/arjun/mobilenet_results \
	--env.CM_DATASET_COMPRESSED=off
```