This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=run,mobilenet-models,_tflite,_armnn,_neon,_populate-readme \
	--adr.mlperf-inference-implementation.compressed_dataset=on \
	--adr.compiler.tags=gcc \
	--results_dir=/home/cmuser/results_dir
```
## Dependent CM scripts 


1.  `cm run script --tags=detect,os`


2.  `cm run script --tags=get,sys-utils-cm`


3.  `cm run script --tags=get,python`


4.  `cm run script --tags=get,mlcommons,inference,src`


5.  `cm run script --tags=get,dataset-aux,imagenet-aux`


## Results

Platform: macbook_pro_m1_2-tflite_armnn_cpp-cpu-armnn_tflite-v23.05-using_neon_with_live_preprocessing

### Accuracy Results 
`acc`: `68.336`, Required accuracy for closed division `>= 75.6954`

### Performance Results 
`90th percentile latency (ns)`: `9667125.0`
