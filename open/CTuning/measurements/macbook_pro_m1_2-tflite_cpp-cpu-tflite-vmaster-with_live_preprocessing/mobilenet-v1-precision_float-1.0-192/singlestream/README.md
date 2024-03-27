This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=run,mobilenet-models,_tflite,_populate-readme \
	--adr.compiler.tags=gcc \
	--adr.mlperf-inference-implementation.compressed_dataset=on \
	--results_dir=/home/cmuser/results_dir
```
## Dependent CM scripts 


1.  `cm run script --tags=detect,os`


2.  `cm run script --tags=get,sys-utils-cm`


3.  `cm run script --tags=get,python`


4.  `cm run script --tags=get,mlcommons,inference,src`


5.  `cm run script --tags=get,dataset-aux,imagenet-aux`


## Results

Platform: macbook_pro_m1_2-tflite_cpp-cpu-tflite-vmaster-with_live_preprocessing

### Accuracy Results 
`acc`: `70.624`, Required accuracy for closed division `>= 75.6954`

### Performance Results 
`90th percentile latency (ns)`: `55998000.0`
