This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=run,mobilenet-models,_tflite,_populate-readme \
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

Platform: macbook_pro_m1_2-tflite_cpp-cpu-tflite-vmaster-default_config

### Accuracy Results 
`acc`: `64.996`, Required accuracy for closed division `>= 75.6954`

### Performance Results 
`90th percentile latency (ns)`: `5473125.0`
