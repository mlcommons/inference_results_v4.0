This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: Linux-6.1.21-v8+-aarch64-with-glibc2.31
* CPU version: 
* Python version: 3.9.2 (default, Feb 28 2021, 17:03:44) 
[GCC 10.2.1 20210110]
* MLCommons CM version: 1.5.0
* MLCommons Git https://github.com/mlcommons/inference.git (3ad853426528a3d692ea34a72b81a7c4fed0346e)


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck --checkout=81f79f792cf7e3ef6c61cf90c183b891f10435f9

cm run script \
	--tags=run,mobilenet-models,_tflite,_populate-readme \
	--adr.compiler.tags=gcc \
	--adr.mlperf-inference-implementation.compressed_dataset=on \
	--results_dir=/home/arjun/results_dir
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload ctuning@mlcommons-ck without checkout and clean CM cache as follows:*

```bash
cm rm repo ctuning@mlcommons-ck
cm pull repo ctuning@mlcommons-ck
cm rm cache -f

```

## Results

Platform: Raspberry_Pi_4-tflite_cpp-cpu-tflite-vmaster-default_config

### Accuracy Results 
`acc`: `49.168`, Required accuracy for closed division `>= 75.6954`

### Performance Results 
`90th percentile latency (ns)`: `17213999.0`
