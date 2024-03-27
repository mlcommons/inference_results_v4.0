This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: Linux-6.2.0-39-generic-x86_64-with-glibc2.29
* CPU version: x86_64
* Python version: 3.8.10 (default, Nov 22 2023, 10:22:35) 
[GCC 9.4.0]
* MLCommons CM version: 1.6.2
* MLCommons Git https://github.com/mlcommons/inference.git (8219069f83c2396be1c1b4b4c8f80cca756db5ca)


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck --checkout=6956962de99d65bcb289a3f770822272dfc3baf7

cm run script \
	--tags=generate-run-cmds,_submission \
	--implementation=nvidia-original \
	--offline_target_qps=3000 \
	--execution-mode=valid \
	--model=dlrm-v2-99 \
	--env.DLRM_DATA_PATH=on
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload ctuning@mlcommons-ck without checkout and clean CM cache as follows:*

```bash
cm rm repo ctuning@mlcommons-ck
cm pull repo ctuning@mlcommons-ck
cm rm cache -f

```

## Results

Platform: GATE_Overflow_Intel_Sapphire_Rapids-nvidia_original-gpu-tensorrt-vdefault-default_config

### Accuracy Results 
`AUC`: `62.297`, Required accuracy for closed division `>= 79.5069`

### Performance Results 
`Samples per second`: `2988.18`
