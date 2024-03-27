This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: Linux-6.5.0-14-generic-x86_64-with-glibc2.29
* CPU version: x86_64
* Python version: 3.8.10 (default, Nov 22 2023, 10:22:35) 
[GCC 9.4.0]
* MLCommons CM version: 1.6.2
* MLCommons Git https://github.com/mlcommons/inference.git (8219069f83c2396be1c1b4b4c8f80cca756db5ca)
* MLCommons Git https://github.com/mlcommons/power-dev.git (f5ee305a867b24905fea1f04f1b68819d392a731)


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck --checkout=12d75c17a629ecfbd14ee1b4ff3b6824ea54ffc9

cm run script \
	--tags=generate-run-cmds,_submission \
	--model=dlrm-v2-99 \
	--implementation=nvidia-original \
	--env.DLRM_DATA_PATH=on \
	--offline_target_qps=1500 \
	--execution-mode=valid \
	--power=yes \
	--adr.mlperf-power-client.power_server=192.168.1.79
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload ctuning@mlcommons-ck without checkout and clean CM cache as follows:*

```bash
cm rm repo ctuning@mlcommons-ck
cm pull repo ctuning@mlcommons-ck
cm rm cache -f

```

## Results

Platform: PCSPECIALIST_AMD_AM5_PC_with_Nvidia_RTX_4090-nvidia_original-gpu-tensorrt-vdefault-default_config

### Accuracy Results 
`AUC`: `62.297`, Required accuracy for closed division `>= 79.5069`

### Performance Results 
`Samples per second`: `1546.75`

### Power Results 
`Power consumed`: `281.617 Watts`, `Power efficiency`: `5492.397 samples per Joule`