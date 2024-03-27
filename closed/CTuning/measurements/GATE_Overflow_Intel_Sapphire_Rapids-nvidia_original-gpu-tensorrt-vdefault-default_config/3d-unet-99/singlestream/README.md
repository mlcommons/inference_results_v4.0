This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: Linux-6.2.0-39-generic-x86_64-with-glibc2.29
* CPU version: x86_64
* Python version: 3.8.10 (default, Nov 22 2023, 10:22:35) 
[GCC 9.4.0]
* MLCommons CM version: 1.6.2
* MLCommons Git https://github.com/mlcommons/inference.git (f9a643c0a0e920588da1b51a1d822e1071a9dbec)
* MLCommons Git https://github.com/mlcommons/inference.git (486a629ea4d5c5150f452d0b0a196bf71fd2021e)


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck --checkout=4d427f5cfee79fe1da7d877adb8dc7de630750f6

cm run script \
	--tags=generate-run-cmds,inference,_submission,_all-scenarios \
	--model=3d-unet-99.9 \
	--implementation=nvidia-original \
	--device=cuda \
	--backend=tensorrt \
	--category=edge \
	--division=closed \
	--quiet \
	--skip_submission_generation=yes \
	--execution-mode=valid \
	--gpu_name=rtx_4090 \
	--offline_target_qps=90000 \
	--offline_target_qps=1700 \
	--offline_target_qps=8000 \
	--offline_target_qps=8
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
`DICE`: `0.86236`, Required accuracy for closed division `>= 0.86084`

### Performance Results 
`90th percentile latency (ns)`: `436008704.0`
