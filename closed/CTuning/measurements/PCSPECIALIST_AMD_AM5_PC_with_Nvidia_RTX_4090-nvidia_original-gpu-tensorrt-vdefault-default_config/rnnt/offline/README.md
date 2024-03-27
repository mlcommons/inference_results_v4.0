This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: Linux-6.5.0-14-generic-x86_64-with-glibc2.29
* CPU version: x86_64
* Python version: 3.8.10 (default, Nov 22 2023, 10:22:35) 
[GCC 9.4.0]
* MLCommons CM version: 1.6.2
* MLCommons Git https://github.com/mlcommons/inference.git (f9a643c0a0e920588da1b51a1d822e1071a9dbec)
* MLCommons Git https://github.com/mlcommons/inference.git (486a629ea4d5c5150f452d0b0a196bf71fd2021e)
* MLCommons Git https://github.com/mlcommons/power-dev.git (f5ee305a867b24905fea1f04f1b68819d392a731)


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck --checkout=6f0105e57ea33b3749871a4046914b943a340e73

cm run script \
	--tags=generate-run-cmds,inference,_submission,_all-scenarios \
	--model=rnnt \
	--implementation=nvidia-original \
	--device=cuda \
	--backend=tensorrt \
	--category=edge \
	--division=closed \
	--quiet \
	--skip_submission_generation=yes \
	--execution-mode=valid \
	--power=yes \
	--adr.mlperf-power-client.power_server=192.168.1.79 \
	--adr.mlperf-power-client.port=4950 \
	--gpu_name=rtx_4090
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
`WER`: `92.55451`, Required accuracy for closed division `>= 91.62252`

### Performance Results 
`Samples per second`: `15225.6`

### Power Results 
`Power consumed`: `601.863 Watts`, `Power efficiency`: `25297.433 samples per Joule`