This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: Linux-6.5.0-14-generic-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]
* MLCommons CM version: 1.6.2
* MLCommons Git https://github.com/mlcommons/inference.git (f9a643c0a0e920588da1b51a1d822e1071a9dbec)
* MLCommons Git https://github.com/neuralmagic/inference (2927a1c2c55bb680a78fdbd78bdd4080f37d0628)
* MLCommons Git https://github.com/mlcommons/power-dev.git (f5ee305a867b24905fea1f04f1b68819d392a731)


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck --checkout=f179322cfeac2858edac6d74b29a23107774f1f5

cm run script \
	--tags=run,mlperf,inference,generate-run-cmds,_submission \
	--adr.python.version_min=3.8 \
	--adr.compiler.tags=gcc \
	--implementation=reference \
	--model=bert-99 \
	--precision=int8 \
	--backend=deepsparse \
	--device=cpu \
	--scenario=SingleStream \
	--execution_mode=valid \
	--skip_submission_generation=yes \
	--adr.mlperf-inference-implementation.max_batchsize=128 \
	--power=yes \
	--adr.mlperf-power-client.power_server=192.168.1.79 \
	--results_dir=/home/arjun/results_dir \
	--env.CM_MLPERF_NEURALMAGIC_MODEL_ZOO_STUB=zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base_quant-none \
	--quiet
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload ctuning@mlcommons-ck without checkout and clean CM cache as follows:*

```bash
cm rm repo ctuning@mlcommons-ck
cm pull repo ctuning@mlcommons-ck
cm rm cache -f

```

## Results

Platform: PCSPECIALIST_AMD_AM5_PC_with_Nvidia_RTX_4090-reference-cpu-deepsparse-vdefault-default_config

### Accuracy Results 
`F1`: `90.79585`, Required accuracy for closed division `>= 89.96526`

### Performance Results 
`90th percentile latency (ns)`: `9580459.0`

### Power Results 
`Power consumed`: `1986.846 milliJoules`, `Power efficiency`: `503.31 samples per Joule`