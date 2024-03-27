This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: Linux-6.5.0-17-generic-x86_64-with-glibc2.38
* CPU version: x86_64
* Python version: 3.11.6 (main, Oct  8 2023, 05:06:43) [GCC 13.2.0]
* MLCommons CM version: 1.5.3
* MLCommons Git https://github.com/mlcommons/inference.git (180014ad5724de84fd16a42bef69e8f58faf9e4f)
* MLCommons Git https://github.com/mlcommons/inference.git (3ad853426528a3d692ea34a72b81a7c4fed0346e)
* MLCommons Git https://github.com/mlcommons/inference.git (268bc9dc8a3c0a96bbb7d38482c0ce5016507633)
* MLCommons Git https://github.com/neuralmagic/inference (2927a1c2c55bb680a78fdbd78bdd4080f37d0628)


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck --checkout=840c88462be3726fcc8736650e68f7acc0796cf3

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
	--adr.mlperf-inference-implementation.max_batchsize=1 \
	--results_dir=/home/arjun/results_dir \
	--env.CM_MLPERF_NEURALMAGIC_MODEL_ZOO_STUB=zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none \
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

Platform: AMD_Mini_PC-reference-cpu-deepsparse-vdefault-default_config

### Accuracy Results 
`F1`: `87.88571`, Required accuracy for closed division `>= 89.96526`

### Performance Results 
`90th percentile latency (ns)`: `29490806.0`
