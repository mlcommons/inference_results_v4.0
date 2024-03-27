This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: macOS-14.3.1-arm64-arm-64bit
* CPU version: arm
* Python version: 3.9.18 (main, Sep 11 2023, 08:25:10) 
[Clang 14.0.6 ]
* MLCommons CM version: 2.0.0
* MLCommons Git https://github.com/mlcommons/inference.git (8219069f83c2396be1c1b4b4c8f80cca756db5ca)
* MLCommons Git https://github.com/neuralmagic/inference (2927a1c2c55bb680a78fdbd78bdd4080f37d0628)


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck --checkout=e1a1f52a17a43f10b81420dcad9f0ef3073a48cb

cm run script \
	--tags=run,mlperf,inference,generate-run-cmds,_submission \
	--adr.python.version_min=3.8 \
	--adr.compiler.tags=gcc \
	--implementation=reference \
	--model=bert-99 \
	--precision=int8 \
	--backend=deepsparse \
	--device=cpu \
	--scenario=Offline \
	--execution_mode=valid \
	--adr.mlperf-inference-implementation.max_batchsize=1 \
	--results_dir=/Users/tejasmalladi/results_dir \
	--env.CM_MLPERF_NEURALMAGIC_MODEL_ZOO_STUB=zoo:nlp/question_answering/bert-large/pytorch/huggingface/squad/base-none \
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

Platform: Tejasvis_MBP-reference-cpu-deepsparse-vdefault-default_config

### Accuracy Results 
`F1`: `89.65191`, Required accuracy for closed division `>= 89.96526`

### Performance Results 
`Samples per second`: `2.68036`
