This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: Linux-5.10.205-195.804.amzn2.x86_64-x86_64-with-glibc2.2.5
* CPU version: x86_64
* Python version: 3.8.16 (default, Aug 30 2023, 23:19:34) 
[GCC 7.3.1 20180712 (Red Hat 7.3.1-15)]
* MLCommons CM version: 2.0.1
* MLCommons Git https://github.com/mlcommons/inference.git (8219069f83c2396be1c1b4b4c8f80cca756db5ca)


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck --checkout=6956962de99d65bcb289a3f770822272dfc3baf7

cm run script \
	--tags=generate-run-cmds,inference,_submission,_all-scenarios \
	--model=retinanet \
	--implementation=qualcomm \
	--device=qaic \
	--backend=glow \
	--category=edge \
	--division=open \
	--quiet \
	--skip_submission_generation=yes \
	--execution-mode=valid \
	--offline_target_qps=2200 \
	--server_target_qps=1900 \
	--adr.mlperf-inference-implementation.tags=_dl2q.24xlarge
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload ctuning@mlcommons-ck without checkout and clean CM cache as follows:*

```bash
cm rm repo ctuning@mlcommons-ck
cm pull repo ctuning@mlcommons-ck
cm rm cache -f

```

## Results

Platform: AWS_dl2q.24xlarge-kilt-qaic-glow-vdefault-default_config

### Accuracy Results 
`mAP`: `37.234`, Required accuracy for closed division `>= 37.1745`

### Performance Results 
`Samples per second`: `2272.64`
