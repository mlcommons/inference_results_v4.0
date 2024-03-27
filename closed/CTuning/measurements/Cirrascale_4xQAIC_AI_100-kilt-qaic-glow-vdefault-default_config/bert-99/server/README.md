This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: Linux-5.15.0-94-generic-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]
* MLCommons CM version: 2.0.0
* MLCommons Git https://github.com/mlcommons/inference.git (268bc9dc8a3c0a96bbb7d38482c0ce5016507633)
* MLCommons Git https://github.com/mlcommons/inference.git (15d14c95c86c7a42215c162827bab2daaec61da1)


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck --checkout=836fac0e7b843f4374904f1508e857a0fa814018

cm run script \
	--tags=generate-run-cmds,_submission \
	--implementation=qualcomm \
	--model=bert-99 \
	--test_query_count=55000 \
	--quiet \
	--adr.mlperf-inference-implementation.tags=_pro,_num-devices.4,_nsp.16 \
	--execution-mode=valid \
	--server_target_qps=2550 \
	--scenario=Server \
	--division=closed
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload ctuning@mlcommons-ck without checkout and clean CM cache as follows:*

```bash
cm rm repo ctuning@mlcommons-ck
cm pull repo ctuning@mlcommons-ck
cm rm cache -f

```

## Results

Platform: Cirrascale_4xQAIC_AI_100-kilt-qaic-glow-vdefault-default_config

### Accuracy Results 
`F1`: `90.07163`, Required accuracy for closed division `>= 89.96526`

### Performance Results 
`Scheduled samples per second`: `2551.06`
