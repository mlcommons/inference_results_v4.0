This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: Windows-10-10.0.22631-SP0
* CPU version: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
* Python version: 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]
* MLCommons CM version: 2.0.0


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck

cm run script ^
	--tags=run-mlperf-inference,_all-scenarios,_r4.0,_submission,_full ^
	--division=open ^
	--category=edge ^
	--device=cpu ^
	--model=resnet50 ^
	--precision=float32 ^
	--implementation=reference ^
	--backend=onnxruntime ^
	--execution_mode=valid ^
	--submitter=CTuning ^
	--power=no ^
	--adr.python.version_min=3.8 ^
	--compliance=no ^
	--j ^
	--jf=run-8bd1d268b6e54749 ^
	--quiet ^
	--time ^
	--host_os=windows
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload ctuning@mlcommons-ck without checkout and clean CM cache as follows:*

```bash
cm rm repo ctuning@mlcommons-ck
cm pull repo ctuning@mlcommons-ck
cm rm cache -f

```

## Results

Platform: LENOVO_P14S_GEN_4_WINDOWS_11-reference-cpu-onnxruntime-v1.16.0-default_config

### Accuracy Results 
`acc`: `76.456`, Required accuracy for closed division `>= 75.6954`

### Performance Results 
`Samples per query`: `266664000.0`
