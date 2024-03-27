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

cm pull repo ctuning@mlcommons-ck --checkout=836fac0e7b843f4374904f1508e857a0fa814018

cm run script \
	--tags=generate-run-cmds,inference,_submission \
	--model=bert-99 \
	--backend=onnxruntime \
	--rerun \
	--mode=performance \
	--device=cpu \
	--quiet \
	--test_query_count=1000 \
	--sut_servers,=http://192.168.1.82:8000 \
	--network=lon \
	--execution-mode=valid
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload ctuning@mlcommons-ck without checkout and clean CM cache as follows:*

```bash
cm rm repo ctuning@mlcommons-ck
cm pull repo ctuning@mlcommons-ck
cm rm cache -f

```

## Results

Platform: AMD_AM5_with_Nvidia_RTX_4090_(Network)-reference-cpu-onnxruntime-v1.17.0-network_loadgen

### Accuracy Results 
`F1`: `90.87487`, Required accuracy for closed division `>= 89.96526`

### Performance Results 
`Samples per second`: `184.308`
