This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: Linux-6.2.0-39-generic-x86_64-with-glibc2.37
* CPU version: x86_64
* Python version: 3.11.4 (main, Dec  7 2023, 15:43:41) [GCC 12.3.0]
* MLCommons CM version: 1.6.2
* MLCommons Git https://github.com/mlcommons/inference.git (268bc9dc8a3c0a96bbb7d38482c0ce5016507633)
* MLCommons Git https://github.com/neuralmagic/inference (2927a1c2c55bb680a78fdbd78bdd4080f37d0628)


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck --checkout=b10d843d2bd2a5f7c2abec8e6a60d0add53ec216

cm run script \
	--tags=generate-run-cmds,inference,_submission \
	--model=bert-99 \
	--backend=pytorch \
	--execution-mode=valid \
	--device=cpu \
	--quiet \
	--test_query_count=1000 \
	--sut_servers,=http://192.168.29.46:8000 \
	--network=lon
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload ctuning@mlcommons-ck without checkout and clean CM cache as follows:*

```bash
cm rm repo ctuning@mlcommons-ck
cm pull repo ctuning@mlcommons-ck
cm rm cache -f

```

## Results

Platform: arjun_spr-reference-cpu-pytorch-v2.1.0-default_config

### Accuracy Results 
`F1`: `90.87487`, Required accuracy for closed division `>= 89.96526`

### Performance Results 
`Samples per second`: `145.36`
