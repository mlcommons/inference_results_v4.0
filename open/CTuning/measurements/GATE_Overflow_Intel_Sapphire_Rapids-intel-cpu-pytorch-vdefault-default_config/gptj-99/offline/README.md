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

cm pull repo ctuning@mlcommons-ck --checkout=f179322cfeac2858edac6d74b29a23107774f1f5

cm run script \
	--tags=generate-run-cmds,inference,_performance-only \
	--implementation=intel-original \
	--model=gptj-99 \
	--quiet \
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

Platform: GATE_Overflow_Intel_Sapphire_Rapids-intel-cpu-pytorch-vdefault-default_config

### Accuracy Results 
`ROUGE1`: `42.9043`, Required accuracy for closed division `>= 42.55663`
`ROUGE2`: `20.0499`, Required accuracy for closed division `>= 19.92226`
`ROUGEL`: `29.9035`, Required accuracy for closed division `>= 29.68822`
`GEN_LEN`: `3928682.0`, Required accuracy for closed division `>= 3615190.2`

### Performance Results 
`Samples per second`: `0.326534`
