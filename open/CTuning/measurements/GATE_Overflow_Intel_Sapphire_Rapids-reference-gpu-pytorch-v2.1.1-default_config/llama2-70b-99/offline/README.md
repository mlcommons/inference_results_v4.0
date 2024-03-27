This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://github.com/mlcommons/ck/tree/master/docs/mlperf) for more details.*

## Host platform

* OS version: Linux-6.2.0-39-generic-x86_64-with-glibc2.37
* CPU version: x86_64
* Python version: 3.11.4 (main, Dec  7 2023, 15:43:41) [GCC 12.3.0]
* MLCommons CM version: 1.6.2
* MLCommons Git https://github.com/mlcommons/inference.git (268bc9dc8a3c0a96bbb7d38482c0ce5016507633)
* MLCommons Git https://github.com/mlcommons/inference.git (22b063c0eb1eaae6a94866a5f5c9d6ac84c9a2e8)
* MLCommons Git https://github.com/neuralmagic/inference (2927a1c2c55bb680a78fdbd78bdd4080f37d0628)


## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo ctuning@mlcommons-ck --checkout=7f78f0864e105f76f14a29634c880a79d4dd6620

cm run script \
	--tags=generate-run-cmds,_performance-only \
	--model=llama2-70b-99 \
	--device=cuda \
	--quiet \
	--precision=bfloat16 \
	--execution-mode=valid \
	--adr.llama2-model.tags=_meta-llama/Llama-2-7b-chat-hf
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload ctuning@mlcommons-ck without checkout and clean CM cache as follows:*

```bash
cm rm repo ctuning@mlcommons-ck
cm pull repo ctuning@mlcommons-ck
cm rm cache -f

```

## Results

Platform: GATE_Overflow_Intel_Sapphire_Rapids-reference-gpu-pytorch-v2.1.1-default_config

### Accuracy Results 
`ROUGE1`: `42.0595`
`ROUGE2`: `19.853`
`ROUGEL`: `26.7729`
`TOKENS_PER_SAMPLE`: `1194.4`

### Performance Results 
`Samples per second`: `199.333`
