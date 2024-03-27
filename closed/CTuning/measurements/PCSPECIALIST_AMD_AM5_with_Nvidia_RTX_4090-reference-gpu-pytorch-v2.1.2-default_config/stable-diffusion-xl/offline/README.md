This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=generate-run-cmds,inference,_submission \
	--scenario=Offline \
	--model=sdxl \
	--backend=pytorch \
	--device=cuda \
	--adr.mlperf-inference-implementation.num_threads=1 \
	--precision=bfloat16 \
	--offline-target-qps=0.2 \
	--execution-mode=valid \
	--test_query_count=2 \
	--adr.compiler.tags=gcc \
	--quiet \
	--adr.mlperf-inference-implementation.tags=_batch_size.1 \
	--results_dir=/home/arjun/results_dir
```