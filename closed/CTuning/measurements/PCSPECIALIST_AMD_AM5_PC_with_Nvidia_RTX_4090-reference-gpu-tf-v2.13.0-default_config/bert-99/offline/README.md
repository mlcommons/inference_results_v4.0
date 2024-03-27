This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=generate-run-cmds,inference,_submission,_all-scenarios \
	--model=bert-99 \
	--implementation=reference \
	--device=cuda \
	--backend=tf \
	--category=edge \
	--division=closed \
	--quiet \
	--results_dir=/home/arjun/results_dir \
	--skip_submission_generation=yes \
	--execution-mode=valid \
	--adr.inference-src.tags=_repo.https://github.com/GATEOverflow/inference
```