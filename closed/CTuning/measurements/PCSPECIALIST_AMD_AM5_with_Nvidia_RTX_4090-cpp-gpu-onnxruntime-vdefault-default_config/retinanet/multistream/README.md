This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=generate-run-cmds,inference,_submission \
	--model=retinanet \
	--implementation=cpp \
	--device=cuda \
	--backend=onnxruntime \
	--scenario=SingleStream \
	--category=edge \
	--division=closed \
	--quiet \
	--adr.compiler.tags=gcc \
	--execution-mode=valid \
	--skip_submission_generation=yes \
	--results_dir=/home/arjun/results_dir
```