# Unified interface to run MLPerf inference benchmarks

Running the [MLPerf inference benchmarks](https://arxiv.org/abs/1911.02549) and preparing valid submissions 
[is not trivial](https://doi.org/10.5281/zenodo.10605079).

This guide explains how to automate all the steps required to prepare, 
customize, run and extend MLPerf inference benchmarks across 
diverse models, datasets, software and hardware using 
the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

CM makes it possible to compose modular benchmarks from [portable and reusable automation recipes (CM scripts)](https://access.cknowledge.org/playground/?action=scripts) 
with a common interface and a [human-friendly GUI](https://access.cknowledge.org/playground/?action=howtorun&bench_uid=39877bb63fb54725).
Such benchmarks attempt to automatically adapt to any software and hardware natively or inside a container with any Operating System.

CM automation for MLPerf benchmarks is being developed by the MLCommons Task Force on Automation and Reproducibility
based on the feedback from MLCommons organizations while automating >90% of all performance and power submissions in the v3.1 round.

# CM interface to run MLPerf inference

The below command is an example of running Nvidia implementation of gptj on Nvidia GPU using CM and generating an edge category submission. Here, `gptj-99` can also be changed to any other MLPerf inference model. If we change `--implementation=nvidia-original` to `--implementation=intel-original` we can run the same benchmark but using Intel implementation on CPUs.
```
cmr "run-mlperf inference _find-performance _all-scenarios" \
--model=gptj-99 --implementation=nvidia-original \
--category=edge --division=open --quiet
```

Please take a look at the individual benchmark folders for the specialized README files for generating an end-to-end MLPerf inference submission.

Don't hesitate to get in touch via [public Discord server](https://discord.gg/JjWNWXKxwT) to get free help to run MLPerf benchmarks and submit valid results.
