# MLPerf Inference - Qualcomm Cloud AI - BERT

This implementation supports the MLPerf Inference BERT workload on systems
equipped with Qualcomm AI Cloud (QAIC) accelerators using KRAI Inference
Library Technology (KILT).

Follow [instructions](https://github.com/krai/axs2qaic-docker) to set up the
Docker environment.

Below are example commands for benchmarking both flavours of BERT (BERT-99 and
BERT-99.9) under the Offline scenario which is common to the Edge and
Datacenter categories.

In the Docker container, define the `SUT` variable for your system-under-test
(SUT) e.g.:

```
export SUT=g293_q16_ultra
```

as well as the `OFFLINE_TARGET_QPS` variable for the expected queries per
second (QPS).

## BERT-99

### Offline

#### [Optional] Compile model
```
axs byquery kilt_ready,device=qaic,sut_name=${SUT},\
model_name=bert-99,loadgen_scenario=Offline
```

#### [Optional] Compile program
```
axs byquery compiled,lib_kilt --- , remove && \
axs byquery compiled,kilt_executable,device=qaic,bert --- , remove && \
axs byquery compiled,lib_kilt && \
axs byquery compiled,kilt_executable,device=qaic,bert
```

#### Accuracy
##### Quick Run
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=bert,model_name=bert-99,\
loadgen_mode=AccuracyOnly,\
loadgen_scenario=Offline,\
collection_name=experiments,\
sut_name=${SUT} \
, get accuracy_report
```

##### Full Run (same as Quick Run)
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=bert,model_name=bert-99,\
loadgen_mode=AccuracyOnly,\
loadgen_scenario=Offline,\
collection_name=experiments,\
sut_name=${SUT} \
, get accuracy_report
```

<details><summary><code>f1=90.284</code></summary><pre>
{"exact_match": 82.88552507095554, "f1": 90.28408874066913}
</pre></details>

#### Performance
##### Quick Run (10 seconds)
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=bert,model_name=bert-99,\
loadgen_mode=PerformanceOnly,loadgen_min_duration_s=10,\
loadgen_scenario=Offline,\
loadgen_target_qps=${OFFLINE_TARGET_QPS},\
collection_name=experiments,\
sut_name=${SUT} \
, get performance
```

##### Full Run (10 minutes)
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=bert,model_name=bert-99,\
loadgen_mode=PerformanceOnly,\
loadgen_scenario=Offline,\
loadgen_target_qps=${OFFLINE_TARGET_QPS},\
collection_name=experiments,\
sut_name=${SUT} \
, get performance
```

## BERT-99.9

### Offline

#### [Optional] Compile model
```
axs byquery kilt_ready,device=qaic,sut_name=${SUT},\
model_name=bert-99.9,loadgen_scenario=Offline
```

#### [Optional] Compile program
```
axs byquery compiled,lib_kilt --- , remove && \
axs byquery compiled,kilt_executable,device=qaic,bert --- , remove && \
axs byquery compiled,lib_kilt && \
axs byquery compiled,kilt_executable,device=qaic,bert
```

#### Accuracy
##### Quick Run
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=bert,model_name=bert-99.9,\
loadgen_mode=AccuracyOnly,\
loadgen_scenario=Offline,\
collection_name=experiments,\
sut_name=${SUT} \
, get accuracy_report
```

##### Full Run (same as Quick Run)
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=bert,model_name=bert-99.9,\
loadgen_mode=AccuracyOnly,\
loadgen_scenario=Offline,\
collection_name=experiments,\
sut_name=${SUT} \
, get accuracy_report
```

<details><summary><code>f1=90.867</code></summary><pre>
{"exact_match": 83.6802270577105, "f1": 90.8666333229796}
</pre></details>

#### Performance
##### Quick Run (10 seconds)
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=bert,model_name=bert-99.9,\
loadgen_mode=PerformanceOnly,loadgen_min_duration_s=10,\
loadgen_scenario=Offline,\
loadgen_target_qps=${OFFLINE_TARGET_QPS},\
collection_name=experiments,\
sut_name=${SUT} \
, get performance
```

##### Full Run (10 minutes)
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=bert,model_name=bert-99.9,\
loadgen_mode=PerformanceOnly,\
loadgen_scenario=Offline,\
loadgen_target_qps=${OFFLINE_TARGET_QPS},\
collection_name=experiments,\
sut_name=${SUT} \
, get performance
```
