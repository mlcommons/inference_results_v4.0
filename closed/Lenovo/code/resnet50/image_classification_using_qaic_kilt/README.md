# MLPerf Inference - Qualcomm Cloud AI - ResNet50

This implementation supports the MLPerf Inference ResNet50 workload on systems
equipped with Qualcomm AI Cloud (QAIC) accelerators using KRAI Inference
Library Technology (KILT).

Follow [instructions](https://github.com/krai/axs2qaic-docker) to set up the
Docker environment.

Below are example commands for benchmarking ResNet50 under the Offline scenario
which is common to the Edge and Datacenter categories.

In the Docker container, define the `SUT` variable for your system-under-test
(SUT) e.g.:

```
export SUT=g293_q16_ultra
```

as well as the `OFFLINE_TARGET_QPS` variable for the expected queries per
second (QPS).

## ResNet50

### Offline

#### [Optional] Compile model
```
axs byquery kilt_ready,device=qaic,sut_name=${SUT},\
model_name=resnet50,index_file=cal_image_list_option_1.txt,\
loadgen_scenario=Offline
```

#### [Optional] Compile program
```
axs byquery compiled,lib_kilt --- , remove && \
axs byquery compiled,kilt_executable,device=qaic,resnet50 --- , remove && \
axs byquery compiled,lib_kilt && \
axs byquery compiled,kilt_executable,device=qaic,resnet50
```

#### Accuracy
##### Quick Run (500 images)
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=image_classification,model_name=resnet50,\
loadgen_mode=AccuracyOnly,loadgen_dataset_size=500,\
loadgen_scenario=Offline,\
collection_name=experiments,\
sut_name=${SUT} \
, get accuracy_report
```
<details><summary><code>top1=75.6%</code></summary><pre>
accuracy=75.600%, good=378, total=500
</pre></details>

##### Full Run (50,000 images)
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=image_classification,model_name=resnet50,\
loadgen_mode=AccuracyOnly,\
loadgen_scenario=Offline,\
collection_name=experiments,\
sut_name=${SUT} \
, get accuracy_report
```
<details><summary><code>top1=75.936%</code></summary><pre>
accuracy=75.936%, good=37968, total=50000
</pre></details>

#### Performance
##### Quick Run (10 seconds)
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=image_classification,model_name=resnet50,\
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
task=image_classification,model_name=resnet50,\
loadgen_mode=PerformanceOnly,\
loadgen_scenario=Offline,\
loadgen_target_qps=${OFFLINE_TARGET_QPS},\
collection_name=experiments,\
sut_name=${SUT} \
, get performance
```
