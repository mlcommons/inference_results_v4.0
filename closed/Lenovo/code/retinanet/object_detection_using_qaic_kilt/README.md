# MLPerf Inference - Qualcomm Cloud AI - RetinaNet

This implementation supports the MLPerf Inference RetinaNet workload on systems
equipped with Qualcomm AI Cloud (QAIC) accelerators using KRAI Inference
Library Technology (KILT).

Follow [instructions](https://github.com/krai/axs2qaic-docker) to set up the
Docker environment.

Below are example commands for benchmarking RetinaNet under the Offline
scenario which is common to the Edge and Datacenter categories.

In the Docker container, define the `SUT` variable for your system-under-test
(SUT) e.g.:

```
export SUT=g293_q16_ultra
```

as well as the `OFFLINE_TARGET_QPS` variable for the expected queries per
second (QPS).

## RetinaNet

### Offline

#### [Optional] Compile model
```
axs byquery kilt_ready,device=qaic,sut_name=${SUT},\
model_name=retinanet,index_file=openimages_cal_images_list.txt,\
loadgen_scenario=Offline
```

#### [Optional] Compile program
```
axs byquery compiled,lib_kilt --- , remove && \
axs byquery compiled,kilt_executable,device=qaic,retinanet --- , remove && \
axs byquery compiled,lib_kilt && \
axs byquery compiled,kilt_executable,device=qaic,retinanet
```

#### Accuracy
##### Quick Run (500 images)
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=object_detection,model_name=retinanet,\
loadgen_mode=AccuracyOnly,loadgen_dataset_size=500,\
loadgen_scenario=Offline,\
collection_name=experiments,\
sut_name=${SUT} \
, get accuracy_report
```

##### Full Run (24,781 images)
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=object_detection,model_name=retinanet,\
loadgen_mode=AccuracyOnly,\
loadgen_scenario=Offline,\
collection_name=experiments,\
sut_name=${SUT} \
, get accuracy_report
```

#### Performance
##### Quick Run (10 seconds)
```
axs byquery loadgen_output,\
framework=kilt,device=qaic,\
task=object_detection,model_name=retinanet,\
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
task=object_detection,model_name=retinanet,\
loadgen_mode=PerformanceOnly,\
loadgen_scenario=Offline,\
loadgen_target_qps=${OFFLINE_TARGET_QPS},\
collection_name=experiments,\
sut_name=${SUT} \
, get performance
```
