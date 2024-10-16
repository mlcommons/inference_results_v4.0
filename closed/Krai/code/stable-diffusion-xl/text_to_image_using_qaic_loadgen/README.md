# MLPerf Inference - Qualcomm Cloud AI - Stable Diffusion XL

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
export SUT=g292_q16_pro
```

as well as the `OFFLINE_TARGET_QPS` variable for the expected queries per
second (QPS).

## Stable Diffusion XL

### Offline

#### Accuracy (5,000 images)
```
axs byquery loadgen_output,task=text_to_image,sut_name=${SUT},model_name=stable-diffusion-xl,framework=torch,device=qaic,dtype=fp16,loadgen_dataset_size=5000,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline
```

##### FID/CLIP scores
```
axs byquery loadgen_output,task=text_to_image,sut_name=${SUT},model_name=stable-diffusion-xl,framework=torch,device=qaic,dtype=fp16,loadgen_dataset_size=5000,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline , get accuracy_dict
```
<pre>{'FID_SCORE': 23.066470965432927, 'CLIP_SCORE': 31.794263212680818}</pre>

##### FID score only
```
axs byquery loadgen_output,task=text_to_image,sut_name=${SUT},model_name=stable-diffusion-xl,framework=torch,device=qaic,dtype=fp16,loadgen_dataset_size=5000,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline , get fid
```
<pre>23.066470965432927</pre>

##### CLIP score only
```
axs byquery loadgen_output,task=text_to_image,sut_name=${SUT},model_name=stable-diffusion-xl,framework=torch,device=qaic,dtype=fp16,loadgen_dataset_size=5000,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline  , get clip
```
<pre>31.794263212680818</pre>

#### Performance
```
axs byquery loadgen_output,task=text_to_image,sut_name=${SUT},model_name=stable-diffusion-xl,framework=torch,device=qaic,dtype=fp16,loadgen_dataset_size=5000,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Offline
```

##### Samples per second
```
axs byquery loadgen_output,task=text_to_image,sut_name=${SUT},model_name=stable-diffusion-xl,framework=torch,device=qaic,dtype=fp16,loadgen_dataset_size=5000,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Offline , get performance
```
<pre>VALID : Samples_per_second=1.18</pre>
