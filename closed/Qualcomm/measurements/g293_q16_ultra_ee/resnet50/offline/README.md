
# MLPerf Inference v4.0 - closed - Qualcomm

To run experiments individually, use the following commands.

## g293_q16_ultra_ee - resnet50 - offline

### Accuracy  

```
axs byquery loadgen_output,task=image_classification,sut_name=g293_q16_ultra_ee,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline
```

### Power 

```
axs byquery power_loadgen_output,task=image_classification,sut_name=g293_q16_ultra_ee,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Offline
```

### Compliance TEST01

```
axs byquery loadgen_output,task=image_classification,sut_name=g293_q16_ultra_ee,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Offline,loadgen_target_qps=880000
```

### Compliance TEST04

```
axs byquery loadgen_output,task=image_classification,sut_name=g293_q16_ultra_ee,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,loadgen_scenario=Offline,loadgen_target_qps=880000
```

### Compliance TEST05

```
axs byquery loadgen_output,task=image_classification,sut_name=g293_q16_ultra_ee,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Offline,loadgen_target_qps=880000
```

