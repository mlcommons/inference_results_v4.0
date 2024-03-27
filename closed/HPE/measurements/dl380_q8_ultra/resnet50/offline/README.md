
# MLPerf Inference v4.0 - closed - HPE

To run experiments individually, use the following commands.

## dl380_q8_ultra - resnet50 - offline

### Accuracy  

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=dl380_q8_ultra,model_name=resnet50,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline
```

### Performance 

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=dl380_q8_ultra,model_name=resnet50,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Offline,loadgen_target_qps=390000
```

### Compliance TEST01

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=dl380_q8_ultra,model_name=resnet50,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Offline,loadgen_target_qps=390000
```

### Compliance TEST04

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=dl380_q8_ultra,model_name=resnet50,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,loadgen_scenario=Offline,loadgen_target_qps=390000
```

### Compliance TEST05

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=dl380_q8_ultra,model_name=resnet50,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Offline,loadgen_target_qps=390000
```

