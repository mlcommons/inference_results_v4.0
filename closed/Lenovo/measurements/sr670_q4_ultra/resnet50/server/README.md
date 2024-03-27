
# MLPerf Inference v4.0 - closed - Lenovo

To run experiments individually, use the following commands.

## sr670_q4_ultra - resnet50 - server

### Accuracy  

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=sr670_q4_ultra,model_name=resnet50,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=Server
```

### Performance 

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=sr670_q4_ultra,model_name=resnet50,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Server,vc:=3:1:3:3,loadgen_target_qps=210000
```

### Compliance TEST01

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=sr670_q4_ultra,model_name=resnet50,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Server,vc:=3:1:3:3,loadgen_target_qps=210000
```

### Compliance TEST04

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=sr670_q4_ultra,model_name=resnet50,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,loadgen_scenario=Server,vc:=3:1:3:3,loadgen_target_qps=210000
```

### Compliance TEST05

```
axs byquery loadgen_output,framework=kilt,task=image_classification,sut_name=sr670_q4_ultra,model_name=resnet50,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Server,vc:=3:1:3:3,loadgen_target_qps=210000
```

