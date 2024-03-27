
# MLPerf Inference v4.0 - closed - Dell

To run experiments individually, use the following commands.

## r760_q4_ultra - resnet50 - server

### Accuracy  

```
axs byquery loadgen_output,task=image_classification,sut_name=r760_q4_ultra,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,vc:=1:3:3:3:1:3:3:3:1:3:3:3:1:3:3:3
```

### Performance 

```
axs byquery loadgen_output,task=image_classification,sut_name=r760_q4_ultra,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Server,vc:=1:3:3:3:1:3:3:3:1:3:3:3:1:3:3:3,loadgen_target_qps=215000
```

### Compliance TEST01

```
axs byquery loadgen_output,task=image_classification,sut_name=r760_q4_ultra,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Server,vc:=1:3:3:3:1:3:3:3:1:3:3:3:1:3:3:3,loadgen_target_qps=215000
```

### Compliance TEST04

```
axs byquery loadgen_output,task=image_classification,sut_name=r760_q4_ultra,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,loadgen_scenario=Server,vc:=1:3:3:3:1:3:3:3:1:3:3:3:1:3:3:3,loadgen_target_qps=215000
```

### Compliance TEST05

```
axs byquery loadgen_output,task=image_classification,sut_name=r760_q4_ultra,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Server,vc:=1:3:3:3:1:3:3:3:1:3:3:3:1:3:3:3,loadgen_target_qps=215000
```

