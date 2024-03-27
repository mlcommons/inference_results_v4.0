
# MLPerf Inference v4.0 - closed - Qualcomm

To run experiments individually, use the following commands.

## g293_q16_ultra_pp - resnet50 - server

### Accuracy  

```
axs byquery loadgen_output,task=image_classification,sut_name=g293_q16_ultra_pp,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,vc_set-
```

### Performance 

```
axs byquery loadgen_output,task=image_classification,sut_name=g293_q16_ultra_pp,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Server,vc_set-,loadgen_target_qps=625000
```

### Compliance TEST01

```
axs byquery loadgen_output,task=image_classification,sut_name=g293_q16_ultra_pp,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Server,vc_set-,loadgen_target_qps=575000
```

### Compliance TEST04

```
axs byquery loadgen_output,task=image_classification,sut_name=g293_q16_ultra_pp,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,loadgen_scenario=Server,vc_set-,loadgen_target_qps=575000
```

### Compliance TEST05

```
axs byquery loadgen_output,task=image_classification,sut_name=g293_q16_ultra_pp,model_name=resnet50,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Server,vc_set-,loadgen_target_qps=600000
```

