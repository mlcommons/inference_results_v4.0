
# MLPerf Inference v4.0 - closed - Qualcomm

To run experiments individually, use the following commands.

## g293_q16_ultra - retinanet - server

### Accuracy  

```
axs byquery loadgen_output,task=object_detection,sut_name=g293_q16_ultra,model_name=retinanet,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=Server
```

### Performance 

```
axs byquery loadgen_output,task=object_detection,sut_name=g293_q16_ultra,model_name=retinanet,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Server,loadgen_target_qps=13750
```

### Compliance TEST01

```
axs byquery loadgen_output,task=object_detection,sut_name=g293_q16_ultra,model_name=retinanet,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Server,loadgen_target_qps=13750
```

### Compliance TEST05

```
axs byquery loadgen_output,task=object_detection,sut_name=g293_q16_ultra,model_name=retinanet,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Server,loadgen_target_qps=13060
```

