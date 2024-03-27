
# MLPerf Inference v4.0 - closed - Dell

To run experiments individually, use the following commands.

## r760_q4_ultra - retinanet - server

### Accuracy  

```
axs byquery loadgen_output,framework=kilt,task=object_detection,sut_name=r760_q4_ultra,model_name=retinanet,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=Server
```

### Performance 

```
axs byquery loadgen_output,framework=kilt,task=object_detection,sut_name=r760_q4_ultra,model_name=retinanet,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Server,vc:=1:3:3:3:1:3:3:3:1:3:3:3:1:3:3:3,loadgen_target_qps=3125
```

### Compliance TEST01

```
axs byquery loadgen_output,framework=kilt,task=object_detection,sut_name=r760_q4_ultra,model_name=retinanet,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Server,vc:=1:3:3:3:1:3:3:3:1:3:3:3:1:3:3:3,loadgen_target_qps=3125
```

### Compliance TEST05

```
axs byquery loadgen_output,framework=kilt,task=object_detection,sut_name=r760_q4_ultra,model_name=retinanet,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Server,vc:=1:3:3:3:1:3:3:3:1:3:3:3:1:3:3:3,loadgen_target_qps=3125
```

