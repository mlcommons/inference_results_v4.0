
# MLPerf Inference v4.0 - closed - Qualcomm

To run experiments individually, use the following commands.

## r282_q2_ultra_pp - retinanet - multistream

### Accuracy  

```
axs byquery loadgen_output,task=object_detection,sut_name=r282_q2_ultra_pp,model_name=retinanet,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=MultiStream
```

### Performance 

```
axs byquery loadgen_output,task=object_detection,sut_name=r282_q2_ultra_pp,model_name=retinanet,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=MultiStream,loadgen_target_latency=11.5
```

### Compliance TEST01

```
axs byquery loadgen_output,task=object_detection,sut_name=r282_q2_ultra_pp,model_name=retinanet,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=MultiStream,loadgen_target_latency=11.5
```

### Compliance TEST05

```
axs byquery loadgen_output,task=object_detection,sut_name=r282_q2_ultra_pp,model_name=retinanet,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=MultiStream,loadgen_target_latency=11.5
```

