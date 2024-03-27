
# MLPerf Inference v4.0 - closed - Qualcomm

To run experiments individually, use the following commands.

## r282_q2_ultra_pp - bert-99 - singlestream

### Accuracy  

```
axs byquery loadgen_output,task=bert,sut_name=r282_q2_ultra_pp,model_name=bert-99,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=SingleStream
```

### Performance 

```
axs byquery loadgen_output,task=bert,sut_name=r282_q2_ultra_pp,model_name=bert-99,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=SingleStream,loadgen_target_latency=6.8
```

### Compliance TEST01

```
axs byquery loadgen_output,task=bert,sut_name=r282_q2_ultra_pp,model_name=bert-99,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=SingleStream,loadgen_target_latency=6.8
```

### Compliance TEST05

```
axs byquery loadgen_output,task=bert,sut_name=r282_q2_ultra_pp,model_name=bert-99,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=SingleStream,loadgen_target_latency=6.8
```

