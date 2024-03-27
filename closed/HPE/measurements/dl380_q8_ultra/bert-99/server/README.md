
# MLPerf Inference v4.0 - closed - HPE

To run experiments individually, use the following commands.

## dl380_q8_ultra - bert-99 - server

### Accuracy  

```
axs byquery loadgen_output,framework=kilt,task=bert,sut_name=dl380_q8_ultra,model_name=bert-99,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=Server
```

### Performance 

```
axs byquery loadgen_output,framework=kilt,task=bert,sut_name=dl380_q8_ultra,model_name=bert-99,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Server,loadgen_target_qps=12875
```

### Compliance TEST01

```
axs byquery loadgen_output,framework=kilt,task=bert,sut_name=dl380_q8_ultra,model_name=bert-99,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Server,loadgen_target_qps=12500
```

### Compliance TEST05

```
axs byquery loadgen_output,framework=kilt,task=bert,sut_name=dl380_q8_ultra,model_name=bert-99,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Server,loadgen_target_qps=12500
```

