
# MLPerf Inference v4.0 - closed - Qualcomm

To run experiments individually, use the following commands.

## g293_q16_ultra_ee - bert-99 - server

### Accuracy  

```
axs byquery loadgen_output,task=bert,sut_name=g293_q16_ultra_ee,model_name=bert-99,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=Server
```

### Power 

```
axs byquery power_loadgen_output,task=bert,sut_name=g293_q16_ultra_ee,model_name=bert-99,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Server
```

### Compliance TEST01

```
axs byquery loadgen_output,task=bert,sut_name=g293_q16_ultra_ee,model_name=bert-99,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Server,loadgen_target_qps=26500
```

### Compliance TEST05

```
axs byquery loadgen_output,task=bert,sut_name=g293_q16_ultra_ee,model_name=bert-99,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Server,loadgen_target_qps=26500
```

