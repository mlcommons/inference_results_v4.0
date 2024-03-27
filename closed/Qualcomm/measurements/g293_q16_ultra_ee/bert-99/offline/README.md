
# MLPerf Inference v4.0 - closed - Qualcomm

To run experiments individually, use the following commands.

## g293_q16_ultra_ee - bert-99 - offline

### Accuracy  

```
axs byquery loadgen_output,task=bert,sut_name=g293_q16_ultra_ee,model_name=bert-99,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline
```

### Power 

```
axs byquery power_loadgen_output,task=bert,sut_name=g293_q16_ultra_ee,model_name=bert-99,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Offline,vc=5
```

### Compliance TEST01

```
axs byquery loadgen_output,task=bert,sut_name=g293_q16_ultra_ee,model_name=bert-99,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Offline,loadgen_target_qps=28000
```

### Compliance TEST05

```
axs byquery loadgen_output,task=bert,sut_name=g293_q16_ultra_ee,model_name=bert-99,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Offline,loadgen_target_qps=28000
```

