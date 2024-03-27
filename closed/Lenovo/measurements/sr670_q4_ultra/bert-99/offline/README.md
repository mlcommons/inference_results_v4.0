
# MLPerf Inference v4.0 - closed - Lenovo

To run experiments individually, use the following commands.

## sr670_q4_ultra - bert-99 - offline

### Accuracy  

```
axs byquery loadgen_output,framework=kilt,task=bert,sut_name=sr670_q4_ultra,model_name=bert-99,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,vc:=3:1:3:3
```

### Performance 

```
axs byquery loadgen_output,framework=kilt,task=bert,sut_name=sr670_q4_ultra,model_name=bert-99,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Offline,vc:=3:1:3:3,loadgen_target_qps=7500
```

### Compliance TEST01

```
axs byquery loadgen_output,framework=kilt,task=bert,sut_name=sr670_q4_ultra,model_name=bert-99,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Offline,vc:=3:1:3:3,loadgen_target_qps=7500
```

### Compliance TEST05

```
axs byquery loadgen_output,framework=kilt,task=bert,sut_name=sr670_q4_ultra,model_name=bert-99,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Offline,vc:=3:1:3:3,loadgen_target_qps=7500
```

