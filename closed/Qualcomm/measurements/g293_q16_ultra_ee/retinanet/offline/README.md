
# MLPerf Inference v4.0 - closed - Qualcomm

To run experiments individually, use the following commands.

## g293_q16_ultra_ee - retinanet - offline

### Accuracy  

```
axs byquery loadgen_output,task=object_detection,sut_name=g293_q16_ultra_ee,model_name=retinanet,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline
```

### Power 

```
axs byquery power_loadgen_output,task=object_detection,sut_name=g293_q16_ultra_ee,model_name=retinanet,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test-,loadgen_scenario=Offline,vc=5
```

### Compliance TEST01

```
axs byquery loadgen_output,task=object_detection,sut_name=g293_q16_ultra_ee,model_name=retinanet,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,loadgen_scenario=Offline,loadgen_target_qps=15000
```

### Compliance TEST05

```
axs byquery loadgen_output,task=object_detection,sut_name=g293_q16_ultra_ee,model_name=retinanet,framework=kilt,device=qaic,collection_name=experiments_submission,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST05,loadgen_scenario=Offline,loadgen_target_qps=15000
```

