
# MLPerf Inference v4.0 - closed - Qualcomm

To run experiments individually, use the following commands.

## r282_q2_ultra_pp - stable-diffusion-xl - singlestream

### Accuracy  

```
axs byquery loadgen_output,task=text_to_image,sut_name=r282_q2_ultra_pp,model_name=stable-diffusion-xl,framework=torch,device=qaic,loadgen_mode=AccuracyOnly,loadgen_scenario=SingleStream,vc=1,device_id=26+27,setting_fan-,fan_rpm-
```

### Performance 

```
axs byquery loadgen_output,task=text_to_image,sut_name=r282_q2_ultra_pp,model_name=stable-diffusion-xl,framework=torch,device=qaic,loadgen_mode=PerformanceOnly,loadgen_scenario=SingleStream,vc=1,device_id=26+27,setting_fan-,fan_rpm-,loadgen_target_latency=12.0
```

