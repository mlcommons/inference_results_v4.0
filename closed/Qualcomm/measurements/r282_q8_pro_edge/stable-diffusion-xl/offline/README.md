
# MLPerf Inference v4.0 - closed - Qualcomm

To run experiments individually, use the following commands.

## r282_q8_pro_edge - stable-diffusion-xl - offline

### Accuracy  

```
axs byquery loadgen_output,task=text_to_image,sut_name=r282_q8_pro_edge,model_name=stable-diffusion-xl,framework=torch,device=qaic,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline
```

### Power 

```
axs byquery power_loadgen_output,task=text_to_image,sut_name=r282_q8_pro_edge,model_name=stable-diffusion-xl,framework=torch,device=qaic,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline
```

