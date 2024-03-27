
# MLPerf Inference v4.0 - closed - Krai

To run experiments individually, use the following commands.

## g292_q16_pro_pp - stable-diffusion-xl - offline

### Accuracy  

```
axs byquery loadgen_output,task=text_to_image,sut_name=g292_q16_pro_pp,model_name=stable-diffusion-xl,framework=torch,device=qaic,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,setting_fan-,fan_rpm-,vc_set-
```

### Performance 

```
axs byquery loadgen_output,task=text_to_image,sut_name=g292_q16_pro_pp,model_name=stable-diffusion-xl,framework=torch,device=qaic,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,setting_fan-,fan_rpm-,vc=17,loadgen_min_duration_s=600,loadgen_target_qps=1.2
```

