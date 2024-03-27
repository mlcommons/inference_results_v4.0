
# MLPerf Inference v4.0 - closed - Krai

To run experiments individually, use the following commands.

## g292_q16_pro_pp - stable-diffusion-xl - server

### Accuracy  

```
axs byquery loadgen_output,task=text_to_image,sut_name=g292_q16_pro_pp,model_name=stable-diffusion-xl,framework=torch,device=qaic,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,setting_fan-,fan_rpm-,vc_set-,timestamp
```

### Performance 

```
axs byquery loadgen_output,task=text_to_image,sut_name=g292_q16_pro_pp,model_name=stable-diffusion-xl,framework=torch,device=qaic,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,setting_fan-,fan_rpm-,vc=17,loadgen_target_qps=0.95,loadgen_count_override_min=2000
```

