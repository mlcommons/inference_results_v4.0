# MLPerf Inference - Object Detection - KILT

To run the experiments you need the following commands

## Benchmarking retinanet model in Performance mode
```
axs byquery loadgen_output,framework=kilt,loadgen_scenario=MultiStream,loadgen_mode=PerformanceOnly,model_name=retinanet,loadgen_dataset_size=24781,loadgen_buffer_size=64,loadgen_compiance_test-,sut_name=r282_q2_ultra_ee,loadgen_multistreamness=None,loadgen_target_latency=11.5
```

