# MLPerf Inference - Bert - KILT

To run the experiments you need the following commands

## Benchmarking bert-99 model in Performance mode
```
axs byquery loadgen_output,framework=kilt,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,model_name=bert-99,loadgen_dataset_size=10833,loadgen_buffer_size=10833,loadgen_compiance_test-,sut_name=r282_q8_pro_edge,loadgen_target_qps=6300
```

