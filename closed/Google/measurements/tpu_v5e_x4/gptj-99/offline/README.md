
# MLPerf Inference v4.0 - closed - Google_Krai

To run experiments individually, use the following commands.

## tpu_v5e_x4 - gptj-99 - offline

### Accuracy  

```
axs byquery loadgen_output,task=gptj,framework=saxml,loadgen_dataset_size=13368,tokenizer_path=EleutherAI/gpt-j-6B,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_target_qps=10.10,loadgen_buffer_size=13368,collection_name=experiments_final3,sut_name=tpu_v5e_x4,mlperf_model_name=gptj-99,sax_admin_server_storage_bucket=sax_admin_server_storage_bucket
```

### Performance 

```
axs byquery loadgen_output,task=gptj,framework=saxml,loadgen_dataset_size=13368,tokenizer_path=EleutherAI/gpt-j-6B,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_target_qps=10.10,loadgen_buffer_size=13368,collection_name=experiments_final3,sut_name=tpu_v5e_x4,mlperf_model_name=gptj-99,sax_admin_server_storage_bucket=sax_admin_server_storage_bucket
```

