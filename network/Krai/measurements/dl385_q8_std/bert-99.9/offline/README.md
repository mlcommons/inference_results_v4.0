
# MLPerf Inference v4.0 - closed - Krai

To run experiments individually, use the following commands.

## dl385_q8_std - bert-99.9 - offline

### Accuracy  

```
axs byquery loadgen_output,task=bert,device=qaic,framework=kilt,model_name=bert-99.9,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,sut_name=dl385_q8_std,flavour=bert_client,fan=null,setting_fan=null,fan_rpm=null,vc=null,actual_vc_dec=null,vc_set=null,network_server_port=7276,network_server_ip_address=192.168.0.2,network_num_sockets=8
```

### Performance 

```
axs byquery loadgen_output,task=bert,device=qaic,framework=kilt,model_name=bert-99.9,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,sut_name=dl385_q8_std,flavour=bert_client,fan=null,setting_fan=null,fan_rpm=null,vc=null,actual_vc_dec=null,vc_set=null,network_server_port=7276,network_server_ip_address=192.168.0.2,network_num_sockets=8,loadgen_target_qps=2900,recommended_batch_size=100
```

### Compliance TEST01

```
axs byquery loadgen_output,task=bert,device=qaic,framework=kilt,model_name=bert-99.9,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,sut_name=dl385_q8_std,flavour=bert_client,fan=null,setting_fan=null,fan_rpm=null,vc=null,actual_vc_dec=null,vc_set=null,network_server_port=7276,network_server_ip_address=192.168.0.2,network_num_sockets=8,loadgen_target_qps=2900,recommended_batch_size=100,loadgen_compliance_test=TEST01
```

### Compliance TEST05

```
axs byquery loadgen_output,task=bert,device=qaic,framework=kilt,model_name=bert-99.9,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,sut_name=dl385_q8_std,flavour=bert_client,fan=null,setting_fan=null,fan_rpm=null,vc=null,actual_vc_dec=null,vc_set=null,network_server_port=7276,network_server_ip_address=192.168.0.2,network_num_sockets=8,loadgen_target_qps=2900,recommended_batch_size=100,loadgen_compliance_test=TEST05
```

