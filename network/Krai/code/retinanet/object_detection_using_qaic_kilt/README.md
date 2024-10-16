# MLPerf Inference - Object Detection Models - KILT
This implementation runs language models with the KILT backend.

Currently it supports the following model:
- retinanet

# Setup Global Variables
```
export BENCHMARK=retinanet
export WORKDIR=/local/mnt/workspace
```

# Setup Docker Images
Please follow instructions [here](https://github.com/krai/axs2qaic-docker) to build the relevant docker images.

# Setup Local Environment
Install axs in `$WORKDIR/$USER`.
```
git clone --branch master https://github.com/krai/axs $WORKDIR/$USER/axs
```

```
echo "

# AXS
export PATH='$PATH:$WORKDIR/$USER/axs'
export AXS_WORK_COLLECTION='$WORKDIR/$USER/work_collection' 

" >> ~/.bashrc
```

```
source ~/.bashrc
```

Upon sucessful installation.
```
user@laptop:~/axs$ axs
DefaultKernel{}
```

Import these repos into your work_collection using HTTPS.
```
axs byquery git_repo,collection,repo_name=axs2qaic
axs byquery git_repo,collection,repo_name=axs2kilt
axs byquery git_repo,collection,repo_name=axs2config
axs byquery git_repo,collection,repo_name=axs2system
axs byquery git_repo,repo_name=kilt-mlperf
```

Set server port number and server ip address.
```
export SERVER_PORT_NUMBER=<server_port_number>
export SERVER_IP_ADDRESS=<server_ip_address>
export SDK_VER=<sdk_ver>
```

Obtain the latest image name. For example:
```
elim@dyson:~/axs2qaic-docker-dev$ docker image ls
REPOSITORY                  TAG                   IMAGE ID       CREATED        SIZE
krai/mlperf.retinanet   deb_${SDK_VER}     5b8592b6fbf0    2 hours ago    81.7GB
```

Set image name.
```
export IMAGE_NAME=krai/mlperf.${BENCHMARK}:deb_{SDK_VER}
```

Launch docker container
```
mkdir -p ${WORKDIR}/${USER}/axs_experiment_${BENCHMARK} && \
docker run -it -p ${SERVER_PORT_NUMBER}:${SERVER_PORT_NUMBER} --name ${USER}_${BENCHMARK} --privileged --group-add $(getent group qaic | cut -d: -f3) -v ${WORKDIR}/${USER}/axs_experiment_${BENCHMARK}:/home/krai/experiment ${IMAGE_NAME}
```

# Run Experiment inside Docker
Use one of the assembled SUT in [here](https://github.com/krai/axs2config), or you can customize your own. For instance, if you would like to benchmark dl385_q8_std:
```
export SUT=dl385_q8_std
```

### Retinanet
#### Server side

Compile the model for `retinanet_server`
```
axs byquery compiled,device=qaic,kilt_executable,retinanet_server
```
Launch the Server program (Scenario: `Server`)
```
axs byquery loadgen_output,object_detection,device=qaic,framework=kilt,model_name=retinanet,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,sut_name=${SUT},flavour=retinanet_server,network_server_port=${SERVER_PORT_NUMBER},network_num_sockets=8,recommended_batch_size=200,kilt_unique_server_id=KILT_Network_SUT
```

#### Client side

Compile the model for `retinanet_client`
```
axs byquery compiled,device=qaic,kilt_executable,retinanet_client
```

Measure Accuracy (Full Run)
```
axs byquery loadgen_output,object_detection,device=qaic,framework=kilt,model_name=retinanet,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,sut_name=${SUT},flavour=retinanet_client,fan=null,setting_fan=null,fan_rpm=null,vc=null,actual_vc_dec=null,vc_set=null,network_server_port=${SERVER_PORT_NUMBER},network_server_ip_address=${SERVER_IP_ADDRESS},network_num_sockets=8 , get accuracy
```

Measure Performance (Full Run)
```
axs byquery loadgen_output,object_detection,device=qaic,framework=kilt,model_name=retinanet,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,sut_name=${SUT},flavour=retinanet_client,fan=null,setting_fan=null,fan_rpm=null,vc=null,actual_vc_dec=null,vc_set=null,network_server_port=${SERVER_PORT_NUMBER},network_server_ip_address=${SERVER_IP_ADDRESS},network_num_sockets=8,loadgen_target_qps=<measured_target_qps> , parse_summary
```


