# Setup from Source
!! This guide is for develop purposes only and is not officially supported.

## HW and SW requirements
### 1. HW requirements
| HW  |      Configuration      |
| --  | ----------------------- |
| CPU | EMR @ 2 sockets/Node  |
| DDR | 512G/socket @ 3200 MT/s |
| SSD | 1 SSD/Node @ >= 1T      |

### 2. SW requirements
| SW       | Version |
|----------|---------|
| GCC      |  12.1.1 |
| Binutils | >= 2.35 |

## Steps to run DLRM

### 1. Prepare DLRM dataset and code
(1) Prepare DLRM dataset
```
   Create a directory (such as ${WORKDIR}\dataset\terabyte_input) which contain:
     day_23_dense.npy
     day_23_sparse_multi_hot.npz
     day_23_labels.npy

   About how to get the dataset, please refer to
      https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch
```
(2) Prepare pre-trained DLRM model
```
   cd python/model
   sudo yum install rclone -y
   rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com

  rclone copy mlc-inference:mlcommons-inference-wg-public/model_weights ./model_weights -P

   # unzip weights.zip
   cd ${WORKDIR}
   export MODEL_DIR=	      # path to model_weights for example ${WORKDIR}\dlrm_pytorch\python\model
   # dump model from snapshot to torch
   ./run_dump_torch_model.sh
```
(3) Calibration int8 model
```
   ./run_calibration.sh
```

# Setup with Docker

## Steps to run DLRM

### 1.Prepare dataset and model in host

The dataset and model of each workload need to be prepared in advance in the host

#### dataset：

about how to get the dataset please refer to https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch.
create a directory (such as dlrm\dataset\terabyte_input) which contain day_day_count.npz, day_fea_count.npz and terabyte_processed_test.bin#For calibration, need one of the following two files:terabyte_processed_val.bin or calibration.npz

#### model:

```
mkdir dlrm/dlrm_pytorch/python/model
cd dlrm/dlrm_pytorch/python/model
wget https://cloud.mlcommons.org/index.php/s/XzfSeLgW8FYfR3S/download -O dlrm_terabyte.pytorch
```

### 2.Start and log into a container

#### (1) start a container

-v mount the dataset to docker container 

```
docker run --privileged --name intel_dlrm -itd --net=host --ipc=host -v </path/to/dataset/and/model>:/root/mlperf_data/dlrm intel/intel-optimized-pytorch:v1.7.0-ipex-v1.2.0-dlrm
```

check container, it will show a container named intel_dlrm

```
docker ps
```

#### (2) into container

```
docker exec -it mlperf_dlrm bash
```

### 3.Run DLRM

#### (1) cd dlrm_pytorch

```
cd /opt/workdir/intel_inference_datacenter_v3-1/closed/Intel/code/dlrm/pytorch-cpu
```

#### (2) configure DATA_DIR and MODEL_DIR 

you can modify the setup_dataset.sh, then 'source ./setup_dataset.sh'

```
export DATA_DIR=    # the path of dataset, for example as ${WORKDIR}\dataset\terabyte_input
export MODEL_DIR=    # the path of pre-trained model, for example as ${WORKDIR}\dlrm_pytorch\python\model
```

#### (3) configure offline/server mode options 

currenlty used options for each mode is in setup_env_offline/server.sh, You can modify it, then 'source ./setup_env_offline/server.sh'

```
export NUM_SOCKETS=        # i.e. 8
export CPUS_PER_SOCKET=    # i.e. 28
export CPUS_PER_PROCESS=   # i.e. 14. which determine how many cores for one processe running on one socket
                           #   process_number = $CPUS_PER_SOCKET / $CPUS_PER_PROCESS
export CPUS_PER_INSTANCE=  # i.e. 14. which determine how many cores used for one instance inside one process
                           #   instance_number_per_process = $CPUS_PER_PROCESS / CPUS_PER_INSTANCE
                           #   total_instance_number_in_system = instance_number_per_process * process_number
```

#### (4) command line

Please update setup_env_server.sh and setup_env_offline.sh and user.conf according to your platform resource.

```
bash run_mlperf.sh --mode=<offline/server> --type=<perf/acc> --dtype=int8
```

for int8 calibration scripts, please look into Intel/calibration/dlrm/pytorch-cpu/ directory.

for int8 execution, calibration result is in int8_configure.json which is copied from that output.
