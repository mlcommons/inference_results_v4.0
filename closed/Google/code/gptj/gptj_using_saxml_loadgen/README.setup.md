# Set up Google Cloud TPU instances for MLPerf Inference benchmarking

## Set up `gcloud`
```
sudo apt-get install apt-transport-https ca-certificates gnupg curl sudo
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli
gcloud auth login --no-launch-browser
```
## Define environment variables
```
export PROJECT_ID=<your project id>
export ZONE=<your zone e.g. us-west4-a>
export SERVICE_ACCOUNT=<your service account>
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
export ACCELERATOR_TYPE=v5litepod-4
export QUOTA_TYPE=
export QUEUED_RESOURCE_ID=mlperf
export TPU_NAME=mlperf
```
## Create a VM
```
gcloud alpha compute tpus queued-resources create ${QUEUED_RESOURCE_ID} \
--node-id ${TPU_NAME} \
--zone ${ZONE} \
--project ${PROJECT_ID} \
--accelerator-type ${ACCELERATOR_TYPE} \
--runtime-version ${RUNTIME_VERSION} \
--service-account ${SERVICE_ACCOUNT}
```
## Create a storage bucket for the admin server
```
export PROJECT_ID=<your project id>
export SAX_ADMIN_SERVER_STORAGE_BUCKET="sax_admin_server_storage_bucket"
gcloud storage buckets create gs://${SAX_ADMIN_SERVER_STORAGE_BUCKET} --project=${PROJECT_ID}
```
## Create a storage bucket for the model server
```
export PROJECT_ID=<your project id>
export SAX_MODEL_SERVER_STORAGE_BUCKET="sax_model_server_storage_bucket"
gcloud storage buckets create gs://${SAX_MODEL_SERVER_STORAGE_BUCKET} --project=${PROJECT_ID}
```
## Connect to the VM
```
export TPU_NAME=mlperf
export ZONE=<your zone e.g. us-west4-a>
export PROJECT_ID=<your project id>
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID}
```
## Update
```
sudo apt update -y
sudo apt upgrade -y
sudo apt install \
  git vim htop tree net-tools \
  libssl-dev openssl libffi-dev libbz2-dev \
  zlib1g zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev
```

# Set up a model checkpoint
**NB:** It is recommended to perform the following model checkpoint conversion and quantization steps on a separate machine (not necessarily a TPU instance) as they require a lot of disk space.

## Set up axs
### Kernel
```
git clone --branch stable https://github.com/krai/axs $HOME/axs
echo "export PATH='$PATH:$HOME/axs'" >> $HOME/.bashrc
source $HOME/.bashrc
```

### [Optional] Refresh state
```
axs byname work_collection , remove
```

### Get required repositories and packages
```
axs byquery git_repo,collection,repo_name=axs2mlperf
axs byquery git_repo,collection,repo_name=axs2gcp
axs byquery git_repo,repo_name=saxml_git,checkout=mlperf4.0
axs byquery git_repo,repo_name=mlperf_inference_git
axs byquery python_package,package_name==mlperf_loadgen,desired_python_version===3.10
axs byquery downloaded,dataset_name=cnndm_v3_0_0
```

### Install Saxml dependencies
```
cd $(axs byquery git_repo,repo_name=saxml_git,checkout=mlperf4.0 , get_path) 
sudo saxml/tools/init_cloud_vm.sh
```

### Install rclone
```
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```

### Configure rclone
```
rclone config create mlc-inference s3 provider=Cloudflare \
access_key_id=f65ba5eef400db161ea49967de89f47b \
secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b \
endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
```

## Convert fine-tuned PT FP32 checkpoint provided by MLCommons to a PAX checkpoint
```
axs byquery pax_model,task=gptj
```

## Quantize checkpoint
```
axs byquery quantized_pax_model,task=gptj
```

## Copy quantized checkpoint to the storage bucket
```
export SAX_MODEL_SERVER_STORAGE_BUCKET="sax_model_server_storage_bucket"
gsutil -m cp -r \
$(axs byquery quantized_pax_model,task=gptj , get_path)/* \
gs://${SAX_MODEL_SERVER_STORAGE_BUCKET}/gptj_sax_int8_converted/
```

# [Optional] Set up an SSD
It is recommended to set up an SSD and store your saxml_build directory there. Thus the process of building admin and model servers will take much less time.

## Define environment variables
```
export INSTANCE=""
export PROJECT_ID=<your project id>
export ZONE=<your zone e.g. us-west4-a>

export _DISK_NAME="ssd${INSTANCE}"
export _DISK_ZONE="${ZONE}"
export _DISK_SIZE="100G"
export _DISK_TYPE="pd-ssd"
echo "_DISK_NAME=${_DISK_NAME}"
echo "_DISK_ZONE=${_DISK_ZONE}"
echo "_DISK_SIZE=${_DISK_SIZE}"
echo "_DISK_TYPE=${_DISK_TYPE}"

export _TPU_NAME=mlperf
export _TPU_USER=${USER}
export _TPU_WORKER_NAME=$(gcloud compute tpus tpu-vm ssh ${_TPU_USER}@${_TPU_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --command="hostname")
```

## Create SSD
```
gcloud compute disks create ${_DISK_NAME} \
--project=${PROJECT_ID} \
--size ${_DISK_SIZE} \
--type ${_DISK_TYPE} \
--zone ${_DISK_ZONE};
```

## Attach SSD
```
gcloud alpha compute tpus tpu-vm attach-disk mlperf${INSTANCE}_0 \
--project=${PROJECT_ID} \
--zone=${ZONE} \
--disk=${_DISK_NAME} \
--mode=read-write
```

## Format and mount SSD
```
gcloud compute tpus tpu-vm ssh root@${_TPU_NAME} \
--project=${PROJECT_ID} \
--zone=${ZONE} \
--command="
set -ex;
lsblk;
mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb;
mkdir -p /mnt/disks/persist;
mount -o discard,defaults /dev/sdb /mnt/disks/persist;
chmod 777 /mnt/disks/persist;
lsblk;";
```