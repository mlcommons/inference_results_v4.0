# MLPerf Stable diffusion Inference on EMR

## Download data & FP32 model

If your servers are working behind network proxy, please ensure docker proxy and the following environment parameters are well configured:
* http_proxy
* HTTP_PROXY
* https_proxy
* HTTPS_PROXY
* no_proxy
* NO_PROXY
* ftp_proxy
* FTP_PROXY
* socks_proxy
* SOCKS_PROXY

### Download Model
```
sudo -v ; curl https://rclone.org/install.sh | sudo bash

rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com

rclone copy mlc-inference:mlcommons-inference-wg-public/stable_diffusion_fp32 ./stable_diffusion_fp32 -P

export MODEL_PATH=./stable_diffusion_fp32
```

### Download captions
```
cd {THIS_DIR}
mkdir -p coco2014/captions
wget -P coco2014/captions/ https://raw.githubusercontent.com/mlcommons/inference/master/text_to_image/coco2014/captions/captions_source.tsv 

```

### Download latent
```
cd {THIS_DIR}
mkdir -p coco2014/latents
wget -P coco2014/latents/ https://github.com/mlcommons/inference/raw/master/text_to_image/tools/latents.pt

```

## Setup Instructions (Conda)
### Download & Install Miniconda3 (If not installed already)

```

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
export PATH=~/miniconda3/bin:$PATH

```

### Setup and Activate Conda Environment
```
source /opt/rh/gcc-toolset-11/enable

bash prepare_env.sh

conda activate sdxl_cpu_bf16
```

## Setup Instructions (Docker)
### Build Docker Image
```
cd docker
bash build_sdxl_container.sh
cd ..
```
### Run Docker Container
```
docker run -e MODEL_PATH=${MODEL_PATH} --name sdxl-cpu-bf16 --privileged -v ${MODEL_PATH}:/opt/workdir/code/stable_diffusion/pytorch-cpu/stable_diffusion_fp32 -v /opt/rh/gcc-toolset-11:/opt/rh/gcc-toolset-11 -itd --net=host --ipc=host mlperf_inference_sdxl_bf16:4.0
docker exec -it sdxl-cpu-bf16 bash
cd code/stable_diffusion/pytorch-cpu/
```


## Run Benchmark - Performance
### Set parameters
Update user.conf inside configs/ with target_qps for both Offline and Server scenario
### Offline
```
bash run_offline.sh
```
### Server - Performance
```
bash run_server.sh
```

## Run Benchmark - Accuracy
### Offline
```
bash run_offline_accuracy.sh
```
### Server - Performance
```
bash run_server_accuracy.sh
```
