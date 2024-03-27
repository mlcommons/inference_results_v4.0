# MLPerf Inference Benchmarks for Medical Image 3D Segmentation
!! This guide is for develop purposes only and is not officially supported.

## Steps to run 3D-UNet


### 0. HW and SW requirements
```
  EMR 2 sockets
  GCC >= 11.2
```

### 1. Download Dataset
```
  cd ~
  mkdir -p mlperf_data/3dunet-kits
  cd mlperf_data/3dunet-kits
  git clone https://github.com/neheller/kits19
  cd kits19
  pip3 install -r requirements.txt
  python3 -m starter_code.get_imaging
  cd ~
```

### 2. Prepare enviroment with Docker
```
  cd docker
  bash build_3dunet_container.sh 
  docker run --name intel_3dunet --privileged -itd -v /data/mlperf_data:/root/mlperf_data --net=host --ipc=host mlperf_inference_3dunet:3.1
  docker ps -a #get container "id"
  docker exec -it <id> bash
  cd /opt/workdir/code/3d-unet-99.9/pytorch-cpu
  bash process_data_model.sh
  
``` 

### 3. Run command for accuracy and performance
```
  bash run.sh acc
  bash run.sh perf
```
