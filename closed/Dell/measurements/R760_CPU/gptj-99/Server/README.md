# Get Started with Intel MLPerf v4.0 Submission with Intel Optimized Docker Images

MLPerf is a benchmark for measuring the performance of machine learning
systems. It provides a set of performance metrics for a variety of machine
learning tasks, including image classification, object detection, machine
translation, and others. The benchmark is representative of real-world
workloads and as a fair and useful way to compare the performance of different
machine learning systems.


In this document, we'll show how to run Intel MLPerf v4.0 submission with Intel
optimized Docker images.

## HW configuration:

| System Info     | Configuration detail                 |
| --------------- | ------------------------------------ |
| CPU             | EMR                                  |
| OS              | CentOS  Stream 8                     |
| Kernel          | 6.6.8-1.el8.elrepo.x86_64            | 
| Memory          | 1024GB (16x64GB 5600MT/s [5600MT/s]) |
| Disk            | 1TB NVMe                             |

## BIOS settings:
| BIOS setting    | Recommended value                    |
| --------------- | ------------------------------------ |
|Hyperthreading|Enabled
|Turbo Boost|Enabled
|Core Prefetchers|Hardware,Adjacent Cache,DCU Streamer,DCU IP
|LLC Prefetch|Disable
|CPU Power and Perf Policy|Performance
|NUMA-based Cluster|SNC2
|Hardware P State|Native (based on OS guidance)
|Energy Perf Bias|OS Controls EPB
|Energy Efficient Turbo|Disabled

## Prerequisite
We provides a kit to automate data ingestion, preprocessing, testing, log collection and submission checker. It requires the following software:
* python 3.9/3.10
* pip
* docker
* wget
* unzip
* rclone
* miniconda3/anaconda3

Note that the following approaches are supported for installing dependencies:
1. Install directly
2. Install in a conda environment
3. Install in a python virtual environment

For the option #2 and #3, you need to create the environment and activate it first before installation.

Install the dependencies by:
```bash
cd <THIS_REPO>/closed/Intel/code/automation
pip3 install -r requirements.txt
```

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

## Downloading data and models
Conda is required for downloading datasets and models. Please install Conda before proceeding. Once Conda is installed, you can download the datasets and models using the following commands:
```bash
model={resnet50,retinanet,rnnt,3d-unet,bert,gpt-j,dlrm_2,stable_diffusion,all} output_dir=<DATA_PATH> conda_path=<CONDA_ROOT_PATH> bash download_data.sh 
```
Parameters:
* model: specify a model name, `all` means all models.
* output_dir: the directory to persist data and model.
* conda_path: optional, ${HOME}/miniconda3 by default, root path of your conda, used by 3d-unet, retinanet and gpt-j only.
* dtype: optional, int8 by default, data preicision, currently unused

Download data for all models by using the following command:
```bash
model=all output_dir=<DATA_PATH> conda_path=<CONDA_ROOT_PATH> bash download_data.sh
```

For specific models, e.g. gpt-j, DLRMv2 and Stable Diffusion, Rclone is needed. If you don't have Rclone already installed, you can install it on Linux/MacOS with one simple command:
```bash
sudo -v ; curl https://rclone.org/install.sh | sudo bash  
```

For specific worload, e.g. gpt-j int4, use the following command:
```bash
model=gpt-j output_dir=<DATA_PATH> conda_path=<CONDA_ROOT_PATH> dtype=int4 bash download_data.sh  
```


## Launching test
Customize your test by the following parameters:
```bash
DATA_DIR=<output_dir of download_data.sh>
OUTPUT_DIR=<The directory to save logs and results>
SUFFIX=<suffix to avoid duplicate container names>
```
Create output diretory if it does not exist:
```bash
mkdir -p ${OUTPUT_DIR}
```
Launch complete benchmark by using the following commands:
| Benchmark              | Hardware    | Precision   | Command                                                                                                                                                 |
|:-----------------------|:------------|:------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3d-unet-99.9           | CPU         | int8        | `python3 run.py -n 3d-unet-99.9 -d ${DATA_DIR} -m ${DATA_DIR} -t ${OUTPUT_DIR} -x ${SUFFIX}`                                                            |
| bert-99                | CPU         | int8        | `python3 run.py -n bert-99 -d ${DATA_DIR}/bert/dataset -m ${DATA_DIR}/bert/model -t ${OUTPUT_DIR} -x ${SUFFIX}`                                         |
| dlrm-v2-99.9           | CPU         | int8        | `python3 run.py -n dlrm-v2-99.9 -i pytorch-cpu-int8 -d ${DATA_DIR}/dlrm_2/data -m ${DATA_DIR}/dlrm_2/model -t ${OUTPUT_DIR} -x ${SUFFIX}`             |
| gptj-99                | CPU         | int4        | `python3 run.py -n gptj-99 -d ${DATA_DIR}/gpt-j/data -m ${DATA_DIR}/gpt-j/data -t ${OUTPUT_DIR} -y int4 -x ${SUFFIX}`                                   |
| rnnt                   | CPU         | mix         | `python3 run.py -n rnnt -d ${DATA_DIR}/rnnt/mlperf-rnnt-librispeech -m ${DATA_DIR}/rnnt/mlperf-rnnt-librispeech -t ${OUTPUT_DIR} -y mix -x ${SUFFIX}`   |
| resnet50               | CPU         | int8        | `python3 run.py -n resnet50 -d ${DATA_DIR}/resnet50 -m ${DATA_DIR}/resnet50 -t ${OUTPUT_DIR} -x ${SUFFIX}`                                              |
| retinanet              | CPU         | int8        | `python3 run.py -n retinanet -d ${DATA_DIR}/retinanet/data -m ${DATA_DIR}/retinanet/data -t ${OUTPUT_DIR} -x ${SUFFIX}`                                 |

More options to customize your tests:
```
usage: run.py [-h] -n {rnnt,bert-99,3d-unet-99.9,resnet50,retinanet,dlrm-v2-99.9}
              [-i IMPLEMENTATION] [-y {int8,bf16,fp16,int4,fp32,mix}] -d DATASET -m MODEL_DIR
              -t OUTPUT -x CONTAINER_NAME_SUFFIX [-p] [-a] [-o] [-s] [-c] [-b] [-r] [-z]

options:
  -h, --help            show this help message and exit
  -n {rnnt,bert-99,3d-unet-99.9,resnet50,retinanet,dlrm-v2-99.9}, --model {rnnt,bert-99,3d-unet-99.9,resnet50,retinanet,dlrm-v2-99.9}
                        Benchmarking model
  -i IMPLEMENTATION, --implementation IMPLEMENTATION
                        Implementation id
  -y {int8,bf16,fp16,int4,fp32,mix}, --dtype {int8,bf16,fp16,int4,fp32,mix}
                        Precision
  -d DATASET, --dataset-dir DATASET_DIR
                        path of the datasets
  -m MODEL_DIR, --model-dir MODEL_DIR
                        path of the models
  -t OUTPUT, --output OUTPUT
                        path of the outputs
  -x CONTAINER_NAME_SUFFIX, --container-name-suffix CONTAINER_NAME_SUFFIX
                        The suffix of docker container name, used for avoiding name conflicts.
  -p, --performance-only
                        The option of running performance test only.
  -a, --accuracy-only   The option of running accuracy test only.
  -o, --offline-only    The option of running offline scenario only.
  -s, --server-only     The option of running server scenario only.
  -c, --compliance-only
                        The option of running compliance test only.
  -b, --skip-docker-build
                        The option of skipping building docker image.
  -u, --skip-create-container
                        The option of skipping docker build and container creation.  
  -r, --skip-data-preprocess
                        The option of skipping data preprocessing.
  -z, --ci-run
                        The option of running ci testings
```
To save time, rerun a benchmark directly within a running container without rebuilding the Docker image, creating the container and data preprocessing. For example, to rerun the bert-99 benchmark, use this command:
```bash
python3 run.py -n bert-99 -d ${DATA_DIR}/bert/dataset -m ${DATA_DIR}/bert/model -t ${OUTPUT_DIR} -x ${SUFFIX} -u -r
```

You can run a specific test of a benchmark in a specific mode. E.g. run bert-99 accuracy test at Offline mode by using the following command:
```bash
python3 run.py -n bert-99 -d ${DATA_DIR}/bert/dataset -m ${DATA_DIR}/bert/model -t ${OUTPUT_DIR} -x ${SUFFIX} -o -a
```

## Test outputs
* Runtime log: `mlperf.log` 
* Preprocessing log: `${OUTPUT_DIR}/preproc_<benchmark>_<implementation>_<precision>.log` 
* Results of Performance/Accuracy run: `${OUTPUT_DIR}/<division>/Intel/results/<system_desc_id>/<benchmark>/<scenario>` 
* Results of compliance test: `${OUTPUT_DIR}/<division>/Intel/compliance/<system_desc_id>/<benchmark>/<scenario>/<test_id>` 
* Measurments: `${OUTPUT_DIR}/<division>/Intel/measurements/<system_desc_id>/<benchmark>/<scenario>` 
* Runtime environment: `${OUTPUT_DIR}/env_<benchmark>_<impl>_<dtype>.log`

## Performing Submission Checker
1. Create `${OUTPUT_DIR}/<division>/<orgnization>/systems` directory, and within it, create the `<system_desc_id>.json` and `<system_desc_id>_<implementation_id>_<scenario>.json` files according to [submission rules](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc).
2. Perform submission checker by running:

```bash
export TRUNCATED_OUTPUT_DIR=<a new directory to save truncated logs>
python3 submission_checker.py -i ${OUTPUT_DIR} -o ${TRUNCATED_OUTPUT_DIR}
```
Please make sure you have read permission of `${OUTPUT_DIR}`, and r/w permission for `${TRUNCATED_OUTPUT_DIR}` and `/tmp`

## Trouble Shooting
```
TypeError: __init__() missing 1 required positional argument: 'status'
```
Solution: start docker serivce before launching the kit.
