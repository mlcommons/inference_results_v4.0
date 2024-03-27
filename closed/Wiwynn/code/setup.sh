#!/usr/bin/env bash
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -e

# Find model folders. Run the model with its folder in current path
EXIST_MODEL=''
model_list=("resnet50" "retinanet" "rnnt" "3d-unet" "bert" "gpt-j" "dlrm_2")
fd_list=("resnet50"  "retinanet"  "rnnt" "3d-unet-99.9"  "bert-99" "gptj-99" "dlrm-v2-99.9")
for i in "${!fd_list[@]}"
do
   if [ -d ${fd_list[$i]} ]; then
       echo ${fd_list[$i]} "Directory exists. index :" $i
       EXIST_MODEL=${model_list[$i]}
       break
   fi
done

if [ -z "${MODEL_NAME}" ]; then
    echo "Model Name is null."
    if [ -z "${EXIST_MODEL}" ]; then
        echo "export MODEL_NAME={resnet50,retinanet,rnnt,3d-unet,bert,gpt-j,dlrm_2,all}"
        exit 1
    else
	MODEL_NAME=${EXIST_MODEL}
	echo "take exist model" ${MODEL_NAME}
    fi
fi

if [ -z "${DATA_DIR}" ]; then
    echo "Path to dataset is null. Set the default dataset path as ./Dataset"
    DATA_DIR=Dataset
    echo "DATA_DIR : " $DATA_DIR
    
    if [ -d "${DATA_DIR}" ]; then
        mkdir -p ${DATA_DIR}
    fi
fi

cd $(pwd)/automation
if [ -z "${CONDA_PATH}" ]; then
    model=${MODEL_NAME} output_dir=${DATA_DIR} bash download_data.sh
else
    if [ -z "${DATA_TYPE}" ]; then
        model=${MODEL_NAME} output_dir=${DATA_DIR} conda_path=${CONDA_PATH} bash download_data.sh
    else
        model=${MODEL_NAME} output_dir=${DATA_DIR} conda_path=${CONDA_PATH} dtype=${DATA_TYPE} bash download_data.sh
    fi
fi

