#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

source code/common/file_downloads.sh

# Make sure the script is executed inside the container
if [ -e /work/code/gptj/tensorrt/download_model.sh ]
then
    echo "Inside container, start downloading..."
else
    echo "WARNING: Please enter the MLPerf container (make prebuild) before downloading gptj6b model."
    echo "WARNING: gptj6b model is NOT downloaded! Exiting..."
    exit 0
fi

# Download the raw weights
download_file models GPTJ-6B \
    https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download \
    gptj6b.zip

# unzip the model
unzip ${MLPERF_SCRATCH_PATH}/models/GPTJ-6B/gptj6b.zip \
    -d ${MLPERF_SCRATCH_PATH}/models/GPTJ-6B

# Move the model up a directory
mv ${MLPERF_SCRATCH_PATH}/models/GPTJ-6B/gpt-j/checkpoint-final ${MLPERF_SCRATCH_PATH}/models/GPTJ-6B/checkpoint-final \
    && rm -r ${MLPERF_SCRATCH_PATH}/models/GPTJ-6B/gpt-j

# Notify the user to generate the quantization model
echo "Please follow README.md for additional steps to generate quantization models for GPTJ."
