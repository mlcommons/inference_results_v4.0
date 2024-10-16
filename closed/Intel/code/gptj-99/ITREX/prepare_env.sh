#!/bin/bash -l

CONDA_ENV_NAME=$1
[ -z $CONDA_ENV_NAME ] && CONDA_ENV_NAME=gpt-j-env

set -xeo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

export WORKDIR=${DIR}/${CONDA_ENV_NAME}
if [ -d ${WORKDIR} ]; then
    rm -rf ${WORKDIR}
fi

echo "Working directory is ${WORKDIR}"
mkdir -p ${WORKDIR}

shopt -s expand_aliases
if [[ -n $(conda env list | grep "$CONDA_ENV_NAME") ]]; then
    conda remove -n "$CONDA_ENV_NAME" --all --yes
fi
conda create -n "$CONDA_ENV_NAME" -c conda-forge python=3.9 mkl=2023 mkl-include=2023 --yes
# conda init bash
conda activate "$CONDA_ENV_NAME"

conda install -c conda-forge -y jemalloc==5.2.1 llvm-openmp==16.0.6 pybind11=2.11.1 gcc=13 gxx=13 cmake ninja git

# ============ Install torch-cpu =========
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# ============ Install NS =========
cd ${WORKDIR}
rm -rf neural-speed
git clone -b mlperf-v4-0 https://github.com/intel/neural-speed.git
pip install -r neural-speed/requirements.txt
pip install -ve ./neural-speed

# ============ Install loadgen ==========
cd ${WORKDIR}
rm -rf mlperf_inference
git clone https://github.com/mlcommons/inference.git mlperf_inference --depth=1
pip install -ve ./mlperf_inference/loadgen

# Copy the mlperf.conf
cp mlperf_inference/mlperf.conf ${DIR}/

# ======== Build utils =======
cd ${DIR}
pip install -ve ./utils

# ======== accuracy validation requirements =======
pip install nltk==3.8.1 evaluate==0.4.1 rouge-score==0.1.2

# ======== download dataset =======
cd ${DIR}
python download-dataset.py --split validation --output-dir ${WORKDIR}
python download-calibration-dataset.py --calibration-list-file calibration-list.txt --output-dir ${WORKDIR}
