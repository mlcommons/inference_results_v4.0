#!/bin/bash

CONDA_ENV_NAME=${1:-gptj-dev-env}

set -x


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

export WORKDIR=${DIR}/${CONDA_ENV_NAME}
if [ -d ${WORKDIR} ]; then
    rm -r ${WORKDIR}
fi

echo "Working directory is ${WORKDIR}"
mkdir -p ${WORKDIR}

source $(dirname `/usr/bin/which conda`)/activate

conda create -n ${CONDA_ENV_NAME} python=3.9 --yes
conda init bash
conda activate ${CONDA_ENV_NAME}

conda install mkl==2023.2.0 mkl-include==2023.2.0 -y
conda install gperftools==2.10 jemalloc==5.2.1 pybind11==2.10.4 llvm-openmp==16.0.6 -c conda-forge -y
conda install gcc=12.3 gxx=12.3 cxx-compiler ninja==1.11.1 zlib -c conda-forge -y
conda install conda-forge::sysroot_linux-64==2.28 -y

# ========== Install INC deps ===========
python -m pip install cmake==3.27.0 cpuid==0.0.11 nltk==3.8.1 evaluate==0.4.0 protobuf==3.20.3 absl-py==1.4.0 rouge-score==0.1.2 tqdm==4.65.0 numpy==1.25.2 cython==3.0.0 sentencepiece==0.1.99 accelerate==0.21.0

# ============= Install Torch ================
PYTORCH_VERSION=4e2aa5dbb86610854f7d42ded4488f6cb1845be6
cd ${WORKDIR}
git clone https://github.com/pytorch/pytorch.git pytorch
cd pytorch
git checkout ${PYTORCH_VERSION}
git submodule sync
git submodule update --init --recursive
export _GLIBCXX_USE_CXX11_ABI=1
export FLAGSCXX=${CXXFLAGS}
export CXXFLAGS="${CXXFLAGS} -Wno-nonnull"
python setup.py install
python setup.py develop


# =========== Install Ipex ==========
cd ${WORKDIR}
IPEX_VERSION=6047b5410bc8d7ad3a0d9e60acd74d924ae6f676
git clone https://github.com/intel/intel-extension-for-pytorch ipex-cpu
cd ipex-cpu
git checkout ${IPEX_VERSION}
export IPEX_DIR=${PWD}
git submodule sync
git submodule update --init --recursive

export CXXFLAGS="${FLAGSCXX} -D__STDC_FORMAT_MACROS"
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee ipex-build.log
python -m pip install --force-reinstall dist/*.whl

# ================ Install ===============
python -m pip install transformers==4.36.2


# ========== Install INC deps ===========
pip install neural-compressor==2.4.1


# ============ Install loadgen ==========
cd ${WORKDIR}
git clone https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference
export MLPERF_INFERENCE_ROOT=${PWD}


cd loadgen
python -m pip install .

# Copy the mlperf.conf
cp ${MLPERF_INFERENCE_ROOT}/mlperf.conf ${DIR}/

# ======== Build utils =======
cd ${DIR}/utils
python -m pip install .

