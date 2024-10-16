
# Install Miniconda3

# mkdir -p ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh
# export PATH=~/miniconda3/bin:$PATH


source $(dirname `/usr/bin/which conda`)/activate

export CONDA_ENV=sdxl_cpu_bf16
export WORKDIR=${PWD}/${CONDA_ENV}
mkdir -p ${WORKDIR}
conda create -y -n ${CONDA_ENV} python=3.9
conda init bash
conda activate ${CONDA_ENV}

# conda/pip installs
conda install -c intel mkl==2023.2.0 mkl-include==2023.2.0 -y
conda install -c conda-forge jemalloc==5.2.1 gperftools==2.10 pybind11==2.10.4 llvm-openmp==16.0.6 -y
conda install setuptools cmake intel-openmp --yes



pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir scipy==1.9.1 numpy==1.24.4 opencv-python==4.8.1.78
pip install --no-cache-dir pandas==2.1.4 pydantic Pillow==10.2.0 


# Install loadgen
cd ${WORKDIR}
git clone --recurse-submodules https://github.com/mlcommons/inference.git --depth 1
cd inference/loadgen
CFLAGS="-std=c++14" python setup.py install
cp ../mlperf.conf ${WORKDIR}/configs/
rm -rf inference/
cd ${WORKDIR}../


# # ========== Install torch ===========

python -m pip install --pre torch==2.3.0.dev20231214+cpu torchvision==0.18.0.dev20231214+cpu --index-url https://download.pytorch.org/whl/nightly/cpu

# =========== Install Ipex ==========

cd ${WORKDIR}
git clone https://github.com/intel/intel-extension-for-pytorch.git ipex-cpu

cd ipex-cpu
git fetch
git checkout 5aeb11d79cb5673bdb017c44e01d0e8bca8a0548

export IPEX_DIR=${PWD}
git submodule sync
git submodule update --init --recursive

python setup.py clean
python setup.py bdist_wheel 2>&1 | tee ipex-build.log
python -m pip install --force-reinstall dist/*.whl
python -m pip install numpy==1.24.4

cd ${WORKDIR}/../
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# ======== Build utils =======
cd ${WORKDIR}/../utils
python -m pip install .


# Pulling coco.py from inference repo
cd ${WORKDIR}/../tools
wget https://raw.githubusercontent.com/mlcommons/inference/master/text_to_image/tools/coco.py


