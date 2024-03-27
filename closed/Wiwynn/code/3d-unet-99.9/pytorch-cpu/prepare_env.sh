set -x

WORKDIR=`pwd`

echo "Install dependencies"
echo "GCC minimum version: 11.1"
source /opt/rh/gcc-toolset-11/enable
conda install -c conda-forge cmake==3.26.4 jemalloc==5.2.1 gperftools==2.9.1 wheel==0.38.1 setuptools==65.5.1 future==0.18.3 numpy==1.23.0 pandas==1.5.3 --yes
conda install intel-openmp==2023.1.0 mkl==2023.1.0 mkl-include==2023.1.0 mkl-service==2.4.0 mkl_fft==1.3.6 mkl_random==1.2.2 --no-update-deps --yes
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

mkdir unet3d_env
cd unet3d_env

echo "Install loadgen"
git clone https://github.com/mlcommons/inference.git
cd inference
git log -1
git submodule update --init --recursive
cd loadgen
CFLAGS="-std=c++14" python setup.py install
cd ../..
cp ./inference/mlperf.conf .


echo "Install Intel Extension for PyTorch"
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout 1.9.0-rc
git submodule sync && git submodule update --init --recursive
git apply ../../unet3d.patch
cd third_party/mkl-dnn/
git checkout v2.7
cd ../..
python setup.py install
cd ../..

cd utils
conda install -y -c conda-forge llvm-openmp==16.0.6 && \
export CPPFLAGS='-I${CONDA_PREFIX}/include -I${CONDA_PREFIX}/include/python3.8' && \
python -m pip install .
cd ../..

bash process_data_model.sh
# python calibrate.py

