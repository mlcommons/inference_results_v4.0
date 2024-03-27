# based onhttps://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BENCHMARK=dlrm-v2-99.9
ARG IMPL=pytorch-cpu-int8
ARG BASE_IMAGE=rockylinux:8.6
ARG PYTHON_VERSION=3.9
ARG GCC_MAJOR_VERSION=12
ARG IPEX_BRANCH=release/2.1

FROM ${BASE_IMAGE} AS dev-base
ARG GCC_MAJOR_VERSION
RUN --mount=type=cache,id=yum-dev,target=/var/cache/yum \
    DEBIAN_FRONTEND=noninteractive dnf install -y \
    ca-certificates \
    git \
    vim \
    numactl \
    cmake \
    sudo \
    wget \
    gcc-toolset-${GCC_MAJOR_VERSION}-gcc \
    gcc-toolset-${GCC_MAJOR_VERSION}-gcc-c++ \
    && rm -rf /var/lib/yum/lists
RUN echo "source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable" >> /root/.bashrc
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION
RUN wget -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda config --add channels intel && \
    /opt/conda/bin/conda install -y -c conda-forge cmake==3.26.3 gperftools==2.10 && \
    /opt/conda/bin/conda install -y -c intel mkl==2023.1.0 mkl-include==2023.1.0 && \
    /opt/conda/bin/conda install -y -c conda-forge numpy==1.25.2 \
                                    ninja==1.10.2 \
                                    pyyaml==6.0 \
                                    setuptools==68.0.0 \
                                    cffi==1.15.1 \
                                    typing_extensions==4.7.1 \
                                    future==0.18.3 \
                                    six==1.16.0 \
                                    requests==2.31.0 \
                                    dataclasses==0.8 \
                                    psutil==5.9.0 \
                                    --no-update-deps && \
    /opt/conda/bin/conda clean -ya
RUN pip install -e git+https://github.com/mlperf/logging@3.0.0-rc2#egg=mlperf-logging && \
    pip install -U 'git+https://github.com/facebookresearch/iopath' && \
    pip install absl-py==1.4.0 \
                tqdm==4.65.0 \
                onnx==1.14.0 \
                lark-parser==0.12.0 \
                hypothesis==6.82.0 \
                pyre-extensions==0.0.30 \
                scikit-learn==1.3.0 \
                pybind11==2.11.1 && \
    pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install fbgemm-gpu==0.5.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install torchmetrics==1.0.3 && \
    pip install torchrec==0.5.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install torchsnapshot

FROM dev-base AS build
ARG GCC_MAJOR_VERSION
ARG IPEX_BRANCH
ARG BENCHMARK
ARG IMPL
COPY --from=conda /opt/conda /opt/conda
COPY ./code/${BENCHMARK}/${IMPL}/ipex.patch ipex.patch
RUN --mount=type=cache,target=/opt/ccache \
    source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable && \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
    git clone https://github.com/intel/intel-extension-for-pytorch.git intel-extension-for-pytorch && \
    cd intel-extension-for-pytorch && \
    git checkout ${IPEX_BRANCH} && \
    git checkout f1bc84b67b65f4de23dc30ec8f8326a3ea40f5ee && \
    git apply --reject --whitespace=fix ../ipex.patch && \
    git submodule sync && git submodule update --init --recursive && \
    python setup.py install

FROM dev-base as mp
ARG GCC_MAJOR_VERSION
ARG BENCHMARK
ARG IMPL
COPY --from=build /opt/conda /opt/conda
WORKDIR /opt/workdir
COPY ./code/${BENCHMARK}/${IMPL} code/${BENCHMARK}/${IMPL}
COPY ./code/run_clean.sh code/run_clean.sh
COPY ./code/user_config.py code/user_config.py
RUN source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git inference && \
    cd inference && \
    git submodule update --init --recursive && cd loadgen && \
    CFLAGS="-std=c++14" python setup.py install && \
    cd .. && cp ./mlperf.conf /opt/workdir/code/${BENCHMARK}/${IMPL}/.

ENV LD_PRELOAD "/opt/conda/lib/libiomp5.so"
ENV CONDA_PREFIX "/opt/conda"
