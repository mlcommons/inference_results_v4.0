# syntax = docker/dockerfile:experimental
# based onhttps://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BENCHMARK=3d-unet-99.9
ARG IMPL=pytorch-cpu
ARG BASE_IMAGE=rockylinux:8.7
FROM ${BASE_IMAGE} AS dev-base
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    DEBIAN_FRONTEND=noninteractive dnf install -y \
    ca-certificates \
    git \
    vim \
    numactl \
    cmake \
    unzip \
    wget \
    sudo \
    gcc-toolset-11-gcc \
    gcc-toolset-11-gcc-c++ \
    && rm -rf /var/lib/apt/lists/*
RUN echo "alias ll='ls -l'" >> /root/.bashrc
RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /root/.bashrc 
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.8
RUN wget -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda config --add channels intel && \
    /opt/conda/bin/conda install -y -c intel intel-openmp==2023.1.0 mkl==2023.1.0 mkl-include==2023.1.0 mkl-service==2.4.0 mkl_fft==1.3.6 mkl_random==1.2.2 --no-update-deps && \
    /opt/conda/bin/conda install -y -c conda-forge cmake==3.26.4 jemalloc==5.2.1 gperftools==2.9.1 wheel==0.38.1 setuptools==65.5.1 future==0.18.3 numpy==1.23.0 pandas==1.5.3 && \
    /opt/conda/bin/conda install -y -c conda-forge pybind11==2.10.4 && \
    /opt/conda/bin/conda clean -ya
RUN pip install pyyaml==6.0.1 batchgenerators==0.25 nibabel==5.2.0 scipy==1.10.1 pandas==2.0.3

FROM dev-base AS build
ARG BENCHMARK
ARG IMPL
COPY --from=conda /opt/conda /opt/conda
COPY ./code/${BENCHMARK}/${IMPL}/unet3d.patch unet3d.patch
ARG IPEX_VERSION=1.9.0-rc
ARG ONEDNN_VERSION=v2.7
RUN --mount=type=cache,target=/opt/ccache \
    source /opt/rh/gcc-toolset-11/enable && \
    git clone https://github.com/intel/intel-extension-for-pytorch.git && \
    cd intel-extension-for-pytorch && git checkout ${IPEX_VERSION} && \
    git submodule sync && git submodule update --init --recursive && \
    cd third_party/mkl-dnn/ && git checkout ${ONEDNN_VERSION} && cd ../.. && \
    git apply ../unet3d.patch && \
    python setup.py install

FROM dev-base as mp
ARG BENCHMARK
ARG IMPL
COPY --from=build /opt/conda /opt/conda
WORKDIR /opt/workdir
COPY ./code/${BENCHMARK} code/${BENCHMARK}
COPY ./code/run_clean.sh code/${BENCHMARK}/${IMPL}/run_clean.sh
COPY ./code/user_config.py code/user_config.py
#COPY ./calibration/3d-unet-99.9 calibration/3d-unet-99.9
RUN source /opt/rh/gcc-toolset-11/enable && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git && \
    cd inference && \
    git submodule update --init --recursive && cd loadgen && \
    CFLAGS="-std=c++14" python setup.py install && \
    cd .. && cp ./mlperf.conf /opt/workdir/code/${BENCHMARK}/${IMPL}/. && \ 
    cd /opt/workdir/code/${BENCHMARK}/${IMPL}/utils && \
    /opt/conda/bin/conda install -y -c conda-forge llvm-openmp==16.0.6 && \
    export CPPFLAGS='-I/opt/conda/include -I/opt/conda/include/python3.8' && \
    python -m pip install .

# ENV MALLOC_CONF "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
ENV LD_PRELOAD "/opt/conda/lib/libiomp5.so"
#ENV LD_LIBRARY_PATH "/opt/conda/lib":${LD_LIBRARY_PATH}
ENV CONDA_PREFIX "/opt/conda"
