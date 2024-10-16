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

ARG BENCHMARK=resnet50
ARG IMPL=pytorch-cpu
ARG BASE_IMAGE=rockylinux:8.7
FROM ${BASE_IMAGE} AS dev-base
RUN --mount=type=cache,id=yum-dev,target=/var/cache/yum \
    DEBIAN_FRONTEND=noninteractive dnf install -y \
    ca-certificates \
    git \
    curl \
    vim \
    numactl \
    numactl-devel \
    cmake \
    sudo \
    wget \
    mesa-libGL \
    gcc-toolset-11-gcc \
    gcc-toolset-11-gcc-c++ \
    && rm -rf /var/lib/yum/lists/*
    
# RUN git config --global user.name "user" && git config --global user.email "user@example.com"
RUN echo "alias ll='ls -l'" >> /root/.bashrc
RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /root/.bashrc
RUN mkdir /opt/ccache
ENV PATH /opt/conda/bin:$PATH



FROM dev-base as conda
ARG PYTHON_VERSION=3.9
RUN echo "Before conda installations"
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} 


FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
ARG PYTORCH_VERSION=v1.12.0-rc7 

FROM dev-base as mp-base
ARG BENCHMARK
ARG IMPL
COPY --from=build /opt/conda /opt/conda
ENV CONDA_PREFIX "/opt/conda"
WORKDIR /opt/workdir
COPY ./code/${BENCHMARK} code/${BENCHMARK}
COPY ./code/run_clean.sh code/${BENCHMARK}/${IMPL}/
COPY ./code/user_config.py code/user_config.py
#COPY ./calibration/resnet50 calibration/resnet50
ARG IPEX_VERSION=v1.12.0
#ARG INFERENCE_COMMIT=8d83d61b5ba8308544aa5a76d311fef1e25d9584
RUN source /opt/rh/gcc-toolset-11/enable && \
    cd code/${BENCHMARK}/${IMPL}/ && chmod +x build_dependencies.sh && ./build_dependencies.sh && \
    cd rn50-mlperf/ && \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
    export IPEX_PATH=${PWD}/ipex-cpu-dev/build/Release/packages/intel_extension_for_pytorch && \
    export TORCH_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` && \
    export LOADGEN_DIR=${PWD}/mlperf_inference/loadgen && \
    export OPENCV_DIR=${PWD}/opencv/build && \
    export RAPIDJSON_INCLUDE_DIR=${PWD}/rapidjson/include && \
    export GFLAGS_DIR=${PWD}/gflags/build && \
    export ONEDNN_DIR=${PWD}/oneDNN && \
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OPENCV_DIR}/lib:${ONEDNN_DIR}/build/src:${CONDA_PREFIX}/lib && \
    export LIBRARY_PATH=${LIBRARY_PATH}:${CONDA_PREFIX}/lib && \
    cd ../ && \
    cmake -DCMAKE_PREFIX_PATH=${TORCH_PATH} \
        -DLOADGEN_DIR=${LOADGEN_DIR} \
        -DOpenCV_DIR=${OPENCV_DIR} \
        -DRapidJSON_INCLUDE_DIR=${RAPIDJSON_INCLUDE_DIR} \
        -Dgflags_DIR=${GFLAGS_DIR} \
        -DINTEL_EXTENSION_FOR_PYTORCH_PATH=${IPEX_PATH} \
        -DONEDNN_DIR=${ONEDNN_DIR} \
        -DCMAKE_BUILD_TYPE=Release \
        -B${PWD}/build \
        -H${PWD}/src && \
    cmake --build ${PWD}/build --config Release -j$(nproc)

ENV MALLOC_CONF "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
ENV LD_PRELOAD "/opt/conda/lib/libjemalloc.so":"/opt/conda/lib/libiomp5.so":${LD_PRELOAD}
ENV CONDA_ENV_NAME "/opt/workdir/code/${BENCHMARK}/${IMPL}"
ENV CONDA_PREFIX "/opt/conda"
