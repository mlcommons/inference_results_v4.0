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

ARG BENCHMARK=stable_diffusion
ARG IMPL=pytorch-cpu
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS dev-base
RUN apt-get update && apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    vim \
    numactl \
    cmake \
    wget \
    findutils \
    build-essential \
    gcc-12 \
    g++-12 \
    libzstd-dev \
    libgtk2.0-dev \
    libgl1 \
    libxml2-dev \
    zlib1g-dev \
    libdata-dumper-simple-perl \
    && rm -rf /var/lib/apt/lists/* &&\
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 10 && \ 
    update-alternatives --set gcc "/usr/bin/gcc-12" && \
    update-alternatives --set g++ "/usr/bin/g++-12"
RUN echo "alias ll='ls -l'" >> /root/.bashrc
RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /root/.bashrc
RUN mkdir /opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.9
RUN wget -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install python=${PYTHON_VERSION} -y && \
    /opt/conda/bin/conda install -c intel mkl==2023.2.0 mkl-include==2023.2.0 intel-openmp==2023.1.0 -y && \
    /opt/conda/bin/conda install -c conda-forge jemalloc==5.2.1 gperftools==2.10 pybind11==2.10.4 llvm-openmp==16.0.6 -y && \
    /opt/conda/bin/conda install -c conda-forge setuptools==68.2.2 cmake==3.26.4 --yes 

RUN pip install scipy==1.9.1 numpy==1.24.4 opencv-python==4.8.1.78 pandas==2.1.4 pydantic Pillow==10.2.0 && \
    #pip install --pre torch==2.3.0.dev20231214+cpu torchvision==0.18.0.dev20231214+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
    pip install --pre torch==2.3.0.dev20231216+cpu torchvision==0.18.0.dev20231216+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
 
FROM conda AS build
ARG BENCHMARK
ARG IMPL
COPY --from=conda /opt/conda /opt/conda
WORKDIR /opt/workdir
COPY ./code/${BENCHMARK} code/${BENCHMARK}
COPY ./code/run_clean.sh code/${BENCHMARK}/${IMPL}/run_clean.sh
ENV CONDA_PREFIX "/opt/conda"
RUN update-ca-certificates -f && \
    cd /opt/workdir/code/${BENCHMARK}/${IMPL} && export WORKDIR=${PWD} && mkdir stable-diffusion-mlperf && cd stable-diffusion-mlperf && \
    git clone https://github.com/huggingface/diffusers.git && cd diffusers && git checkout v0.25.1 && cp ../../diffusers.patch . && git apply diffusers.patch && ls && python setup.py install && \
    cd /opt/workdir/code/${BENCHMARK}/${IMPL} && python -m pip install -r requirements.txt && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git --depth 1 && \
    cd inference/loadgen && \
    CFLAGS="-std=c++14" python setup.py install && \
    cp ../mlperf.conf ${WORKDIR}/configs/&& \
    cd ../../ && rm -rf inference/ && \
    cd ${WORKDIR}/stable-diffusion-mlperf/ && git clone https://github.com/intel/intel-extension-for-pytorch.git ipex-cpu && cd ipex-cpu && git checkout 5aeb11d79cb5673bdb017c44e01d0e8bca8a0548 && \
    export IPEX_DIR=${PWD} && git submodule sync && git submodule update --init --recursive && \
    python setup.py clean && python setup.py bdist_wheel 2>&1 | tee ipex-build.log && python -m pip install --force-reinstall dist/*.whl && cd ${WORKDIR} && export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH} && \
    cd ${WORKDIR}/utils && python -m pip install . && \
    cd ${WORKDIR}/tools && wget https://raw.githubusercontent.com/mlcommons/inference/master/text_to_image/tools/coco.py

ENV LD_LIBRARY_PATH "/opt/conda/lib:${LD_LIBRARY_PATH}"
FROM dev-base as mp
COPY --from=build /opt/conda /opt/conda
COPY --from=build /opt/workdir /opt/workdir
WORKDIR /opt/workdir
ENV CONDA_PREFIX "/opt/conda"

