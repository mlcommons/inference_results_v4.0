#!/bin/bash

set -ex
# Create new env and activate it
conda install -y ninja==1.10.2 cmake==3.22.1 jemalloc==5.2.1 inflect==5.3.0 libffi==3.4.4 pandas==1.5.3 requests==2.29.0 toml==0.10.2 tqdm==4.65.0 unidecode==1.2.0 scipy==1.9.3
conda install -c intel -y mkl==2023.1.0 mkl-include==2023.1.0 intel-openmp==2023.1.0
conda install -c conda-forge -y llvm-openmp==12.0.1 wheel==0.38.1 setuptools==65.5.1 future==0.18.3
pip install sox

set +x
