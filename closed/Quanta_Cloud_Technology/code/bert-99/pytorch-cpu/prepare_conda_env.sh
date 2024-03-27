#! /bin/bash

# Create new env and activate it
conda install -y python=3.8
conda install -y ninja=1.10.2
conda install -y cmake=3.22.1
conda install -c intel mkl=2023.1.0 --yes
conda install -c intel mkl-include=2023.1.0 --yes
conda install -c intel intel-openmp=2023.1.0 --yes
conda install -c conda-forge llvm-openmp=12.0.1 --yes
conda install -c conda-forge jemalloc=5.2.1 --yes
conda install -c conda-forge wheel=0.38.1 --yes
conda install -c conda-forge setuptools=65.5.1 --yes
conda install -c conda-forge future=0.18.3 --yes