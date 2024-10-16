echo "Install dependency packages"
pip install -e git+https://github.com/mlperf/logging@3.0.0-rc2#egg=mlperf-logging
pip install absl-py==1.4.0 tqdm==4.65.0
pip install sklearn==0.0.post7 onnx==1.14.0
pip install lark-parser==0.12.0 hypothesis==6.82.0
conda install numpy==1.25 ninja==1.10.2 pyyaml==6.0 mkl==2023.1.0 mkl-include setuptools==68.0.0 cmake==3.26.4 cffi==1.15.1 typing_extensions==4.7.1 future==0.18.3 six==1.16.0 requests==2.31.0 dataclasses==0.8 psutil==5.9.0 -y
conda install -c conda-forge gperftools==2.10  --yes