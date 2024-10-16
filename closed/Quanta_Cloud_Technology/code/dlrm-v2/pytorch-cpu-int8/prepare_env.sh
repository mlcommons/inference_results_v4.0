  export WORKDIR=$PWD
  REPODIR=$WORKDIR/frameworks.ai.benchmarking.mlperf.develop.inference-datacenter

  PATTERN='[-a-zA-Z0-9_]*='
  if [ $# -lt "0" ] ; then
      echo 'ERROR:'
      printf 'Please use following parameters:
      --code=<mlperf workload repo directory>
      '
      exit 1
  fi

  for i in "$@"
  do
      case $i in
         --code=*)
              code=`echo $i | sed "s/${PATTERN}//"`;;
          *)
              echo "Parameter $i not recognized."; exit 1;;
      esac
  done

  if [[ -n $code && -d ${code} ]];then
     REPODIR=$code
  fi

  echo "Install python packages"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu # torch 2.1 official release
  pip install pyre-extensions
  pip install scikit-learn

  echo "Install loadgen"
  git clone https://github.com/mlcommons/inference.git
  cd inference
  git log -1
  git submodule update --init --recursive
  cd loadgen
  CFLAGS="-std=c++14" python setup.py install
  cd ..; cp ${WORKDIR}/inference/mlperf.conf ${REPODIR}/closed/Intel/code/dlrm2-99.9/pytorch-cpu/.

  echo "Clone source code and Install"
  echo "Install Intel Extension for PyTorch"
  cd ${WORKDIR}
  # clone Intel Extension for PyTorch
  git clone https://github.com/intel/intel-extension-for-pytorch.git intel-extension-for-pytorch
  cd intel-extension-for-pytorch
  git checkout llm_feature_branch
  git apply ../ipex.patch
  git submodule sync
  git submodule update --init --recursive
  cd third_party/ideep/mkl-dnn
  patch -p1 < ${REPODIR}/closed/Intel/code/dlrm2-99.9/pytorch-cpu/onednngraph.patch
  git log -1
  python setup.py install
  cd ..
