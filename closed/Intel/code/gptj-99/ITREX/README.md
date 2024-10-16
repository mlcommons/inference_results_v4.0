# Setup Instructions

## Anaconda and Conda Environment
+ Download and install conda
  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh
  ```
+ Setup conda environment & dependencies
  ```bash
  bash -l prepare_env.sh gpt-j-env
  conda activate gpt-j-env
  ```


## Get finetuned checkpoint
```bash
pushd gpt-j-env
ENV_DIR=$(pwd)
wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download -O gpt-j-checkpoint.zip
unzip gpt-j-checkpoint.zip
mv gpt-j/checkpoint-final/ "$ENV_DIR/finetuned_gptj"
popd
```

## Get quantized model
+ int4
  ```bash
  CHECKPOINT_DIR="$ENV_DIR/finetuned_gptj" OUT_DIR=${ENV_DIR} FILE_TAG=final bash run_quantization.sh
  # will create quantized_model as "$OUT_DIR/$(basename $CHECKPOINT_DIR)-$FILE_TAG-q4-j-int8-pc.bin"
  MODEL_PATH_INT4="$ENV_DIR/finetuned_gptj-final-q4-j-int8-pc.bin"
  ```

## Run Benchmarks
Note: `WORKERS_PER_PROC` here are for a platform with 32 plysical cores per numa-node.Change accordingly if you have different settings with `WORKERS_PER_PROC=cores_per_numa_node/8`.

+ Offline (Performance)
  ```bash
  USER_CONF=user_int4.conf SCENARIO=Offline MODE=Performance MODEL_PATH=$MODEL_PATH_INT4 WORKERS_PER_PROC=4 BATCH_SIZE=12 bash run_inference.sh
  ```

+ Offline (Accuracy)
  ```bash
  USER_CONF=user_int4.conf SCENARIO=Offline MODE=Accuracy    MODEL_PATH=$MODEL_PATH_INT4 WORKERS_PER_PROC=4 BATCH_SIZE=12 bash run_inference.sh
  ```

+ Server (Performance)
  ```bash
  USER_CONF=user_int4.conf SCENARIO=Server MODE=Performance  MODEL_PATH=$MODEL_PATH_INT4 WORKERS_PER_PROC=1 BATCH_SIZE=4  bash run_inference.sh
  ```

+ Server (Accuracy)
  ```bash
  USER_CONF=user_int4.conf SCENARIO=Server MODE=Accuracy     MODEL_PATH=$MODEL_PATH_INT4 WORKERS_PER_PROC=1 BATCH_SIZE=4  bash run_inference.sh
  ```
