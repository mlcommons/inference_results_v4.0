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

+ int3
  ```bash
  # GPTQ processing
  pushd gpt-j-env
  git clone https://github.com/intel/neural-compressor.git inc -b gptj-mlperf-new-scripts
  cd inc
  CALIBRATION_DATA=${ENV_DIR}/cnn_dailymail_calibration.json
  VALIDATION_DATA=${ENV_DIR}/cnn_dailymail_validation.json
  pip install -e .
  pip install -r examples/pytorch/nlp/huggingface_models/language-modeling/quantization/llm/requirements.txt
  python -u examples/pytorch/nlp/huggingface_models/language-modeling/quantization/llm/run_gptj_mlperf_int4.py \
      --model_name_or_path "$ENV_DIR/finetuned_gptj" \
      --wbits 3 \
      --sym \
      --group_size 128 \
      --nsamples 256 \
      --calib-data-path ${CALIBRATION_DATA} \
      --val-data-path ${VALIDATION_DATA} \
      --calib-iters 256 \
      --use_max_length \
      --pad_max_length 2048
  CHECKPOINT_DIR_3BIT="${ENV_DIR}/3bit-gpt-j-6b-gptq"
  cp -r "$ENV_DIR/finetuned_gptj" "$CHECKPOINT_DIR_3BIT"
  rm "$CHECKPOINT_DIR_3BIT"/pytorch_model-*.bin
  mv gptj_w3g128_compressed_model.pt $CHECKPOINT_DIR_3BIT/
  popd
  cp ./3bit-config/* $CHECKPOINT_DIR_3BIT/

  # Convert to a all-in-one weight pack
  CHECKPOINT_DIR="$CHECKPOINT_DIR_3BIT" OUT_DIR="$ENV_DIR" FILE_TAG=final bash run_quantization_3bit.sh
  # will create quantized_model as "$OUT_DIR/$(basename $CHECKPOINT_DIR)-$FILE_TAG-gptq.bin"
  MODEL_PATH_INT3="$ENV_DIR/3bit-gpt-j-6b-gptq-final-gptq.bin"
  ```

## Run Benchmarks
Note: `WORKERS_PER_PROC` here are for a platform with 32 plysical cores per numa-node.Change accordingly if you have different settings with `WORKERS_PER_PROC=cores_per_numa_node/8`.

+ Offline (Performance)
  ```bash
  USER_CONF=user<_int4|_int3>.conf SCENARIO=Offline MODE=Performance MODEL_PATH=<$MODEL_PATH_INT4|_INT3> WORKERS_PER_PROC=4 BATCH_SIZE=12 bash run_inference.sh
  ```

+ Offline (Accuracy)
  ```bash
  USER_CONF=user<_int4|_int3>.conf SCENARIO=Offline MODE=Accuracy    MODEL_PATH=<$MODEL_PATH_INT4|_INT3> WORKERS_PER_PROC=4 BATCH_SIZE=12 bash run_inference.sh
  ```

+ Server (Performance)
  ```bash
  USER_CONF=user<_int4|_int3>.conf SCENARIO=Server MODE=Performance  MODEL_PATH=<$MODEL_PATH_INT4|_INT3> WORKERS_PER_PROC=1 BATCH_SIZE=4  bash run_inference.sh
  ```

+ Server (Accuracy)
  ```bash
  USER_CONF=user<_int4|_int3>.conf SCENARIO=Server MODE=Accuracy     MODEL_PATH=<$MODEL_PATH_INT4/_INT3> WORKERS_PER_PROC=1 BATCH_SIZE=4  bash run_inference.sh
  ```
