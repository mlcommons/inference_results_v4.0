CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data

export CALIBRATION_DATA_JSON=${WORKLOAD_DATA}/calibration-data/cnn_dailymail_calibration.json

export CHECKPOINT_DIR=${WORKLOAD_DATA}/gpt-j-checkpoint

export VALIDATION_DATA_JSON=${WORKLOAD_DATA}/validation-data/cnn_dailymail_validation.json

export INT8_MODEL_DIR=${WORKLOAD_DATA}/gpt-j-int8-model

export INT4_MODEL_DIR=${WORKLOAD_DATA}/gpt-j-int4-model

export INT4_CALIBRATION_DIR=${WORKLOAD_DATA}/quantized-int4-model

mkdir -p ${INT8_MODEL_DIR}
mkdir -p ${INT4_MODEL_DIR}
