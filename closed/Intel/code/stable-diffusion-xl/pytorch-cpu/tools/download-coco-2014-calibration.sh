#!/bin/bash

: "${DOWNLOAD_PATH:=../coco2014}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --download-path )       shift
                                     DOWNLOAD_PATH=$1
                                     ;;
    esac
    case $1 in
        -i | --images )
                                     IMAGES=1
                                     ;;
    esac
    case $1 in
        -n | --num-workers  )        shift
                                      NUM_WORKERS=$1
                                      ;;
    esac
    shift
done

# if [ -z ${IMAGES} ];
# then
export CUR_DIR=${PWD}
export DATASET_DIR=${DOWNLOAD_PATH}
mkdir -p ${DATASET_DIR}
echo ${DATASET_DIR}
mkdir -p ${DATASET_DIR}/raw
mkdir -p ${DATASET_DIR}/download_aux
mkdir -p ${DATASET_DIR}/captions

cd ${DATASET_DIR}/download_aux/ && wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip --show-progress
unzip annotations_trainval2014.zip -d ../raw
mkdir -p ${DATASET_DIR}/captions/
echo "Before Moving"
mv ${DATASET_DIR}/raw/annotations/captions_train2014.json ${DATASET_DIR}/captions/
rm -rf ${DATASET_DIR}/raw
rm -rf ${DATASET_DIR}/download_aux

cd ${CUR_DIR}
python3 coco_calibration.py \
    --dataset-dir ${DOWNLOAD_PATH} \
    --num-workers ${NUM_WORKERS}


