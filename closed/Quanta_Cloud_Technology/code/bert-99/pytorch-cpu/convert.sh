set -x

if [ ! -d "${DATA_PATH}" ]; then
	echo "please export the DATA_PATH first!"
	exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
	echo "please export the MODEL_PATH first!"
	exit 1
fi


#convert dataset and model
pushd models
python save_bert_inference.py -m $MODEL_PATH -o ../bert.pt
popd

pushd datasets
python save_squad_features.py -m $MODEL_PATH -d $DATA_PATH -o ../squad.pt
popd

