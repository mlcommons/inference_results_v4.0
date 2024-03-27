export DOWNLOAD_DATA_DIR=/data/model/kits19/data

pip install nibabel scipy pandas
make setup
make duplicate_kits19_case_00185

make preprocess_data
make preprocess_calibration_data
make preprocess_gaussian_patches

export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:$LD_PRELOAD
python trace_model.py

mv build/model/3dunet_kits19_pytorch_checkpoint.pth /data/model/
