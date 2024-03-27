#!/bin/bash

export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
# export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

DIR_SCRIPT=$(dirname "${BASH_SOURCE[0]}")
[ -z $DIR_NS ] && DIR_NS="$DIR_SCRIPT/gpt-j-env/neural-speed"
[ -z $CHECKPOINT_DIR ] && CHECKPOINT_DIR=CHECKPOINT_TO_BE_SET
[ -z $OUT_DIR ] && OUT_DIR=$(dirname $CHECKPOINT_DIR)
[ -z $FILE_TAG ] && FILE_TAG=""
[ -z $PATH_CONVERTED ] && PATH_CONVERTED="$OUT_DIR/$(basename $CHECKPOINT_DIR)-$FILE_TAG-fp32.bin"
[ -z $PATH_QUANTIZED ] && PATH_QUANTIZED="$OUT_DIR/$(basename $CHECKPOINT_DIR)-$FILE_TAG-q4-j-int8-pc.bin"

find "$DIR_NS" -name CMakeCache.txt -exec rm {} \;
pip install -e "$DIR_NS"
INSTALLED_NS=$(python -c "import neural_speed; print(neural_speed.__path__[0])")

# convert
python "$INSTALLED_NS/convert/convert_gptj.py" --outfile "$PATH_CONVERTED" "$CHECKPOINT_DIR"

#quantize
"$INSTALLED_NS/quant_gptj" --model_file "$PATH_CONVERTED" --out_file $PATH_QUANTIZED --weight_dtype int4 --compute_dtype int8 --group_size -1 --nthread 32
