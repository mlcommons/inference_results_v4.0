# Stable Diffusion XL 1.0 Base Quantization with Ammo

This example shows how to use Ammo to calibrate and quantize the UNet part of the SDXL. The UNet part typically consumes >95% of the e2e Stable Diffusion latency.

## Get Started

You may choose to run this example with MLPerf docker or by installing required softwares by yourself.

### SDXL INT8

```sh
python quantize_sdxl.py --pretrained-base {YOUR_FP16_PIPE} --batch-size 1 \
    --calib-size 500 --calib-data ./captions.tsv --percentile 0.4 \
    --n_steps 20 --latent {MLPerf_fixed_lantent_file} --alpha 0.9 --quant-level 2.5 \
    --int8-ckpt-path ./sdxl.int8.pt
```

## ONNX Export

```sh
python export_onnx.py --pretrained-base {YOUR_FP16_PIPE} \
    --quantized-ckpt {YOUR_QUANTIZED_CKPT} --device cuda \
    --quant-level {1.0|2.0|2.5|3.0} --onnx-dir onnx_int8
```