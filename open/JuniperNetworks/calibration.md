# MLPerf Inference v4.0 - Juniper Networks Calibration

Juniper's submission uses the Nvidia Ammo Toolkit for Quantization. The sections below describe in detail the type of quantization used based on the hardware the results were gathered on.

## For Results on A100 GPUs

INT8-WO Quantization

This submission performs only a weight-only quantization for A100 GPUs.
INT8 Weight-Only techniques consist in quantizing the weights of a model and de-quantizing those weights on-the-fly in linear layers (Matmuls). The activations are encoded using floating-point values.

The following tensors are quantized for INT8-WO quantization method

- Weights
- KV Cache entries

## For Results on H100 GPUs

FP8 Quantization

All quantization (including weight quantization) is symmetric, per-tensor.

The dynamic range for each tensor is defined to be the 99.9 percentile value observed in the values of that tensor when the model is executed in FP32 on the calibration dataset. For a tensor with dynamic range dr, the quantized value x_q is computed from the unquantized value x as:

x_q = round(clip(x / dr * m, -m, m))
where m is max of FP8 format, in this case 448, and ties are rounded to even.

When quantizing Llama2-70B, The following tensors are quantized in each decoder:

- linear layer inputs and weights
- Q, K, V input to fMHA
- matmul inputs and weights
- KV Cache entries
