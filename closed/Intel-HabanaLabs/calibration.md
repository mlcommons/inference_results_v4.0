# MLPerf Inference – Intel-Habana Labs – Calibration and Quantization Details

## Calibration stage

We pass a set of calibration samples through the neural network executed in BFloat16 to obtain a profile of every activation tensor of the network. 
The profile consists of maximum absolute values over all samples.

## FP8 quantization with controllable exponent bias
In the conversion to FP8-143, we employed four values of exponent bias, specifically [3, 7, 11, 15]. These biases represent ranges of [+/- 0.9375, +/- 15, +/- 240, +/- 3840], respectively.
The bias is selected per-tensor (activation and weight) according to their measured ranges as described in the following.


## LLAMA2 Model quantization

All linear operators' input activations and weights (linear and matmul operators) are quantized to FP8-143. 
The output of the linear operators, as well as the inputs and outputs of all other operators (e.g., softmax, RMS-norm, SwiGLU, RoPE) are in BFloat16 precision. 
The weights are pre-quantized to FP8-143.
For each activation, the  exponent bias is determined by selecting the one whose range, multiplied by a backoff factor, encompasses the measured range during the calibration stage
Similarly, the exponent bias for each weight is determined given its range and backoff factor.
The backoff factor is 0.25 for activations and 0.5 for weights.


## Stable-Diffusion-XL Model quantization

Out of the SDXL model only the UNET block quantized. For each input processed, the UNET block is iteratively invoked 20 times. The quantized model has two forms: a more aggressive form for the first 18 iterations; and a less aggressive one for the final 2 iterations. For each activation, the exponent bias is determined by selecting the one whose range, multiplied by a backoff factor of 0.25, encompasses the measured range during the calibration stage For each weight, the exponent bias is determined as the one which minimizes the weight quantization error.

### For all iterations
The input activations of matmul operations are quantized to FP8-143. The input to all other operations is Bfloat16. 
The output of the softmax operation is in FP8-143 precision. The outputs of all other operators are in BFloat16 precision. 

### For the first 18 iterations
In addition to matmul and softmax operations which are quantized as in the previous section, the input activations and weights of all linear and conv operations are quantized to FP8-143.
The outputs of linear and conv operations are in BFloat16 precision.

### Avoiding double quantization
We use the FP32 base model which is executed in BFloat16 precision. We ensure that linear and conv weights which are quantized to FP8-143 are quantized directly from FP32, and not from BFloat16.