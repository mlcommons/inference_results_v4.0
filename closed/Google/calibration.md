# MLPerf Inference Calibration and Quantization Details for TPU

## Quantization steps

All Post training quantization methods we’ve applied for this submission contain these quantization steps. All steps below starts from the pre-trained FP32 model:

*   Calibration step: Run forward pass with the calibration dataset and collect some statistics to quantize the model.
*   Materialization: Quantize model weights and rewrite the model graph to quantize activations be quantized using the statistics collected from the calibration steps.
*   Inference: Run the quantized model for the inference with the updated model graph and weights.


## Calibration and Quantization Details

We use the following formulas for quantization and dequantization: Each quantized value has a “scale” and “zero\_point”. Original float values are quantized linearly (aka. uniform quantization.) For quantization, `Q = round(F / scale + zero_point)`. For dequantization, `F = (Q - zero_point) * scale`. Note that scale and zero\_point are floating values.


### Weight Quantization

We collect weights min/max value during the calibration step. This weight statistic doesn’t use any information from the calibration dataset. It can be calculated from pre-trained float weights. By default we collect w\_min/w\_max for each channel of the weights for channel-wise quantization.

During the materialization step, we compute scale and zero\_point using the w\_min/w\_max range from the calibration dataset. We use 8-bits quantization which Q\_MIN = -128 and Q\_MAX = 127. Here, we only use -127 ~ 127 for symmetric quantization for the weights. The scale is computed using the w\_min and w\_max values. The zero\_point is always zero due to the symmetry. We also quantize the weights using a computed scale, as the quantization scheme above cast the type as int8.

#### Algorithm
We did weight quantization using the saxml offline quantizer tool. For details on the algorithm, please refer to the [README](https://github.com/google/saxml/blob/main/saxml/tools/README.md) for the [offline_quantize.py](https://github.com/google/saxml/blob/main/saxml/tools/offline_quantize.py) tool. 

Specific quantization configs for the models can be found [here](https://github.com/google/saxml/blob/main/saxml/tools/quantization_configs.py).

# For Results using Nvidia Original Implementation

For the results taken using Nvidia original implementation, we are following the same calibration procedure detailed by [Nvidia for their MLPerf Inference v4.0 submissions](https://github.com/mlcommons/inference_results_v4.0/blob/main/closed/NVIDIA/calibration.md)

