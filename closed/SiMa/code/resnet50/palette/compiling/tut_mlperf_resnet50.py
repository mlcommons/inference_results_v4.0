#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2023-2023 SiMa.ai, All Rights Reserved.  ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************
import dataclasses
import logging
import numpy as np
import os
from pathlib import Path

from afe.apis.defines import HistogramMSEMethod, default_quantization, ScalarType
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.apis.loaded_net import load_model, onnx_source
from afe.apis.release_v1 import get_model_sdk_version, DataGenerator, convert_data_generator_to_iterable
from afe.apis.model import L2CachingMode


"""
Script for quantizing and compiling
"""

model = "resnet50_v1_opt.onnx"

sample_start = 10
max_calib_samples = 35
number_of_histogram_bins = 2048
calib_method = HistogramMSEMethod(number_of_histogram_bins)
quant_configs = dataclasses.replace(default_quantization, calibration_method=calib_method)

MODEL_DIR = './'


def compile_model(model_name: str, arm_only: bool, batch_size: int):

    # Uncomment the following line to enable verbose error messages.
    enable_verbose_error_messages()
    
    print(f"Compiling model {model_name} with batch_size={batch_size} and arm_only={arm_only}", flush=True)

    # Models importer parameters
    # input shape in format NCHW with N (the batchsize) = 1
    input_name, input_shape, input_type = ("input_tensor:0", (1,3,224,224), ScalarType.float32)

    input_shapes_dict = {input_name: input_shape}
    input_types_dict = {input_name: input_type}

    model_path = Path(MODEL_DIR) / model_name

    # refer to the SDK User Guide for the specific format 
    importer_params = onnx_source(str(model_path), input_shapes_dict, input_types_dict)

    model_prefix = model_path.stem
    output_dir = f'./output_bs{batch_size}_{calib_method.name}'
    os.makedirs(output_dir, exist_ok=True)
    loaded_net = load_model(importer_params)

    # Read images
    cal_data_path = os.path.join(MODEL_DIR, "calibration/mlperf_resnet50_cal_NCHW.dat")
    cal_label_path = os.path.join(MODEL_DIR, "calibration/mlperf_resnet50_cal_labels_int32.dat")
    cal_dat = np.fromfile(cal_data_path, dtype=np.float32).reshape(500, 3, 224, 224)
    cal_labels = np.fromfile(cal_label_path, dtype=np.int32)
    
    # Tranpose images from NCHW to NHWC
    cal_dat_NHWC = cal_dat.transpose(0, 2, 3, 1)

    dg = DataGenerator({input_name: cal_dat_NHWC[sample_start:sample_start + max_calib_samples]})
    calibration_data = convert_data_generator_to_iterable(dg)

    model_sdk_net = loaded_net.quantize(calibration_data,
                                        quant_configs,
                                        model_name=model_prefix,
                                        arm_only=arm_only)

    saved_model_directory = f"sdk_bs{batch_size}"
    model_sdk_net.save(model_name=model_name, output_directory=saved_model_directory)

    l2_caching_mode = L2CachingMode.SINGLE_MODEL if batch_size == 1 else L2CachingMode.NONE

    model_sdk_net.compile(output_path=output_dir,
                          batch_size=batch_size,
                          log_level=logging.INFO,
                          l2_caching_mode=l2_caching_mode,
                          tessellate_parameters={"MLA_0/placeholder_0": [[], [224], [3]]})

    print(f"Compiling model {model_name} with batch_size={batch_size} and arm_only={arm_only} done", flush=True)


def main():
    print("SiMa Model SDK tutorial example of onnx MLPerf ResNet50", flush=True)

    # Get Model SDK version
    sdk_version = get_model_sdk_version()
    print(f"Model SDK version: {sdk_version}", flush=True)

    batch_sizes = [1, 8, 14]
    for batch_size in batch_sizes:
        compile_model(model, False, batch_size)


if __name__ == "__main__":
    main()

