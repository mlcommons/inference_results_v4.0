# RetinaNetNMS PVA Plugin
Input to the plugin is 10 tensors (5 box tensors and 5 score tensors) in **int8** format. It performs TopK and NMS on these inputs using PVA (It uses only single VPU). Output from the plugin is 1000 boxes, along with their confidence and class labels.

**Below are the contents of cupva-algos-gen2-2.0.0-cupva_algo_dlops.deb**
- pva/include/RetinaNetNMSAppSync.hpp  
- pva/include/cuda_helper_api.h
- pva/lib/libcupvadlops.so
- onnx/retinanet_nms_standalone_kchw32_vpuAny_anchorXYWH_indScoreThresh.onnx

**libcupvadlops.so** contains RetinaNetNMS PVA kernel

IMPORTANT NOTE: Currently, compilation is supported only for JetPack 5.1.1 - Tegra/aarch64-Linux

## NMSPVAPlugin : Sample application to demonstrate how to use this debian package

This sample application is part of NVIDIA’s MLPerf-Inference 3.1 submission.

### Name and Version of the NMSPVAPlugin
- The default name of the plugin is RetinaNetNMSPVATRT: If the name/op of the NMS node in your onnx graph is not 'RetinaNetNMSPVATRT', then need to change name and attributes as per the onnx/retinanet_nms_standalone_kchw32_vpuAny_anchorXYWH_indScoreThresh.onnx .
- The version of the plugin is '1'

### Prerequisites for NMSPVAPlugin Utilization
```
1. Cuda-Toolkit:
    - Follow instructions in DRIVE OS Documentation to install Cuda Toolkit
    - Ensure that the Version (Major, Minor) is exactly the same for Cuda-Toolkit-Cross and Cuda-Toolkit-Target
        - For example: In CUDA-Toolkit Version 11.4 or 'CUDA-11.4', the major is '11' and minor is '4'
    - Cuda-Toolkit-Cross allows us to Cross-Compile the code for a required Target (Tegra-Linux)
    - Cuda-Tookit-Target is cross compiled for the target and must be installed on the target.

2. TensorRT:
    - Follow instructions in DRIVE OS Documentation to install TensorRT
    - Ensure that the Version (Major, Minor, Patch, Build) is exactly the same for TensorRT-Cross and TensorRT-Target
        - For example: In TensorRT Version 8.4.12.5, the major is '8', minor is '4', patch is '12' and build is '5'
    - TensorRT-Cross allows us to Cross-Compile the code for a required Target (Tegra-Linux)
    - TensorRT-Target is cross compiled for the target and must be installed on the target.

3. CUPVA runtime:
    - CUPVA 2.0.0 runtime library is installed in JP 5.1.1 by default.

4. cupva-algos-gen2-2.0.0-cupva_algo_dlops.deb

```

### Standalone Compilation of NMSPVAPlugin
CMakeList.txt in the NMSPVAPlugin/ directory enables compilation of the plugin.

#### Various Paths Required for Compilation 

##### Common Paths for aarch64/Tegra Linux

1. CUDA_INSTALL_PATH:
    - If CUDA is installed at default path (`/usr/local/cuda`), setting this variable is not required
    - If CUDA is extracted from .deb file and is not at default path, set this path to one level above `/usr` in the location where the files are extracted

2. TRT_INSTALL_PATH:
    - If TensorRT is installed at default paths (specified below), setting this variable is not required
    - If TensorRT is extracted from .deb file and not installed at default paths, set this variable to one level above `/usr` in the location where the files are extracted.
    - Default paths for TensorRT are:
        - L4T: .so files at `/usr/lib/aarch64-linux-gnu`, headers at `/usr/include/aarch64-linux-gnu`

3. CUPVA_LIB_PATH:
    - If CUPVA runtime is installed at default path (/opt/nvidia/cupva-2.0/lib/aarch64-linux-gnu/), setting this variable is not required. 
    - If it is not at default path set the path till the level of /opt/nvidia/cupva-2.0/lib/aarch64-linux-gnu/ directory of the extracted CUPVA runtime Packages

##### Aarch64-Linux or Tegra-Linux Specific Paths
1. COMPILER_PATH: If `aarch64-linux-gnu-g++` is not installed at default location (`/usr/bin`), provide the path to the directory containing `'aarch64-linux-gnu-g++'`.

#### Steps to build the standalone plugin
```
1. Disable vpu authentication
    $ echo 0 | sudo tee /sys/kernel/debug/pva0/vpu_app_authentication

2. Ensure that you are in plugin directory
    $ cd NMSPVAPlugin/

3. Create build directory and move to build directory
    $ mkdir build && cd build

4.  A. ENSURE THAT REQUIRED PACKAGES ARE INSTALLED/EXTRACTED ON THE SYSTEM
    B. ENSURE THAT REQUIRED PATH VARIABLES SPECIFIED ABOVE ARE SET IN THE ENVIRONMENT

5. To Build for aarch64-Linux or Tegra-Linux:
    $ cmake .. -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
    $ make

NOTE: 
1. '$' represents bash command prompt

2. aarch64-linux-gnu-g++ is not installed then install using following commands

    $sudo apt update
    $sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

```

### MACROS in the NMSPVAPlugin
1. #define DEBUG_RETINANET_NMS_PLUGIN:
- Location: file NMSPVAPlugin/src/retinanetNMSPVAPlugin.cpp
- When set to 1, will enable debug prints in various Plugin Methods
- Set to 0 by default

2. #define READ_SCALES_FROM_DESC
- Location: file NMSPVAPlugin/src/retinanetNMSPVAPlugin.cpp
- When set to 1, will read dequantization scales from PluginTensorDesc
- When set to 0, will use default scales set in RetinaNetNMSPlugin::configurePlugin
- Set to 1 by default

