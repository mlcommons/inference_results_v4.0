
# Specify Cmake requirement
cmake_minimum_required(VERSION 3.10)

if(DEFINED CUDA_INSTALL_PATH)
  set(CUDA_DIR ${CUDA_INSTALL_PATH})
else()
  set(CUDA_DIR /usr/local/cuda)
endif()

if(DEFINED TRT_INSTALL_PATH)
  set(TRT_DIR ${TRT_INSTALL_PATH})
else()
  set(TRT_DIR "")
endif()

if(DEFINED CUPVA_LIB_PATH)
  set(CUPVA_LIB ${CUPVA_LIB_PATH})
else()
  set(CUPVA_LIB /opt/nvidia/cupva-2.0/lib/aarch64-linux-gnu/)
endif()

##################################################################################################
set(TRT_LIB ${TRT_DIR}/usr/lib/aarch64-linux-gnu)
set(TRT_HEADER ${TRT_DIR}/usr/include/aarch64-linux-gnu)
set(CUDA_LIB ${CUDA_DIR}/targets/aarch64-linux/lib)
set(CUDA_HEADER ${CUDA_DIR}/targets/aarch64-linux/include)

# Define Project
project(retinanetNMSPVA_aarch64_linux VERSION 1.0.0 DESCRIPTION "RetinaNet NMS Using PVA Kernel and TRT Plugin Wrapper")

# Add cpp sources - will depend on the project's directories
add_library(retinanetNMSPVA_aarch64_linux SHARED
    src/retinanetNMSPVAPlugin.cpp
)

# Set basic project properties
set_target_properties(retinanetNMSPVA_aarch64_linux PROPERTIES VERSION ${PROJECT_VERSION})
# Set SO version
set_target_properties(retinanetNMSPVA_aarch64_linux PROPERTIES SOVERSION 1)


# Declare all directories with PUBLIC headers
target_include_directories(retinanetNMSPVA_aarch64_linux PUBLIC ${CUDA_HEADER})
target_include_directories(retinanetNMSPVA_aarch64_linux PUBLIC ${TRT_HEADER})
target_include_directories(retinanetNMSPVA_aarch64_linux PUBLIC /opt/nvidia/cupva-algos-gen2-2.0/algos/gen2/retinanet_nms/pva/include/) #Replace once the deb package is hosted
target_include_directories(retinanetNMSPVA_aarch64_linux PUBLIC src/)

# Target libs to be linked
target_link_libraries(retinanetNMSPVA_aarch64_linux PUBLIC /opt/nvidia/cupva-algos-gen2-2.0/algos/gen2/retinanet_nms/pva/lib/libcupvadlops.so) #Replace once the deb package is hosted
target_link_libraries(retinanetNMSPVA_aarch64_linux PUBLIC ${CUPVA_LIB}/libcupva_host.so.2.0)
target_link_libraries(retinanetNMSPVA_aarch64_linux PUBLIC ${CUDA_LIB}/libcudart.so)
target_link_libraries(retinanetNMSPVA_aarch64_linux PUBLIC ${TRT_LIB}/libnvinfer.so)
