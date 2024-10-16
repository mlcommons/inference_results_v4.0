# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(mlperf-inference)

include(GNUInstallDirs)
find_package(CUDA REQUIRED)

# Build options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Pass the Loadgen include directory from command line
add_definitions(-DLOADGEN_INCLUDE_DIR=${LOADGEN_INCLUDE_DIR})

# Pass the TRTLLM directory from command line
add_definitions(-DTRTLLM_DIR=${TRTLLM_DIR})

# Workaround for TRT header warning
execute_process(COMMAND echo "Warning: setting -Wno-deprecated-declarations to avoid header warnings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# Set sm versions
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.0)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_80,code=sm_80")
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.1)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_86,code=sm_86")
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.4)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_87,code=sm_87")
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.8)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_90,code=sm_90")
endif()

project(harness LANGUAGES CXX CUDA)

# Find the static Loadgen library
unset(LOADGEN_LIB CACHE)
find_library(LOADGEN_LIB NAMES libmlperf_loadgen.a PATHS ${LOADGEN_LIB_DIR})

# Set the path to the LWIS library
unset(LWIS_INCLUDE_DIR CACHE)
set(LWIS_INCLUDE_DIR lwis/include)

# Set the path to the Triton library
unset(TRITON_DIR CACHE)
set(TRITON_DIR ../../build/triton-inference-server)

# Set NVTX library path
unset(NV_TOOLS_EXT_LIB CACHE)
set(NV_TOOLS_EXT_LIB ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so)

# Build the harness for the Triton harness and for DLRM if not on "aarch64" platform
execute_process(COMMAND uname -p OUTPUT_VARIABLE ARCH)

######### DEFAULT HARNESS ########
if (${SOC_SM})
    if (${SOC_SM} STREQUAL "87")
        add_compile_definitions(IS_ORIN)
    endif()
endif()

# Add the LWIS subdirectory (which will generate a static LWIS library)
add_subdirectory(lwis)

# Build the default harness which covers single_stream and offline scenarios on image benchmarks.
execute_process(COMMAND echo "Building default harness...")
add_executable(harness_default
    harness_default/main_default.cc
    common/logger.cpp
)

target_link_libraries(harness_default
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    lwis
    ${LOADGEN_LIB}
    numa
)

target_include_directories(harness_default
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LWIS_INCLUDE_DIR}
        common
)

######### LLM HARNESS ########
# SOC doesn't run LLM for now
if (${BUILD_TRTLLM})
    execute_process(COMMAND echo "TRTLLM harness build param: IS_SOC: ${IS_SOC} ARCH: ${ARCH}")
    if ("${IS_SOC}${ARCH}" MATCHES "1aarch64")
        execute_process(COMMAND echo "Building LLM harness on Jetson Orin...")
        find_package(MPI REQUIRED)
        message(STATUS "Using MPI_C_INCLUDE_DIRS: ${MPI_C_INCLUDE_DIRS} and MPI_C_INCLUDE_PATH  is ${MPI_C_INCLUDE_PATH} ")
        message(STATUS "Using MPI_C_LIBRARIES: ${MPI_C_LIBRARIES}MPI_CXX_LIBRARIES is  ${MPI_CXX_LIBRARIES} ")     
        unset(TRTLLM_CPP_DIR CACHE)
        set(TRTLLM_CPP_DIR ${TRTLLM_DIR}/cpp)
        message(STATUS "TRTLLM_CPP_DIR path: " ${TRTLLM_CPP_DIR})
        unset(TRTLLM_LIB CACHE)
        find_library(TRTLLM_LIB NAMES libtensorrt_llm.so PATHS ${TRTLLM_CPP_DIR}/build/tensorrt_llm/)
        message(STATUS "TRTLLM_LIB path: " ${TRTLLM_LIB})
        unset(TRTLLM_PLUGIN_LIB CACHE)
        find_library(TRTLLM_PLUGIN_LIB NAMES libnvinfer_plugin_tensorrt_llm.so PATHS ${TRTLLM_CPP_DIR}/build/tensorrt_llm/plugins)
        message(STATUS "TRTLLM_PLUGIN_LIB path: " ${TRTLLM_PLUGIN_LIB})
        unset(TRTLLM_BATCH_MANAGER_LIB CACHE)
        find_library(TRTLLM_BATCH_MANAGER_LIB NAMES libtensorrt_llm_batch_manager_static.a PATHS ${TRTLLM_CPP_DIR}/tensorrt_llm/batch_manager/aarch64-linux-gnu)
        message(STATUS "TRTLLM_BATCH_MANAGER_LIB path: " ${TRTLLM_BATCH_MANAGER_LIB})
        unset(TRTLLM_INCLUDE_DIR CACHE)
        set(TRTLLM_INCLUDE_DIR ${TRTLLM_CPP_DIR}/include)

        add_executable(harness_llm
            harness_llm/main_llm.cc
            harness_llm/llm_server.cc
            harness_llm/llm_core.cc
            common/logger.cpp
        )

        target_link_libraries(harness_llm
            nvinfer
            nvinfer_plugin
            gflags
            glog
            ${MPI_C_LIBRARIES}
            ${MPI_CXX_LIBRARIES}
            ${TRTLLM_LIB}
            ${TRTLLM_PLUGIN_LIB}
            ${TRTLLM_BATCH_MANAGER_LIB}
            ${CUDA_LIBRARIES}
            ${LOADGEN_LIB}
        )

        target_include_directories(harness_llm
            PUBLIC
                ${MPI_C_INCLUDE_PATH}
                ${CUDA_INCLUDE_DIRS}
                ${LOADGEN_INCLUDE_DIR}
                ${LWIS_INCLUDE_DIR}
                ${TRTLLM_CPP_DIR}
                ${TRTLLM_INCLUDE_DIR}
                common
                harness_llm
        )
    else()
        execute_process(COMMAND echo "Building LLM harness...")
        find_package(MPI REQUIRED)
        # Set the path to the NCCL
        unset(NCCL_LIB CACHE)
        find_library(NCCL_LIB NAMES libnccl.so PATHS /usr/lib/)
        unset(NCCL_INCLUDE_DIRS CACHE)
        set(NCCL_INCLUDE_DIRS /usr/include/nccl)
        # Set the path to the TRTLLM
        unset(TRTLLM_CPP_DIR CACHE)
        set(TRTLLM_CPP_DIR ${TRTLLM_DIR}/cpp)
        message(STATUS "TRTLLM_CPP_DIR path: " ${TRTLLM_CPP_DIR})
        unset(TRTLLM_LIB CACHE)
        find_library(TRTLLM_LIB NAMES libtensorrt_llm.so PATHS ${TRTLLM_CPP_DIR}/build/tensorrt_llm/)
        message(STATUS "TRTLLM_LIB path: " ${TRTLLM_LIB})
        unset(TRTLLM_PLUGIN_LIB CACHE)
        find_library(TRTLLM_PLUGIN_LIB NAMES libnvinfer_plugin_tensorrt_llm.so PATHS ${TRTLLM_CPP_DIR}/build/tensorrt_llm/plugins)
        message(STATUS "TRTLLM_PLUGIN_LIB path: " ${TRTLLM_PLUGIN_LIB})
        # The batch manager is pre-built in the release TRTLLM
        if (${USE_RELEASE_TRTLLM})
            # TODO: Enable aarch64
            if (NOT ${ARCH} MATCHES "aarch64")
                set(TRTLLM_BATCH_MANAGER_LIB_PATH ${TRTLLM_CPP_DIR}/tensorrt_llm/batch_manager/x86_64-linux-gnu)
            else()
                set(TRTLLM_BATCH_MANAGER_LIB_PATH ${TRTLLM_CPP_DIR}/tensorrt_llm/batch_manager/aarch64-linux-gnu)
            endif()
        else()
            set(TRTLLM_BATCH_MANAGER_LIB_PATH ${TRTLLM_CPP_DIR}/build/tensorrt_llm/batch_manager)
        endif()
        unset(TRTLLM_BATCH_MANAGER_LIB CACHE)
        find_library(TRTLLM_BATCH_MANAGER_LIB NAMES libtensorrt_llm_batch_manager_static.a PATHS ${TRTLLM_BATCH_MANAGER_LIB_PATH})
        message(STATUS "TRTLLM_BATCH_MANAGER_LIB path: " ${TRTLLM_BATCH_MANAGER_LIB})
        unset(TRTLLM_INCLUDE_DIR CACHE)
        set(TRTLLM_INCLUDE_DIR ${TRTLLM_CPP_DIR}/include)

        add_executable(harness_llm
            harness_llm/main_llm.cc
            harness_llm/llm_server.cc
            harness_llm/llm_core.cc
            common/logger.cpp
        )

        target_link_libraries(harness_llm
            nvinfer
            nvinfer_plugin
            gflags
            glog
            ${MPI_C_LIBRARIES}
            ${MPI_CXX_LIBRARIES}
            ${NCCL_LIB}
            ${TRTLLM_LIB}
            ${TRTLLM_PLUGIN_LIB}
            ${TRTLLM_BATCH_MANAGER_LIB}
            ${CUDA_LIBRARIES}
            ${LOADGEN_LIB}
        )

        target_include_directories(harness_llm
            PUBLIC
                ${MPI_C_INCLUDE_PATH}
                ${NCCL_INCLUDE_DIRS}
                ${CUDA_INCLUDE_DIRS}
                ${LOADGEN_INCLUDE_DIR}
                ${LWIS_INCLUDE_DIR}
                ${TRTLLM_CPP_DIR}
                ${TRTLLM_INCLUDE_DIR}
                common
                harness_llm
        )
    endif()
endif()
