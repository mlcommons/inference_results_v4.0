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

# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(DLRMv2EmbeddingLookupPluginDLRMv2 LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")
set(CMAKE_CXX_STANDARD 17)

# -------- CONFIGURATION --------
if (NOT DEFINED $ENV{TRT_LIB_DIR})
  set(TRT_LIB_DIR /usr/lib/x86_64-linux-gnu)
else()
  set(TRT_LIB_DIR $ENV{TRT_LIB_DIR})
endif()
if (NOT DEFINED $ENV{TRT_INCLUDE_DIR})
  set(TRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
else()
  set(TRT_INCLUDE_DIR $ENV{TRT_INCLUDE_DIR})
endif()
if (NOT DEFINED $ENV{CUB_DIR})
  set(CUB_DIR /usr/local/cub)
else()
  set(CUB_DIR $ENV{CUB_DIR})
endif()
if (NOT DEFINED $ENV{CUDNN_LIB_DIR})
  set(CUDNN_LIB_DIR /usr/lib/x86_64-linux-gnu)
else()
  set(CUDNN_LIB_DIR $ENV{CUDNN_LIB_DIR})
endif()
if (NOT DEFINED $ENV{CUDNN_INC_DIR})
  set(CUDNN_INC_DIR /usr/include/x86_64-linux-gnu)
else()
  set(CUDNN_INC_DIR $ENV{CUDNN_INC_DIR})
endif()
if (NOT DEFINED $ENV{CUDA_INC_DIR})
  set(CUDA_INC_DIR /usr/local/cuda/include)
else()
  set(CUDA_INC_DIR $ENV{CUDA_INC_DIR})
endif()

set(LWIS_INCLUDE_DIR ../../harness/lwis/include)

enable_language(CUDA)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
string(REPLACE ":" ";" LIBRARY_DIRS $ENV{LD_LIBRARY_PATH})
find_library(NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR} ${LIBRARY_DIRS} PATH_SUFFIXES lib lib64)
find_library(CUDNN_LIB cudnn HINTS ${CUDNN_LIB_DIR} ${LIBRARY_DIRS} PATH_SUFFIXES lib lib64)

# -------- BUILDING --------

# Add include directories
include_directories(${CUDA_INC_DIR} ${TRT_INCLUDE_DIR} ${CUB_DIR} ${CUDNN_INC_DIR} ${LWIS_INCLUDE_DIR}
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                    ${CMAKE_SOURCE_DIR}/src/)

# Define dlrmv2 embedding lookup plugin library target
FILE(GLOB SOURCE_CU_FILES ${CMAKE_SOURCE_DIR}/src/*.cu)
FILE(GLOB SOURCE_CPP_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
FILE(GLOB HEADER_FILES ${CMAKE_SOURCE_DIR}/src/*.h)
add_library(dlrmv2embeddinglookupplugin MODULE
  ${SOURCE_CU_FILES}
  ${SOURCE_CPP_FILES}
  ${HEADER_FILES}
)

# Use C++11
target_compile_features(dlrmv2embeddinglookupplugin PUBLIC cxx_std_11)

# Set sm versions
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.0)
  string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_80,code=sm_80")
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.1)
  string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_86,code=sm_86")
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.8)
  string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_89,code=sm_89")
  string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_90,code=sm_90")
endif()

# Link TensorRT's nvinfer lib
target_link_libraries(dlrmv2embeddinglookupplugin PUBLIC ${NVINFER_LIB} glog)
target_link_libraries(dlrmv2embeddinglookupplugin PUBLIC ${CUDNN_LIB})
