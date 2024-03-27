/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <cuda_fp16.h>
#include <stdio.h>

#ifndef DENSE_INPUT_TYPE
#define DENSE_INPUT_TYPE half // int8_t // float
#endif

#define THREADBLOCK_SIZE 512

__launch_bounds__(THREADBLOCK_SIZE) __global__ void gatherKernel(
    const DENSE_INPUT_TYPE** __restrict numericalInputPtrBuffer,
    const int32_t** __restrict categoricalInputPtrBuffer,
    const size_t* __restrict sampleSizesBuf,
    const size_t* __restrict sampleOffsetsBuf,
    DENSE_INPUT_TYPE* __restrict numericalOutputBuf,
    int32_t* __restrict categoricalOutputBuf,
    int sampleCount,
    int numericVolume,
    int categoricalVolume)
{
    int sampleId = blockIdx.x;
    int laneId = threadIdx.x;
    const DENSE_INPUT_TYPE* numericalInputBuffer = numericalInputPtrBuffer[sampleId];
    const int32_t* categoricalInputBuffer = categoricalInputPtrBuffer[sampleId];
    int sampleSize = sampleSizesBuf[sampleId];
    int sampleOffset = sampleOffsetsBuf[sampleId];

    int numericalElems = sampleSize * numericVolume;
    DENSE_INPUT_TYPE* numericDstBuf = numericalOutputBuf + sampleOffset * numericVolume;
    for (int elemId = laneId; elemId < numericalElems; elemId += THREADBLOCK_SIZE)
    {
        numericDstBuf[elemId] = __ldg(numericalInputBuffer + elemId);
    }

    int categoricalElems = sampleSize * categoricalVolume;
    int32_t* categoricalDstBuf = categoricalOutputBuf + sampleOffset * categoricalVolume;
    for (int elemId = laneId; elemId < categoricalElems; elemId += THREADBLOCK_SIZE)
    {
        categoricalDstBuf[elemId] = __ldg(categoricalInputBuffer + elemId);
    }
}

void runGatherKernel(
    const DENSE_INPUT_TYPE** numericalInputPtrBuffer,
    const int32_t** categoricalInputPtrBuffer,
    const size_t* sampleSizesBuf,
    const size_t* sampleOffsetsBuf,
    DENSE_INPUT_TYPE* numericalOutputBuf,
    int32_t* categoricalOutputBuf,
    int sampleCount,
    int numericVolume,
    int categoricalVolume,
    cudaStream_t stream)
{
    gatherKernel<<<sampleCount, THREADBLOCK_SIZE, 0, stream>>>(numericalInputPtrBuffer, categoricalInputPtrBuffer,
        sampleSizesBuf, sampleOffsetsBuf, numericalOutputBuf, categoricalOutputBuf, sampleCount, numericVolume,
        categoricalVolume);
    assert(cudaGetLastError() == cudaSuccess);
}
