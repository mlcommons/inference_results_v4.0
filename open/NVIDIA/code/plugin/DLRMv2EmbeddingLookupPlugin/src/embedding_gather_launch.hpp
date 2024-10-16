/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

// #define DEBUG_PRINT

enum IOType : int
{
    FLOAT = 1,
    HALF = 2,
    INT8 = 3
};

enum IO2Types : int
{
    EMBED_DENSE_F32F32 = (FLOAT) | (FLOAT << 4),
    EMBED_DENSE_H16H16 = (HALF) | (HALF << 4),
    EMBED_DENSE_I8I8 = (INT8) | (INT8 << 4),

    EMBED_DENSE_I8F32 = (INT8) | (FLOAT << 4),
    EMBED_DENSE_I8H16 = (INT8) | (HALF << 4),
};

int run_mega_embedding_gather(cudaStream_t stream,
    /* the indices.
     * currently using int32_t type for dlrmv2
     */
    const int* sparse_input, const int* index_remap, const int* index_hotnesses, const int* index_offset,

    IOType embed_iotype, /* table/output type*/
    IOType dense_iotype, /* dense input  type*/

    /* the tables
     * dense_input is also viewed as a separate table
     */
    const void* dense_input, const void* mega_table, const void* mega_table_host,

    void* output,

    int batch_size, int embed_dim, /*the table width(element)*/
    int embed_feature_total,       /*the number of categorical features*/
    int embed_hotness_total,       /*the sum of hotness among categorical features*/
    int embed_rows_gpu,            /*the number of mega table rows on gpu*/

    const float* scales_gpu, const float* scales_inv_gpu);
