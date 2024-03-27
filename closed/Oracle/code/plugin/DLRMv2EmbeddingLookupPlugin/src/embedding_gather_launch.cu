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
#include <cstdio>

#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

#include "embedding_gather_kernel.cuh"
#include "embedding_gather_kernel_dt_hybrid.cuh"
#include "embedding_gather_launch.hpp"
#include "launch_host_init.hpp"

int run_mega_embedding_gather(cudaStream_t stream,
    /* the indices.
     * current always using int32_t type
     */
    const int* sparse_input, const int* index_remap, const int* index_hotnesses, const int* index_offsets,

    IOType embed_iotype, IOType dense_iotype,

    /* the tables
     * dense_input is also viewed as a separate table
     */
    const void* dense_input, const void* mega_table, const void* mega_table_host,

    void* output,

    int batch_size, int embed_dim, /*the table width(element)*/
    int embed_feature_total,       /*the number of categorical features*/
    int embed_hotness_total, int embed_rows_gpu,

    const float* scales, const float* scales_inv)
{

    using IndexType = int;
    const int BYTES_PER_INDEX_ELT = sizeof(IndexType);

    const int32_t dt_cases = (embed_iotype) | (dense_iotype << 4);

    if (dt_cases == IO2Types::EMBED_DENSE_F32F32)
    {
        using EmbedType = float;

        const int BYTES_PER_EMBED_ELT = sizeof(float);
        const int NUM_SAMPLES_PER_CTA = 2;
        const int BYTES_LDST_EMBED = 16;

        LAUNCH_HOST_INIT;
        PRINT_HOST_INIT;

        if (index_remap == nullptr)
        {
            LAUNCH_WO_INDEX_REMAP;
        }
        else
        {
            LAUNCH_INDEX_REMAP;
        }
    }
    else if (dt_cases == IO2Types::EMBED_DENSE_H16H16)
    {
        using EmbedType = half;

        const int BYTES_PER_EMBED_ELT = sizeof(half);
        const int NUM_SAMPLES_PER_CTA = 4;
        const int BYTES_LDST_EMBED = 16;

        LAUNCH_HOST_INIT;
        PRINT_HOST_INIT;

        if (index_remap == nullptr)
        {
            LAUNCH_WO_INDEX_REMAP;
        }
        else
        {
            LAUNCH_INDEX_REMAP;
        }
    }
    else if (dt_cases == IO2Types::EMBED_DENSE_I8I8)
    {
        using EmbedType = int8_t;

        const int BYTES_PER_EMBED_ELT = sizeof(int8_t);
        const int NUM_SAMPLES_PER_CTA = 8;
        const int BYTES_LDST_EMBED = 16;

        // const int NUM_SAMPLES_PER_CTA = 4;
        // const int BYTES_LDST_EMBED    = 8;

        // const int NUM_SAMPLES_PER_CTA = 2;
        // const int BYTES_LDST_EMBED    = 4;

        LAUNCH_HOST_INIT;
        PRINT_HOST_INIT;

        assert(scales != nullptr);
        assert(scales_inv != nullptr);

        if (index_remap == nullptr)
        {
            LAUNCH_WO_INDEX_REMAP;
        }
        else
        {
            LAUNCH_INDEX_REMAP;
        }
    }
    else if (dt_cases == IO2Types::EMBED_DENSE_I8F32)
    {
        using EmbedType = int8_t;
        using DenseType = float;

        const int BYTES_PER_EMBED_ELT = sizeof(int8_t);
        const int NUM_SAMPLES_PER_CTA = 8;
        const int BYTES_LDST_EMBED = 16;

        LAUNCH_HOST_INIT;
        PRINT_HOST_INIT;

        const int BYTES_PER_DENSE_ELT = sizeof(float);
        const int BYTES_LD_DENSE = 16;

        int dense_dim_size = embed_dim * BYTES_PER_DENSE_ELT;

        assert(scales != nullptr);
        assert(scales_inv != nullptr);

        if (index_remap == nullptr)
        {
            LAUNCH_WO_INDEX_REMAP_EMBED_DENSE_HYBRID;
        }
        else
        {
            LAUNCH_INDEX_REMAP_EMBED_DENSE_HYBRID;
        }
    }
    else if (dt_cases == IO2Types::EMBED_DENSE_I8H16)
    {
        using EmbedType = int8_t;
        using DenseType = half;

        const int BYTES_PER_EMBED_ELT = sizeof(int8_t);
        const int NUM_SAMPLES_PER_CTA = 8;
        const int BYTES_LDST_EMBED = 16;

        LAUNCH_HOST_INIT;
        PRINT_HOST_INIT;

        const int BYTES_PER_DENSE_ELT = sizeof(half);
        const int BYTES_LD_DENSE = 16;

        int dense_dim_size = embed_dim * BYTES_PER_DENSE_ELT;

        assert(scales != nullptr);
        assert(scales_inv != nullptr);

        if (index_remap == nullptr)
        {
            LAUNCH_WO_INDEX_REMAP_EMBED_DENSE_HYBRID;
        }
        else
        {
            LAUNCH_INDEX_REMAP_EMBED_DENSE_HYBRID;
        }
    }
    else
    {
        assert(false);
    }

    return 0;
}