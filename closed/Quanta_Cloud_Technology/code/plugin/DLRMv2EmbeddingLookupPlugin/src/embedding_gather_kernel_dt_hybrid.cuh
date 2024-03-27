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

#include <cstdio>

#include <cuda_pipeline_primitives.h>

#include "common_ptx.cuh"
#include "smem_staging.cuh"

template <bool INDEX_REMAP, typename Embed_t, /*table type and output type*/
    typename Dense_t,                         /*dense input type*/
    typename Index_t,

    int BYTES_LDST_EMBED, int ELETS_PER_LDST_EMBED,

    int BYTES_LD_DENSE, int ELETS_PER_LD_DENSE>
__launch_bounds__(64) __global__ void mega_embedding_gather(
    const char* __restrict__ sparse_input, const char* __restrict__ index_remap, /*optional*/
    const char* __restrict__ index_hotness, const char* __restrict__ index_offsets,

    const char* __restrict__ dense_input, const char* __restrict__ mega_table, const char* __restrict__ mega_table_host,

    char* __restrict__ output,

    int embed_feature_total, int embed_hotness_total, int embed_rows_gpu, int embed_dim_size,
    int dense_dim_size, /*size of 1 dense feature of a sample in bytes*/

    const float* __restrict__ scales, const float* __restrict__ scales_inv /*scales_inv[-1] is for fp32 dense input*/
)
{

    extern __shared__ char smem_buff[];

    const int BYTES_PER_INDEX = sizeof(Index_t);

    const int tnum = blockDim.x * blockDim.y;                      /* total threads in cta*/
    const int snum = blockDim.y;                                   /* total samples in cta*/
    const int tidx_ldi = (blockDim.x * threadIdx.y) + threadIdx.x; /* thread 1D index for ldgsts index*/
    const int tidx_lde = threadIdx.x;
    const int sidx = threadIdx.y; /* sample idx in cta*/
    const int cidx = blockIdx.x;  /* cta idx*/

    const uint32_t smem_hotness = get_smem_pointer(smem_buff);
    const uint32_t smem_offsets = smem_hotness + embed_feature_total * BYTES_PER_INDEX;
    const uint32_t smem_indices = smem_offsets + embed_feature_total * BYTES_PER_INDEX;

    const char* gmem_base_indices = sparse_input + (BYTES_PER_INDEX * embed_hotness_total) * (snum * cidx);

    char* gmem_base_output
        = output + (embed_dim_size * (embed_feature_total + 1)) * (snum * cidx + sidx); // plus 1 for dense

    // dense feature start
    Embed_t local_embed[ELETS_PER_LDST_EMBED];
    Dense_t local_dense_chunk[ELETS_PER_LD_DENSE];
    const int NUM_DENSE_CHUNKS = DivUp<ELETS_PER_LDST_EMBED, ELETS_PER_LD_DENSE>::VALUE;
    const float dense_scale_inv = scales_inv[embed_feature_total]; // the last is for dense input;

#ifdef DEBUG_PRINT
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
    {
        printf("dense_scale_inv: %8.1f\n", dense_scale_inv);
    }
#endif

#pragma unroll
    for (int ci = 0; ci < NUM_DENSE_CHUNKS; ++ci)
    {
        ldg<BYTES_LD_DENSE>(reinterpret_cast<char*>(&local_dense_chunk[0]),
            dense_input + ((uint64_t) (snum * cidx + sidx) * (uint64_t) dense_dim_size)
                + (tidx_lde * NUM_DENSE_CHUNKS * BYTES_LD_DENSE) + ci * BYTES_LD_DENSE);

        // currently designed for fp32/ldg.16. ELETS_PER_LD_DENSE == 4;
        //  if (ELETS_PER_LD_DENSE == 4) {
        //    float4& local_dense_chunk_float4 = *(reinterpret_cast<float4*>(&local_dense_chunk[0]));

        //   local_dense_chunk_float4.x *= dense_scale_inv;
        //   local_dense_chunk_float4.y *= dense_scale_inv;
        //   local_dense_chunk_float4.z *= dense_scale_inv;
        //   local_dense_chunk_float4.w *= dense_scale_inv;

        //   uint32_t& local_embed_s8x4 = *(reinterpret_cast<uint32_t*>(&local_embed[ci * 4]));
        //   local_embed_s8x4 = float4_to_s8x4(local_dense_chunk_float4);
        // } else {
#pragma unroll 1
        for (int ei = 0; ei < ELETS_PER_LD_DENSE; ++ei)
        {
            float local_dense_elt = static_cast<float>(local_dense_chunk[ei]);

#ifdef DEBUG_PRINT
            if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
            {
                printf("%8.1f ", local_dense_elt);
            }
#endif
            local_dense_elt *= dense_scale_inv;
            local_embed[ci * ELETS_PER_LD_DENSE + ei] = f2i_s8(local_dense_elt);
        }

#ifdef DEBUG_PRINT
        if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
        {
            printf("\n");
        }
#endif

        // }
    }
    stg<BYTES_LDST_EMBED>(gmem_base_output + (tidx_lde * BYTES_LDST_EMBED),
        reinterpret_cast<const char*>(&local_embed[0])); // dense | sparse
    // dense feature end

    // sparse feature start

    /*the alignment is not ensured, so just use ldgsts.32*/
    const int BYTES_PER_LDGSTS = BYTES_PER_INDEX;
    stream_gmem_to_smem<BYTES_PER_LDGSTS>(index_hotness, smem_hotness, embed_feature_total, tidx_ldi, tnum);
    stream_gmem_to_smem<BYTES_PER_LDGSTS>(index_offsets, smem_offsets, embed_feature_total, tidx_ldi, tnum);
    stream_gmem_to_smem<BYTES_PER_LDGSTS>(gmem_base_indices, smem_indices, embed_hotness_total * snum, tidx_ldi, tnum);

    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

#ifdef DEBUG_PRINT
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
    {
        printf("embed_feature_total: %d\n", embed_feature_total);
        printf("embed_hotness_total: %d\n", embed_hotness_total);
    }
#endif

    uint32_t ft_smem_indices = smem_indices + (BYTES_PER_INDEX * embed_hotness_total * sidx);

#pragma unroll 1
    for (int ft_i = 0; ft_i < embed_feature_total; ++ft_i)
    {

        Index_t ft_hotness = 0;
        Index_t ft_offset = 0;

        lds<BYTES_PER_INDEX>(reinterpret_cast<char*>(&ft_hotness), smem_hotness + BYTES_PER_INDEX * ft_i);
        lds<BYTES_PER_INDEX>(reinterpret_cast<char*>(&ft_offset), smem_offsets + BYTES_PER_INDEX * ft_i);

#ifdef DEBUG_PRINT
        if (std::is_same<Embed_t, int8_t>::value)
        {
            if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
            {
                printf("table%2d nnz: %4d offset: %10d gpu_row: %10d scale: %f scale_inv: %f\n", ft_i, ft_hotness,
                    ft_offset, embed_rows_gpu, scales[ft_i], scales_inv[ft_i]);
            }
        }
        else
        {
            if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
            {
                printf("table%2d nnz: %4d offset: %10d gpu_row: %10d\n", ft_i, ft_hotness, ft_offset, embed_rows_gpu);
            }
        }
#endif

        float local_acc[ELETS_PER_LDST_EMBED];
        clear<ELETS_PER_LDST_EMBED>(reinterpret_cast<uint32_t*>(&local_acc[0]));

        // embed start
        // #pragma unroll 2
        for (int id = 0; id < ft_hotness; ++id)
        {

            Index_t index = 0;
            lds<BYTES_PER_INDEX>(reinterpret_cast<char*>(&index), ft_smem_indices);

            index += ft_offset; /*add offset to get the index to the mega table*/

            Embed_t local_embed[ELETS_PER_LDST_EMBED];
            if (INDEX_REMAP)
            {
                // remap to pos
                Index_t index_pos = 0;
                ldg<BYTES_PER_INDEX>(reinterpret_cast<char*>(&index_pos), index_remap + index * BYTES_PER_INDEX);

                if (index_pos < embed_rows_gpu)
                {
                    ldg<BYTES_LDST_EMBED>(reinterpret_cast<char*>(&local_embed[0]),
                        mega_table + ((uint64_t) index_pos * (uint64_t) embed_dim_size)
                            + (tidx_lde * BYTES_LDST_EMBED));
                }
                else
                {
                    index_pos -= embed_rows_gpu; /*roll back to the start of mega table on host mem*/
                    ldg<BYTES_LDST_EMBED>(reinterpret_cast<char*>(&local_embed[0]),
                        mega_table_host + ((uint64_t) index_pos * (uint64_t) embed_dim_size)
                            + (tidx_lde * BYTES_LDST_EMBED));
                }
            }
            else
            {
                ldg<BYTES_LDST_EMBED>(reinterpret_cast<char*>(&local_embed[0]),
                    mega_table + ((uint64_t) index * (uint64_t) embed_dim_size) + (tidx_lde * BYTES_LDST_EMBED));
            }

            if (std::is_same<Embed_t, int8_t>::value)
            {
// #pragma unroll ELETS_PER_LDST_EMBED
// for (int ei = 0; ei < ELETS_PER_LDST_EMBED; ++ei)
// {
//   // local_acc[ei] += static_cast<float>(local_embed[ei]) * scales[ft_i];
//   local_acc[ei] += i2f_rn(local_embed[ei]) * scales[ft_i];
// }
#pragma unroll
                for (int ei = 0; ei < DivUp<ELETS_PER_LDST_EMBED, 4>::VALUE; ++ei)
                {
                    float4& local_acc_float4 = *(reinterpret_cast<float4*>(&local_acc[ei * 4]));
                    const uint32_t& local_embed_s8x4 = *(reinterpret_cast<uint32_t*>(&local_embed[ei * 4]));

                    float4 local_embed_float4 = s8x4_to_float4(local_embed_s8x4);

                    local_acc_float4.x += local_embed_float4.x * scales[ft_i];
                    local_acc_float4.y += local_embed_float4.y * scales[ft_i];
                    local_acc_float4.z += local_embed_float4.z * scales[ft_i];
                    local_acc_float4.w += local_embed_float4.w * scales[ft_i];
                }
            }
            else
            {
#pragma unroll
                for (int ei = 0; ei < ELETS_PER_LDST_EMBED; ++ei)
                {
                    local_acc[ei] += static_cast<float>(local_embed[ei]);
                }
            }
            // acc current fetched row

            ft_smem_indices += BYTES_PER_INDEX; // next index
        }
        // embed end
#ifdef DEBUG_PRINT
        for (int ei = 0; ei < ELETS_PER_LDST_EMBED; ++ei)
        {
            if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
            {
                printf("%8.1f ", static_cast<float>(local_acc[ei]));
            }
        }
#endif

        // write out
        Embed_t local_output[ELETS_PER_LDST_EMBED];
        if (std::is_same<Embed_t, int8_t>::value)
        {
            // #pragma unroll
            // for (int ei = 0; ei < ELETS_PER_LDST_EMBED; ++ei) {
            // 	// local_output[ei] = static_cast<Embed_t>(local_acc[ei] * scales_inv[ft_i]);
            // 	local_output[ei] = f2i_s8(local_acc[ei] * scales_inv[ft_i]);
            // }

#pragma unroll
            for (int ei = 0; ei < DivUp<ELETS_PER_LDST_EMBED, 4>::VALUE; ++ei)
            {
                float4& local_acc_float4 = *(reinterpret_cast<float4*>(&local_acc[ei * 4]));

                local_acc_float4.x *= scales_inv[ft_i];
                local_acc_float4.y *= scales_inv[ft_i];
                local_acc_float4.z *= scales_inv[ft_i];
                local_acc_float4.w *= scales_inv[ft_i];

                uint32_t& local_output_s8x4 = *(reinterpret_cast<uint32_t*>(&local_output[ei * 4]));
                local_output_s8x4 = float4_to_s8x4(local_acc_float4);
            }
        }
        else
        {
#pragma unroll
            for (int ei = 0; ei < ELETS_PER_LDST_EMBED; ++ei)
            {
                local_output[ei] = static_cast<Embed_t>(local_acc[ei]);
            }
        }

#ifdef DEBUG_PRINT
        if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
        {
            printf("\n");
        }
#endif

        // dense  | sparse
        stg<BYTES_LDST_EMBED>(gmem_base_output + ((ft_i + 1) * embed_dim_size) + (tidx_lde * BYTES_LDST_EMBED),
            reinterpret_cast<const char*>(&local_output[0]));
    }
    // feature ends
}
