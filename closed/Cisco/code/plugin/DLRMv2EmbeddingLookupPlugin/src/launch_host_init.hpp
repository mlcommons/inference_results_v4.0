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

#include <cassert>

#define LAUNCH_HOST_INIT                                                                                               \
    int num_samples_per_cta = NUM_SAMPLES_PER_CTA;                                                                     \
                                                                                                                       \
    assert(batch_size % num_samples_per_cta == 0);                                                                     \
    assert((embed_dim * BYTES_PER_EMBED_ELT) % BYTES_LDST_EMBED == 0);                                                 \
                                                                                                                       \
    dim3 launch_grid(batch_size / num_samples_per_cta, 1, 1);                                                          \
    dim3 launch_block((embed_dim * BYTES_PER_EMBED_ELT) / BYTES_LDST_EMBED, num_samples_per_cta, 1);                   \
                                                                                                                       \
    int smem_size = (embed_feature_total * BYTES_PER_INDEX_ELT * (1 + 1)) + /*hotness and offsets; same for all cta*/  \
        (embed_hotness_total * BYTES_PER_INDEX_ELT * num_samples_per_cta);  /*indices*/                                \
                                                                                                                       \
    int embed_dim_size = embed_dim * BYTES_PER_EMBED_ELT;

#define LAUNCH_WO_INDEX_REMAP                                                                                          \
    mega_embedding_gather<false, EmbedType, IndexType, BYTES_LDST_EMBED,                                               \
        DivUp<BYTES_LDST_EMBED, BYTES_PER_EMBED_ELT>::VALUE><<<launch_grid, launch_block, smem_size, stream>>>(        \
        reinterpret_cast<const char*>(sparse_input), reinterpret_cast<const char*>(index_remap),                       \
        reinterpret_cast<const char*>(index_hotnesses), reinterpret_cast<const char*>(index_offsets),                  \
                                                                                                                       \
        reinterpret_cast<const char*>(dense_input), reinterpret_cast<const char*>(mega_table),                         \
        reinterpret_cast<const char*>(mega_table_host),                                                                \
                                                                                                                       \
        reinterpret_cast<char*>(output),                                                                               \
                                                                                                                       \
        embed_feature_total, embed_hotness_total, embed_rows_gpu, embed_dim_size,                                      \
                                                                                                                       \
        scales, scales_inv);

#define LAUNCH_INDEX_REMAP                                                                                             \
    mega_embedding_gather<true, EmbedType, IndexType, BYTES_LDST_EMBED,                                                \
        DivUp<BYTES_LDST_EMBED, BYTES_PER_EMBED_ELT>::VALUE><<<launch_grid, launch_block, smem_size, stream>>>(        \
        reinterpret_cast<const char*>(sparse_input), reinterpret_cast<const char*>(index_remap),                       \
        reinterpret_cast<const char*>(index_hotnesses), reinterpret_cast<const char*>(index_offsets),                  \
                                                                                                                       \
        reinterpret_cast<const char*>(dense_input), reinterpret_cast<const char*>(mega_table),                         \
        reinterpret_cast<const char*>(mega_table_host),                                                                \
                                                                                                                       \
        reinterpret_cast<char*>(output),                                                                               \
                                                                                                                       \
        embed_feature_total, embed_hotness_total, embed_rows_gpu, embed_dim_size,                                      \
                                                                                                                       \
        scales, scales_inv);

#define LAUNCH_WO_INDEX_REMAP_EMBED_DENSE_HYBRID                                                                       \
    mega_embedding_gather<false, EmbedType, DenseType, IndexType, BYTES_LDST_EMBED,                                    \
        DivUp<BYTES_LDST_EMBED, BYTES_PER_EMBED_ELT>::VALUE, BYTES_LD_DENSE,                                           \
        DivUp<BYTES_LD_DENSE, BYTES_PER_DENSE_ELT>::VALUE><<<launch_grid, launch_block, smem_size, stream>>>(          \
        reinterpret_cast<const char*>(sparse_input), reinterpret_cast<const char*>(index_remap),                       \
        reinterpret_cast<const char*>(index_hotnesses), reinterpret_cast<const char*>(index_offsets),                  \
                                                                                                                       \
        reinterpret_cast<const char*>(dense_input), reinterpret_cast<const char*>(mega_table),                         \
        reinterpret_cast<const char*>(mega_table_host),                                                                \
                                                                                                                       \
        reinterpret_cast<char*>(output),                                                                               \
                                                                                                                       \
        embed_feature_total, embed_hotness_total, embed_rows_gpu, embed_dim_size, dense_dim_size,                      \
                                                                                                                       \
        scales, scales_inv);

#define LAUNCH_INDEX_REMAP_EMBED_DENSE_HYBRID                                                                          \
    mega_embedding_gather<true, EmbedType, DenseType, IndexType, BYTES_LDST_EMBED,                                     \
        DivUp<BYTES_LDST_EMBED, BYTES_PER_EMBED_ELT>::VALUE, BYTES_LD_DENSE,                                           \
        DivUp<BYTES_LD_DENSE, BYTES_PER_DENSE_ELT>::VALUE><<<launch_grid, launch_block, smem_size, stream>>>(          \
        reinterpret_cast<const char*>(sparse_input), reinterpret_cast<const char*>(index_remap),                       \
        reinterpret_cast<const char*>(index_hotnesses), reinterpret_cast<const char*>(index_offsets),                  \
                                                                                                                       \
        reinterpret_cast<const char*>(dense_input), reinterpret_cast<const char*>(mega_table),                         \
        reinterpret_cast<const char*>(mega_table_host),                                                                \
                                                                                                                       \
        reinterpret_cast<char*>(output),                                                                               \
                                                                                                                       \
        embed_feature_total, embed_hotness_total, embed_rows_gpu, embed_dim_size, dense_dim_size,                      \
                                                                                                                       \
        scales, scales_inv);

#ifdef DEBUG_PRINT
#define PRINT_HOST_INIT                                                                                                \
    printf("batch_size:           %d\n", batch_size);                                                                  \
    printf("embed_dim:            %d\n", embed_dim);                                                                   \
    printf("embed_feature_total:  %d\n", embed_feature_total);                                                         \
    printf("embed_hotness_total:  %d\n", embed_hotness_total);                                                         \
    printf("embed_rows_gpu:       %d\n", embed_rows_gpu);                                                              \
    printf("launch_grid:  %10d %5d %5d\n", launch_grid.x, launch_grid.y, launch_grid.z);                               \
    printf("launch_block: %10d %5d %5d\n", launch_block.x, launch_block.y, launch_block.z);                            \
    printf("smem_size:           %d\n", smem_size);                                                                    \
    printf("BYTES_PER_INDEX_ELT: %10d\n", BYTES_PER_INDEX_ELT);                                                        \
    printf("BYTES_PER_EMBED_ELT: %10d\n", BYTES_PER_EMBED_ELT);                                                        \
    printf("BYTES_LDST_EMBED:    %10d\n", BYTES_LDST_EMBED);
#else
#define PRINT_HOST_INIT
#endif
