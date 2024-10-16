/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <vector>
#include <cstring>
#include <stdint.h>
#include <iostream>

#include <fstream>

#include <cuda_fp16.h>

#include "ssdOpt.h"
#include "ssdOptMacros.h"
#include "fast_divmod.h"
//#include "ssd_internal.h"

// #include <cooperative_groups/memcpy_async.h>
// #include <cuda/pipeline>

// C-API for the async copy
#include <cuda_pipeline.h>

inline __device__ void ldgsts(uint32_t dst, const void *src, bool p = true) {            
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800                                                 
  uint32_t m = p ? 16u : 0u;                                                                     
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(m)); 
#endif                                                                                             
}                                                                                           
 
inline __device__ void ldgdepbar() {                                                               
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800                
  asm volatile("cp.async.commit_group;\n" ::);                                                     
#endif                                                                                             
}                    
 
template< int N >                                                                                  
inline __device__ void depbar() {                                                                  
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800                
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));                                             
#endif                                                                                             
}  

template <typename T>
void saveDeviceBuffer_(const T* buf, size_t count, std::ofstream& output)
{
    T* hBuf = new T[count];
    cudaDeviceSynchronize();
    cudaMemcpy(hBuf, buf, count * sizeof(T), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 1000; i ++) {
        printf("%zu ", i);
        printf("%f ", hBuf[i]);
        printf("\n");
    }

    output.write((char *)hBuf, count * sizeof(T));
    delete [] hBuf;
}

namespace nvinfer1
{
namespace plugin
{

template <typename Dtype, int NUM_LAYERS>
struct PermuteConfData {
    const Dtype * conf_data[NUM_LAYERS];
    int feature_size[NUM_LAYERS];
    int num_anchors[NUM_LAYERS];
    int end_layer_prior[NUM_LAYERS];
    int box_channels[NUM_LAYERS];
    // support reduced math
    uint32_t mul[NUM_LAYERS];
    uint32_t shr[NUM_LAYERS];
    bool packed32_nchw;
    bool permute_before_reshape;
    bool concatInputs;
};

#if 0

#include <cublas_v2.h>
#include "nms_common.h"

template <int TILE_X, int TILE_Y, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void batched_transpose_kernel(const float* data_in, float* data_out, const int num_classes, const int num_priors){
    // num_classes is "fast" dimension in data_in
    // BLOCK_DIM_X == TILE_X

    constexpr int SMEM_STRIDE = TILE_X + 1;
    __shared__ float smem[SMEM_STRIDE * TILE_Y];

    // batch id
    const int n = blockIdx.z;
    const int n_offset = n * num_classes * num_priors;

    const int tile_x_in_offset = blockIdx.x * TILE_X;
    const int tile_y_in_offset = blockIdx.y * TILE_Y;
    
    const float* data_in_tile = data_in + n_offset + tile_x_in_offset + tile_y_in_offset * num_classes;

    if (tile_x_in_offset >= num_classes || tile_y_in_offset >= num_priors) {
        return;
    }

    #pragma unroll
    for (int j = 0; j < TILE_Y; j += BLOCK_DIM_Y) {
        int x_idx = threadIdx.x;
        int y_idx = j + threadIdx.y;
        bool is_not_valid = tile_x_in_offset + x_idx >= num_classes 
                        || tile_y_in_offset + y_idx >= num_priors;
        //int tile_in_idx = threadIdx.x + tile_x_in_offset
        smem[SMEM_STRIDE * y_idx + x_idx] = (is_not_valid)? 0.F : data_in_tile[y_idx * num_classes + x_idx];
    }

    __syncthreads();

    int tile_x_out_offset = blockIdx.y * TILE_Y;
    int tile_y_out_offset = blockIdx.x * TILE_X;
    float* data_out_tile = data_out + n_offset + tile_x_out_offset + tile_y_out_offset * num_priors;

    #pragma unroll
    for (int x_idx = threadIdx.x; x_idx < TILE_Y; x_idx += BLOCK_DIM_X){
        #pragma unroll
        for (int j = 0; j < TILE_X; j += BLOCK_DIM_Y) {
            int y_idx = j + threadIdx.y;
            bool is_not_valid = tile_x_out_offset + x_idx >= num_priors
                            || tile_y_out_offset + y_idx >= num_priors;
            if (!is_not_valid) {
                data_out_tile[y_idx * num_priors+ x_idx] = smem[SMEM_STRIDE * x_idx + y_idx];
            }
        }
    }
}

#define ENABLE_LDGSTS 1
template <int TILE_X, int TILE_Y, int BLOCK_DIM_X, int BLOCK_DIM_Y, int N_TILES = 1>
__global__ void batched_transpose_pf_kernel(const float* data_in, float* data_out, const int num_classes, const int num_priors){
    // num_classes is "fast" dimension in data_in
    // BLOCK_DIM_X == TILE_X
    // N_TILES = number of tiles in Y direction (we have a very narrow matrix)

    constexpr int SMEM_STRIDE = TILE_X + 1;
    constexpr int SMEM_FACTOR = (N_TILES > 1)? 2 : 1;
    __shared__ float smem[SMEM_FACTOR * SMEM_STRIDE * TILE_Y];

    // batch id
    const int n = blockIdx.z;
    const int n_offset = n * num_classes * num_priors;

    const int tile_x_in_offset = blockIdx.x * TILE_X;
    const int tile_y_in_offset = blockIdx.y * TILE_Y * N_TILES;

    if (tile_x_in_offset >= num_classes || tile_y_in_offset >= num_priors) {
        return;
    }

    const float* data_in_tile = data_in + n_offset + tile_x_in_offset + tile_y_in_offset * num_classes;
    #pragma unroll
    for (int j = 0; j < TILE_Y; j += BLOCK_DIM_Y) {
        int x_idx = threadIdx.x;
        int y_idx = j + threadIdx.y;
        bool is_not_valid = tile_x_in_offset + x_idx >= num_classes 
                        || tile_y_in_offset + y_idx >= num_priors;
        //int tile_in_idx = threadIdx.x + tile_x_in_offset
#if ENABLE_LDGSTS == 1
        size_t zfill = (is_not_valid)? sizeof(float) : 0;
        __pipeline_memcpy_async(&smem[SMEM_STRIDE * y_idx + x_idx],
                                &data_in_tile[y_idx * num_classes + x_idx],
                                sizeof(float),
                                zfill);
#else
        smem[SMEM_STRIDE * y_idx + x_idx] = (is_not_valid)? 0.F : data_in_tile[y_idx * num_classes + x_idx];
#endif

    } 

#if ENABLE_LDGSTS == 1
        __pipeline_commit();
#endif

    int istage = 0;
    int istage_next = 1;
    for (int itile = 0; itile < N_TILES; itile++) {

#if ENABLE_LDGSTS == 1
        __pipeline_wait_prior(0); 
#endif
        __syncthreads();

        istage_next = (istage + 1) % 2;

        const int tile_y_in_offset = blockIdx.y * TILE_Y * N_TILES + (itile + 1) * TILE_Y;
        const float* data_in_tile = data_in + n_offset + tile_x_in_offset + tile_y_in_offset * num_classes;
        if (tile_y_in_offset < num_priors && itile < N_TILES-1) {
            #pragma unroll
            for (int j = 0; j < TILE_Y; j += BLOCK_DIM_Y) {
                int x_idx = threadIdx.x;
                int y_idx = j + threadIdx.y;
                bool is_not_valid = tile_x_in_offset + x_idx >= num_classes 
                                || tile_y_in_offset + y_idx >= num_priors;
#if ENABLE_LDGSTS == 1
                size_t zfill = (is_not_valid)? sizeof(float) : 0;
                __pipeline_memcpy_async(&smem[istage_next * SMEM_STRIDE*TILE_Y + SMEM_STRIDE * y_idx + x_idx],
                                        &data_in_tile[y_idx * num_classes + x_idx],
                                        sizeof(float),
                                        zfill);
#else
                smem[istage_next * SMEM_STRIDE*TILE_Y + SMEM_STRIDE * y_idx + x_idx] 
                        = (is_not_valid)? 0.F : data_in_tile[y_idx * num_classes + x_idx];
#endif
            }
#if ENABLE_LDGSTS == 1
            __pipeline_commit();
#endif
        }

        // transpose the previous stage while loading the next one
        int tile_x_out_offset = blockIdx.y * TILE_Y * N_TILES + itile * TILE_Y;
        int tile_y_out_offset = blockIdx.x * TILE_X;
        float* data_out_tile = data_out + n_offset + tile_x_out_offset + tile_y_out_offset * num_priors;

        #pragma unroll
        for (int x_idx = threadIdx.x; x_idx < TILE_Y; x_idx += BLOCK_DIM_X){
            #pragma unroll
            for (int j = 0; j < TILE_X; j += BLOCK_DIM_Y) {
                int y_idx = j + threadIdx.y;
                bool is_not_valid = tile_x_out_offset + x_idx >= num_priors
                                || tile_y_out_offset + y_idx >= num_priors;
                if (!is_not_valid) {
                    data_out_tile[y_idx * num_priors+ x_idx] 
                        = smem[istage * SMEM_STRIDE * TILE_Y + SMEM_STRIDE * x_idx + y_idx];
                }
            }
        }

        istage = istage_next;
    }
}

__global__ void sigmoid_kernel(float* data, int nthreads) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= nthreads) return;

    float val = data[tid];
    data[tid] = __expf(val) / (1.0f + __expf(val));
}

cublasHandle_t handle;

void permuteConfData_cublas(
        cudaStream_t stream,
        const int nthreads,
        const int num_classes,
        const int num_priors,
        int num_layers,
        bool confSigmoid,
        void* new_data,
        void* tmp_data,
        const void* const* conf_data,
        void *active_counts_per_class,
        const int * feature_size,
        const int * num_anchors,
        const bool packed32_nchw)
{
    static bool is_first_call = true;

    assert(packed32_nchw == 0);

    if (is_first_call) {
        cublasCreate(&handle);
    }

    cublasSetStream(handle, stream);

    float alpha = 1.f;
    float beta = 0.f;
    float* b_matrix = 0;

    int output_n_stride = num_classes * num_priors;
    int num_images = nthreads / output_n_stride;
    //printf("permuteConfData_cublas: num_images = %d, num_classes = %d, output_n_stride = %d, \n", num_images, num_classes, output_n_stride);

    cudaMemsetAsync(active_counts_per_class, 0, sizeof(int), stream);

    #define CUSTOM_TRANSPOSE 1

    for (int ib = 0; ib < num_images; ib++) {
        int output_n_offset = ib * output_n_stride;
        int output_concat_offset = output_n_offset;
        for (int layer = 0; layer < num_layers;layer++) {

            int layer_hw_size = feature_size[layer] * feature_size[layer];
            int prev_layer_prior_size = (layer == 0)? 0 : num_anchors[layer-1] * feature_size[layer-1] * feature_size[layer-1];
            output_concat_offset += prev_layer_prior_size * num_classes;
            //printf("permuteConfData_cublas: layer_hw_size = %d, num_anchors = %d, prev_layer_prior_size = %d, output_concat_offset = %d\n",layer_hw_size,  num_anchors[layer], prev_layer_prior_size, output_concat_offset);
            // cublas implies column major
            //number of rows of transposed input(in column major), i.e. output
            int m = num_anchors[layer] * num_classes; // number of conf channels
            // number of columns(in column major)
            int n = layer_hw_size;
            int lda = layer_hw_size;
            int ldb = m;
            int ldc = m;
            cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        m, n,
                        &alpha, reinterpret_cast<const float *>(conf_data[layer]) + ib * m * n, lda,
                        &beta, b_matrix, ldb,
                        reinterpret_cast<float*>(tmp_data) + output_concat_offset, ldc);

            if (layer == num_layers-1) {
                assert(output_concat_offset + num_anchors[layer] * layer_hw_size * num_classes - output_n_offset == num_priors * num_classes);
            }
        }

#if (CUSTOM_TRANSPOSE == 0)
        // transpose 
        int m = num_priors;
        int n = num_classes;
        int lda = n;
        int ldb = m;
        int ldc = m;
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    m, n,
                    &alpha, reinterpret_cast<float*>(tmp_data) + output_n_offset, lda,
                    &beta, b_matrix, ldb,
                    reinterpret_cast<float*>(new_data) + output_n_offset, ldc);
#endif
    }

#if (CUSTOM_TRANSPOSE == 1)
    constexpr int BLOCK_DIM_X = 32;
    constexpr int BLOCK_DIM_Y = 8;
    constexpr int TILE_X = BLOCK_DIM_X;
    constexpr int TILE_Y = 64;
#if 0
    dim3 block = dim3(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid = dim3(div_up(num_classes, TILE_X), div_up(num_priors,TILE_Y), num_images);
    batched_transpose_kernel<TILE_X, TILE_Y, BLOCK_DIM_X, BLOCK_DIM_Y><<<grid, block, 0, stream>>>
            (reinterpret_cast<float*>(tmp_data),
             reinterpret_cast<float*>(new_data),
             num_classes,
             num_priors);
#else
    constexpr int N_TILES = 1;
    dim3 block = dim3(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid = dim3(div_up(num_classes, TILE_X), div_up(num_priors,TILE_Y * N_TILES), num_images);
    batched_transpose_pf_kernel<TILE_X, TILE_Y, BLOCK_DIM_X, BLOCK_DIM_Y, N_TILES><<<grid, block, 0, stream>>>
            (reinterpret_cast<float*>(tmp_data),
             reinterpret_cast<float*>(new_data),
             num_classes,
             num_priors);
#endif
#endif

#if SSD_CUBLAS_CONF_PERMUTE == 0
    int block_size_sigmoid = 256;
    int grid = div_up(nthreads, block_size_sigmoid);
    sigmoid_kernel<<<div_up(nthreads, block_size_sigmoid), block_size_sigmoid, 0, stream>>>(reinterpret_cast<float*>(new_data), nthreads);
#endif
}

#endif

/* This function maps the input index to the corresponding conf_data offset.
The input "conf_data" is composed of "num_layers" conf tensors from the CONV
layers in SSD. These tensors are in NCHW layout. 
The input index is broken down to 4 components: i, c, d, n
i - box coordinate (max 4)
c - class (max num_classes)
d - prior (max num_priors)
n - batch size

The transformed conf_data is generated by:
conf_data[id_layer](CHW)->permute(1,2,0)(HWC)->reshape(H*W*C/num_classes/num_dims, num_classes, num_dims)
->concat(axis=1, num_layers)(num_priors, num_classes, num_dims)[->flatten(num_priors * num_classes * num_dims, 1, 1)]->permute(num_classes, num_priors, num_dims)
Correspondingly, the mapping process will first locate id_layer according to prior and then transform
the index based on (num_classes, num_priors, num_dims) backed to CHW.
C = num_anchors_layer * num_classes * num_dims
HW = num_priors_layer / num_anchors_layer
*/


template <typename Dtype, unsigned nthds_per_cta, int NUM_LAYERS>
//__launch_bounds__(nthds_per_cta)
__launch_bounds__(128)
__global__ void permuteConfData_kernel(
        const int nthreads,
        const int num_classes, int num_classes_mul, int num_classes_shr,
        const int num_priors, int num_priors_mul, int num_priors_shr,
        const int num_dim, int num_dim_mul, int num_dim_shr,
        // int fast_divmod3_mul, int fast_divmod3_shr,
        // int fast_divmod6_mul, int fast_divmod6_shr,
        // int fast_divmod4_mul, int fast_divmod4_shr,
        bool confSigmoid,
        Dtype* new_data,
        int *active_counts_per_class,
        const PermuteConfData<Dtype, NUM_LAYERS> permute_conf_data)
{
    int feature_size[NUM_LAYERS];
    int all_num_anchors[NUM_LAYERS];
    const Dtype *conf_data[NUM_LAYERS];

    #pragma unroll
    for (int layer = 0;layer < NUM_LAYERS;++layer) {
        feature_size[layer] = permute_conf_data.feature_size[layer];
        all_num_anchors[layer] = permute_conf_data.num_anchors[layer];
        conf_data[layer] = permute_conf_data.conf_data[layer];
    }
    const bool packed32_nchw = permute_conf_data.packed32_nchw;

    int index = blockIdx.x * nthds_per_cta + threadIdx.x;
    
    if (index < nthreads)
    {
        int i, i_div, d, d_div, c, n;

        fast_divmod(i_div, i, index, num_dim, num_dim_mul, num_dim_shr);
        fast_divmod(d_div, d, i_div, num_priors, num_priors_mul, num_priors_shr);
        fast_divmod(n, c, d_div, num_classes, num_classes_mul, num_classes_shr);

        //int d_old = d;

        if (n == 0) {
            active_counts_per_class[n] = 0;
        }

        //find layer_id
        int start_layer_prior = 0, end_layer_prior = 0;
        int prior_in_layer = 0;
        const Dtype *conf_data_layer;

        int num_hw;
        int layer_mul;
        int layer_shr;
        int layer;
        int num_anchors;
        int box_channel;
        #pragma unroll
        for(layer = 0; layer < NUM_LAYERS; layer++) {
            end_layer_prior = permute_conf_data.end_layer_prior[layer];

            if(d < end_layer_prior) {
                conf_data_layer = conf_data[layer];
                num_hw = feature_size[layer];

                num_anchors = all_num_anchors[layer];
                box_channel = permute_conf_data.box_channels[layer];

                prior_in_layer = d - start_layer_prior;

                d = INT_MAX;
                layer_mul = permute_conf_data.mul[layer];
                layer_shr = permute_conf_data.shr[layer];
            }
            start_layer_prior = end_layer_prior;
        }

        int mappedIndex;
        // int hw = prior_in_layer % num_hw;
        // int anchor = prior_in_layer / num_hw;
        int hw, anchor;

        int num_ch, ch;

        int concatInputs = permute_conf_data.concatInputs;

        box_channel = (concatInputs)? box_channel : 0;

        if (permute_conf_data.permute_before_reshape) {
            fast_divmod(hw, anchor, prior_in_layer, num_anchors, layer_mul, layer_shr);
            num_ch = box_channel + num_anchors * num_classes * num_dim;
            ch = box_channel + (anchor*num_classes+c)*num_dim + i;
        } else {
            fast_divmod(anchor, hw, prior_in_layer, num_hw, layer_mul, layer_shr);
            // in merged tensor, we prepend box_channel before conf_channels
            num_ch = box_channel + num_dim * num_classes * num_anchors; 
            ch = box_channel + (i*num_classes + c)*num_anchors + anchor;
        }


        if(packed32_nchw) {
            int packed_num_ch = (num_ch+31)/32;
            
            int packed_ch = ch >> 5; // ch/32;
            int packed_ch_offset = ch & 31; // ch%32;

            mappedIndex = ((n * packed_num_ch + packed_ch)*num_hw + hw)*32 + packed_ch_offset;
        }
        else {
            mappedIndex = (n * num_ch + ch)*num_hw + hw;
        }
    
        float result = conf_data_layer[mappedIndex];

        //float reslut_old = result;
        if (confSigmoid)
            result = __expf(result) / (1.0f + __expf(result));
            //result = __frcp_rn(__fadd_rn(1.f, __expf(-result)));

        new_data[index] = result;
        // debug
        // if (c == 1) {
        //     printf("%d %d %f %f\n", c, d_old, result, reslut_old);
        // }
    }
}

template <typename Dtype, int NUM_LAYERS>
ssdStatus_t permuteConfData_gpu(
    cudaStream_t stream,
    const int nthreads,
    const int num_classes,
    const int num_priors,
    const int num_dim,
    bool confSigmoid,
    const void* const* conf_data,
    void* new_data,
    void* active_count_per_class,
    const int num_layers,
    const int* feature_size,
    const int* num_anchors,
    const int* box_channels,
    const bool permute_before_reshape,
    const bool concatInputs,
    const bool packed32_nchw)
{
    const int BS = 128;
    const int GS = (nthreads + BS - 1) / BS;

    PermuteConfData<Dtype, NUM_LAYERS> permute_conf_data;

    //int permute_before_reshape = !permute_before_reshape_;

    // precompute pow2(feature_size) and end_prior_layer for each loop iteration.
    int start_layer_prior = 0;
    for (int i = 0;i < NUM_LAYERS;++i) {
        permute_conf_data.feature_size[i] = feature_size[i] * feature_size[i];
        permute_conf_data.num_anchors[i] = num_anchors[i];
        permute_conf_data.box_channels[i] = box_channels[i];

        int layer_prior_size = num_anchors[i] * permute_conf_data.feature_size[i];
        int end_layer_prior = start_layer_prior + layer_prior_size;

        permute_conf_data.end_layer_prior[i] = end_layer_prior;
        start_layer_prior = end_layer_prior;

        if (permute_before_reshape) {
            find_divisor(permute_conf_data.mul[i], permute_conf_data.shr[i], permute_conf_data.num_anchors[i]);
        } else {
            find_divisor(permute_conf_data.mul[i], permute_conf_data.shr[i], permute_conf_data.feature_size[i]);
        }
    }

    permute_conf_data.packed32_nchw = packed32_nchw;
    permute_conf_data.permute_before_reshape = permute_before_reshape;
    permute_conf_data.concatInputs = concatInputs;

    // determine constants for efficient integer division
    uint32_t num_classes_mul, num_classes_shr;
    uint32_t num_priors_mul, num_priors_shr;
    uint32_t num_dim_mul, num_dim_shr;
    find_divisor(num_classes_mul, num_classes_shr, num_classes);
    find_divisor(num_priors_mul, num_priors_shr, num_priors);
    find_divisor(num_dim_mul, num_dim_shr, num_dim);

    uint32_t fast_divmod_3_mul, fast_divmod_3_shr;
    uint32_t fast_divmod_6_mul, fast_divmod_6_shr;
    uint32_t fast_divmod_4_mul, fast_divmod_4_shr;
    find_divisor(fast_divmod_3_mul, fast_divmod_3_shr, 3);
    find_divisor(fast_divmod_6_mul, fast_divmod_6_shr, 6);
    find_divisor(fast_divmod_4_mul, fast_divmod_4_shr, 4);

    std::memcpy(permute_conf_data.conf_data, conf_data, NUM_LAYERS * sizeof(void*));
    permuteConfData_kernel<Dtype, BS, NUM_LAYERS><<<GS, BS, 0, stream>>>(nthreads,
                                                                num_classes, num_classes_mul, num_classes_shr,
                                                                num_priors, num_priors_mul, num_priors_shr,
                                                                num_dim, num_dim_mul, num_dim_shr,
                                                                // fast_divmod_3_mul, fast_divmod_3_shr,
                                                                // fast_divmod_6_mul, fast_divmod_6_shr,
                                                                // fast_divmod_4_mul, fast_divmod_4_shr,
                                                                confSigmoid,
                                                                (Dtype*) new_data, 
                                                                (int*) active_count_per_class, 
                                                                permute_conf_data);

    CSC(cudaGetLastError(), STATUS_FAILURE);

    static int iter = 0;
    if(iter == -1)
    {
        // debug

        auto output_file = std::ofstream("permuted_scores.bin", std::ios::binary);

        //std::vector<int> header = {num_top_k, num_preds_per_class, segments};
        //output_file.write((char *) &header[0], header.size() * sizeof(int));
        saveDeviceBuffer_((const Dtype *)new_data, nthreads , output_file);
        output_file.close();

        //exit(1);

    } else {iter++;};

    return STATUS_SUCCESS;
}

// permuteConfData LAUNCH CONFIG {{{
typedef ssdStatus_t (*pdFunc)(cudaStream_t,
                              const int,
                              const int,
                              const int,
                              const int,
                              bool,
                              const void* const*,
                              void*,
                              void*,
                              const int,
                              const int*,
                              const int*,
                              const int*,
                              const bool,
                              const bool,
                              const bool);

struct pdLaunchConfig
{
    DType_t t_data;
    int num_layers;
    pdFunc function;

    pdLaunchConfig(DType_t t_data, int num_layers)
        : t_data(t_data)
        , num_layers(num_layers)
    {
    }
    pdLaunchConfig(DType_t t_data, int num_layers, pdFunc function)
        : t_data(t_data)
        , num_layers(num_layers)
        , function(function)
    {
    }
    bool operator==(const pdLaunchConfig& other)
    {
        return (t_data == other.t_data && num_layers == other.num_layers);
    }
};

static std::vector<pdLaunchConfig> pdFuncVec;

bool permuteConfDataInit()
{
    pdFuncVec.push_back(pdLaunchConfig(DataType::kFLOAT, 5,
                                       permuteConfData_gpu<float, 5>));
    pdFuncVec.push_back(pdLaunchConfig(DataType::kFLOAT, 6,
                                       permuteConfData_gpu<float, 6>));
    // pdFuncVec.push_back(pdLaunchConfig(DataType::kHALF, 5, 
    //                                    permuteConfData_gpu<__half, 5>));
    // pdFuncVec.push_back(pdLaunchConfig(DataType::kHALF, 6,
    //                                    permuteConfData_gpu<__half, 6>));
    return true;
}

static bool initialized = permuteConfDataInit();

//}}}

ssdStatus_t permuteConfData(cudaStream_t stream,
                        const int nthreads,
                        const int num_classes,
                        const int num_priors,
                        const int num_dim,
                        const DType_t DT_DATA,
                        bool confSigmoid,
                        const void* const* conf_data,
                        void* new_data,
                        void* active_classes_per_batch,
                        const int num_layers,
                        const int * feature_size,
                        const int * num_anchors,
                        const int * box_channels,
                        const bool permute_before_reshape,
                        const bool concatInputs,
                        const bool packed32_nchw)
{
    pdLaunchConfig lc = pdLaunchConfig(DT_DATA, num_layers);
    for (unsigned i = 0; i < pdFuncVec.size(); ++i)
    {
        if (lc == pdFuncVec[i])
        {
            DEBUG_PRINTF("permuteConfData kernel %d\n", i);
            return pdFuncVec[i].function(stream,
                                         nthreads,
                                         num_classes,
                                         num_priors,
                                         num_dim,
                                         confSigmoid,
                                         conf_data,
                                         new_data,
                                         active_classes_per_batch,
                                         num_layers,
                                         feature_size,
                                         num_anchors,
                                         box_channels,
                                         permute_before_reshape,
                                         concatInputs,
                                         packed32_nchw);
        }
    }
    std::cerr<< "Permute conf data type or num_layers is not supported" << std::endl;
    return STATUS_BAD_PARAM;
}

} // namespace plugin
} // namespace nvinfer1
