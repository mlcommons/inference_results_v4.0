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

#include "common_ptx.cuh"

template<int BYTES_PER_LDGSTS>
inline __device__
void stream_gmem_to_smem
(
  const char* __restrict__ &gmem_base,
  const uint32_t            smem_base,
  const int                 num_ldgsts,
  const int                 tidx,
  const int                 bdim
) {

  #pragma unroll 1
  for (int idx = tidx; idx < num_ldgsts; idx += bdim) {

    const int      offset   = idx * BYTES_PER_LDGSTS;
    const uint32_t smem_ptr = smem_base + offset;
    const char*    gmem_ptr = gmem_base + offset;

    ldgsts<BYTES_PER_LDGSTS>(smem_ptr, gmem_ptr);
  }
}
