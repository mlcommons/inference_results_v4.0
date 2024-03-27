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

#ifndef NVTX_WRAPPER_HPP
#define NVTX_WRAPPER_HPP

#include "nvtxUtils.hpp"

// ================================
//     Debug support: nvtx ranges
// ================================
#define NVTX_ON

#ifndef NVTX_ON
#define NVTX_GLOBAL_START(A, B, C)
#define NVTX_GLOBAL_START_WITH_PAYLOAD(A, B, C, D)
#define NVTX_GLOBAL_END(a)
#endif

// ================================
//     Debug support: perf tracing
// ================================

#undef PERFTR_ON

#ifndef PERFTR_ON
#undef PERF_TR_PUSH
#undef PERF_TR_POP
#undef PERF_TR_EVENT
#undef PERT_TR_EVENT_SUBSAMPLE
#define PERF_TR_PUSH(A, B, C, D) 
#define PERF_TR_POP(A, B, C, D)
#define PERF_TR_EVENT(A, B, C, D)
#define PERT_TR_EVENT_SUBSAMPLE(SUBSAMP, UPDATE, A, B, D)
#endif


#endif // NVTX_WRAPPER_HPP
