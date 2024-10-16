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

#include <cuda_runtime.h>
#include <cstdint>

template<int M_, int N_>
struct DivUp {
    static constexpr int VALUE = (M_ + N_ - 1) / N_;
};

extern "C" {
  __device__ uint32_t __nvvm_get_smem_pointer(void *ptr);
}

inline __device__ uint32_t get_smem_pointer(const void *ptr) {
  return __nvvm_get_smem_pointer(const_cast<void *>(ptr));
}

/*32bit/4B reg*/
template<int N_REGS>
inline __device__ void clear(uint32_t* regs) {
#pragma unroll N_REGS
  for (int ii = 0; ii < N_REGS; ++ii) {
    asm volatile ( "mov.u32 %0, 0; \n"  : "=r"( regs[ii] ) :  );
  }
}

/***** LDGSTS PTX starts *****/
template<int BYTES_PER_LDGSTS>
inline __device__ void ldgsts(const uint32_t &dst, const void* src) {}

template<>
inline __device__ void ldgsts<4>(const uint32_t &dst, const void* src) {
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 4;\n"	\
    :: "r"(dst), "l"(src));
}

template<>
inline __device__ void ldgsts<8>(const uint32_t &dst, const void* src) {
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 8;\n"	\
    :: "r"(dst), "l"(src));
}

template<>
inline __device__ void ldgsts<16>(const uint32_t &dst, const void* src) {
  asm volatile(
    "cp.async.cg.shared.global [%0], [%1], 16;\n"	\
    :: "r"(dst), "l"(src));
}
/***** LDGSTS PTX ends *****/

/***** LDS PTX starts *****/

template<int BYTES_PER_LDS>
inline __device__ void lds(char* dst, uint32_t ptr) {}

template<>
inline __device__ void lds<4>(char* dst, uint32_t ptr) {
  uint32_t &tmp = *(reinterpret_cast<uint32_t*>(dst));
  asm volatile("ld.shared.b32 %0, [%1];\n"
    : "=r"(tmp)
    :  "r"(ptr));
}

template<>
inline __device__ void lds<8>(char* dst, uint32_t ptr) {
  uint2    &tmp = *(reinterpret_cast<uint2*>(   dst));
  asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n"
    : "=r"(tmp.x)
    , "=r"(tmp.y)
    :  "r"(ptr));
}

template<>
inline __device__ void lds<16>(char* dst, uint32_t ptr) {
  uint4    &tmp = *(reinterpret_cast<uint4*>(   dst));
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(tmp.x)
    , "=r"(tmp.y)
    , "=r"(tmp.z)
    , "=r"(tmp.w)
    :  "r"(ptr));
}

template<int BYTES_PER_SDS>
inline __device__ void sts(uint32_t ptr, const char* src) {}


template<>
inline __device__ void sts<4>(uint32_t ptr, const char* src) {
  const uint32_t &tmp = *(reinterpret_cast<const uint32_t*>(src));
  asm volatile("st.shared.b32 [%0], %1;\n"
    :
    : "r"(ptr)
    , "r"(tmp));
}

template<>
inline __device__ void sts<8>(uint32_t ptr, const char* src) {
  const uint2 &tmp = *(reinterpret_cast<const uint2*>(src));
  asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n"
    :
    : "r"(ptr)
    , "r"(tmp.x)
    , "r"(tmp.y));
}

template<>
inline __device__ void sts<16>(uint32_t ptr, const char* src) {
  const uint4 &tmp = *(reinterpret_cast<const uint4*>(src));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
    :
    : "r"(ptr)
    , "r"(tmp.x)
    , "r"(tmp.y)
    , "r"(tmp.z)
    , "r"(tmp.w));
}
/***** LDS PTX ends *****/

/***** LDG PTX starts *****/

template<int BYTES_PER_LDG>
inline __device__ void ldg(char* dst, const void *ptr) {}

template<>
inline __device__ void ldg<4>(char* dst, const void *ptr) {
  uint32_t &tmp = *(reinterpret_cast<uint32_t*>(dst));
  // asm volatile(
  //   "ld.global.ca.b32 {%0}, [%1];\n"		\
  //     : "=r"(tmp)
  //     :  "l"(ptr));
  tmp = *reinterpret_cast<const uint32_t*>(ptr);
}

template<>
inline __device__ void ldg<8>(char* dst, const void *ptr) {
  uint2    &tmp = *(reinterpret_cast<uint2*>   (dst));
  // asm volatile(
  //   "ld.global.ca.v2.b32 {%0, %1}, [%2];\n"	\
  //     : "=r"(tmp.x)
  //     , "=r"(tmp.y)
  //     :  "l"(ptr));
  tmp = *reinterpret_cast<const uint2*>(ptr);
}

template<>
inline __device__ void ldg<16>(char* dst, const void *ptr) {
  uint4    &tmp = *(reinterpret_cast<uint4*>   (dst));
  // asm volatile(
  //   "ld.global.ca.v4.b32 {%0, %1, %2, %3}, [%4];\n"	\
  //     : "=r"(tmp.x)
  //     , "=r"(tmp.y)
  //     , "=r"(tmp.z)
  //     , "=r"(tmp.w)
  //     :  "l"(ptr));
  tmp = *reinterpret_cast<const uint4*>(ptr); // LDG.E.128.CONSTANT, what's the ptx??
}
/***** LDG PTX ends *****/

/***** STG PTX starts *****/
template<int BYTES_PER_STG>
inline __device__ void stg(void *ptr, const char* src) {}

template<>
inline __device__ void stg<4>(void *ptr, const char* src) {
  const uint32_t &tmp = *(reinterpret_cast<const uint32_t*>(src));
  *reinterpret_cast<uint32_t*>(ptr) = tmp;
}

template<>
inline __device__ void stg<8>(void *ptr, const char* src) {
  const uint2    &tmp = *(reinterpret_cast<const uint2*>   (src));
  *reinterpret_cast<uint2*>(ptr) = tmp;
}

template<>
inline __device__ void stg<16>(void *ptr, const char* src) {
  const uint4    &tmp = *(reinterpret_cast<const uint4*>   (src));
  *reinterpret_cast<uint4*>(ptr) = tmp;
}
/***** STG PTX ends *****/

/***** CONVERSIONS start *****/
inline __device__ float i2f_rn(int32_t i) {
  float f;

  asm volatile("cvt.rn.f32.s32 %0,%1;" : "=f"(f) : "r"(i));

  return f;
}

inline __device__ int32_t f2i_s8(float f) {
  int32_t i;

  asm volatile("cvt.rni.s8.f32 %0,%1;" : "=r"(i) : "f"(f));

  return i;
}


inline __device__ int32_t idp_4a(const int32_t a, const int32_t b, const int32_t c) {
  int32_t r;
  asm volatile("dp4a.s32.s32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

inline __device__ int32_t byte_extract(int32_t x, int32_t p)
{
  int32_t c = 0;
  const int32_t b = 0x00000001 << (p * 8);
  int32_t r = idp_4a(x, b, c);
  return r;
}

inline __device__ float4 s8x4_to_float4(uint32_t in) {
  int32_t t[4];
  float4 f;

  t[0] = byte_extract(in, 0);
  t[1] = byte_extract(in, 1);
  t[2] = byte_extract(in, 2);
  t[3] = byte_extract(in, 3);

  f.x = (float)(t[0]);
  f.y = (float)(t[1]);
  f.z = (float)(t[2]);
  f.w = (float)(t[3]);
  return f;
}

inline __device__ int32_t pack_int8x4(int32_t (&a)[4]) {
  int32_t res;
  // asm volatile(
  //     "{\n" \
  //     "prmt.b32 %2, %1, %2, 0x1140;\n" \
  //     "prmt.b32 %4, %3, %4, 0x1140;\n" \
  //     "prmt.b32 %0, %2, %4, 0x5410;\n" \
  //     "}" \
  //     : "=r"(res) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));
  asm volatile(
      "{\n" \
      ".reg .u32 r4;\n" \
      "cvt.pack.sat.s8.s32.b32   r4, %4, %3,  0;\n" \
      "cvt.pack.sat.s8.s32.b32   %0, %2, %1, r4;\n" \
      "}" \
      : "=r"(res) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));
  return res;
}

inline __device__ uint32_t float4_to_s8x4(float4 in) {
  uint32_t ret;

  int32_t tmp[4];
  tmp[0] = f2i_s8( in.x );
  tmp[1] = f2i_s8( in.y );
  tmp[2] = f2i_s8( in.z );
  tmp[3] = f2i_s8( in.w );

  ret = pack_int8x4( tmp );

  return ret;
}

/***** CONVERSIONS end   *****/