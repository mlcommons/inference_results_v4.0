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

#ifndef _PLUGIN_HELPER_H
#define _PLUGIN_HELPER_H

#include <cassert>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <limits>
#include <numeric>

using namespace nvinfer1;

#define TRT_ASSERT(assertion) assert(assertion)

#if DEBUG
#define print_log(...)                                                                                                 \
    {                                                                                                                  \
        char str[100];                                                                                                 \
        sprintf(str, __VA_ARGS__);                                                                                     \
        std::cout << "CUSTOM PLUGIN TRACE----> call "                                                                  \
                  << "[" << __FILE__ << "][" << __FUNCTION__ << "][Line " << __LINE__ << "] " << str << std::endl;     \
    }
#else
#define print_log(...)
#endif

namespace helper
{

//! \brief Safe wrapper of static_cast
//! \details numericCast is free (equivalent to static_cast) when DstType
//!          can hold all possible values of SrcType. There are two versions,
//!          one when sizeof(SrcType) >= sizeof(DstType) that has to check
//!          for the case where a signed casts into an unsigned of larger type.
//!          The second version is when sizeof(SrcType) < sizeof(DstType) and the
//!          destination can always represent the source type.
//! \param[in] src: source value of SrcType
//! \return a variable of type DstType.
//! \remark Non-blocking, reentrant, thread safe
template <typename DstType, typename SrcType>
auto numericCast(SrcType src) noexcept -> typename std::enable_if<std::is_integral<SrcType>::value
        && std::is_integral<DstType>::value && std::is_same<SrcType, DstType>::value,
    DstType>::type
{
    return src;
}

//! \brief numericCast
template <typename DstType, typename SrcType>
auto numericCast(SrcType src) noexcept(false) -> typename std::enable_if<std::is_integral<SrcType>::value
        && std::is_integral<DstType>::value && !std::is_same<SrcType, DstType>::value
        && sizeof(SrcType) >= sizeof(DstType) && std::is_unsigned<SrcType>::value && std::is_unsigned<DstType>::value,
    DstType>::type
{
    TRT_ASSERT(src <= static_cast<SrcType>(std::numeric_limits<DstType>::max()));
    return static_cast<DstType>(src);
}

//! \brief numericCast with upper and lower bound check
template <typename DstType, typename SrcType>
auto numericCast(SrcType src) noexcept(false) -> typename std::enable_if<std::is_integral<SrcType>::value
        && std::is_integral<DstType>::value && !std::is_same<SrcType, DstType>::value
        && sizeof(SrcType) >= sizeof(DstType) && (std::is_signed<SrcType>::value || std::is_signed<DstType>::value),
    DstType>::type
{
    constexpr size_t dstSize = sizeof(DstType);
    constexpr size_t srcSize = sizeof(SrcType);
    constexpr bool isUnsignedToSigned = (std::is_unsigned<SrcType>::value && std::is_signed<DstType>::value);
    constexpr bool needUpperCheck = ((dstSize < srcSize) || ((dstSize == srcSize) && isUnsignedToSigned));
    if (needUpperCheck)
    {
        TRT_ASSERT(src <= static_cast<SrcType>(std::numeric_limits<DstType>::max()));
    }
    constexpr bool isSignedToUnsigned = (std::is_signed<SrcType>::value && std::is_unsigned<DstType>::value);
    constexpr bool isSignedToSigned = (std::is_signed<SrcType>::value && std::is_signed<DstType>::value);
    constexpr bool needLowerCheck = (isSignedToUnsigned || (isSignedToSigned && (dstSize < srcSize)));
    if (needLowerCheck)
    {
        TRT_ASSERT(src >= static_cast<SrcType>(std::numeric_limits<DstType>::lowest()));
    }
    return static_cast<DstType>(src);
}

//! \brief numericCast with lower bound check
template <typename DstType, typename SrcType>
auto numericCast(SrcType src) noexcept(false) -> typename std::enable_if<std::is_integral<SrcType>::value
        && std::is_integral<DstType>::value && sizeof(SrcType) < sizeof(DstType),
    DstType>::type
{
    constexpr size_t dstSize = sizeof(DstType);
    constexpr size_t srcSize = sizeof(SrcType);
    constexpr bool isSignedToUnsigned = (std::is_signed<SrcType>::value && std::is_unsigned<DstType>::value);
    constexpr bool isSignedToSigned = (std::is_signed<SrcType>::value && std::is_signed<DstType>::value);
    constexpr bool needLowerCheck = (isSignedToUnsigned || (isSignedToSigned && (dstSize < srcSize)));
    if (needLowerCheck)
    {
        TRT_ASSERT(src >= static_cast<SrcType>(std::numeric_limits<DstType>::lowest()));
    }
    return static_cast<DstType>(src);
}

//!
//! \brief Perform range checks and arithmetic addition
//!
//! \param x the first operand
//! \param y the second operand
//!
//! \return Sum of x and y after range check
//!
template <typename T>
inline T addPositiveRangeCheck(T const x, T const y)
{
    T constexpr zero = static_cast<T>(0);
    T const maxValX = std::numeric_limits<T>::max() - static_cast<T>(y);
    T const maxValY = std::numeric_limits<T>::max() - static_cast<T>(x);

    T result = zero;
    // In case of unsigned >= zero is understood
    bool positiveX = std::is_signed<T>::value ? (x >= zero) : true;
    bool positiveY = std::is_signed<T>::value ? (y >= zero) : true;

    // If typename is an unsigned then no need to compare with equal to 0 scenario
    if (positiveX && positiveY && x < maxValX && y < maxValY)
    {
        result = x + y;
    }
    else
    {
        throw std::runtime_error("Out of range to perform arithmetic addition operation");
    }
    return result;
}

size_t type2size(DataType type) noexcept
{
    if (type == DataType::kFLOAT)
    {
        return 4;
    }
    else if (type == DataType::kHALF)
    {
        return 2;
    }
    else if (type == DataType::kINT8)
    {
        return 1;
    }
    assert(false && "error type");
    return 0;
}

inline int64_t volume(const Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

} // namespace helper

#endif // _PLUGIN_HELPER_H
