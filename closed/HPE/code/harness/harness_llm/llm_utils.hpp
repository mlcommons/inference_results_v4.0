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

#pragma once

#include "../../common/nvtx_common_wrapper.h"
#include "lwis_buffers.h"
#include "qsl.hpp"
#include "utils.hpp"

#include "tensorrt_llm/batch_manager/BatchManager.h"

#include <iostream>
#include <string>
#include <unordered_map>

using namespace tensorrt_llm::batch_manager;

using trtllmSchedulerPolicy = batch_scheduler::SchedulerPolicy;

// ================================
//     Debug support: nvtx ranges
// ================================
#define NVTX_ON

#ifndef NVTX_ON
#define NVTX_GLOBAL_START(A, B, C)
#define NVTX_GLOBAL_END(A)
#define NVTX_THREAD_START(B, C)
#define NVTX_THREAD_END()
#define NVTX_MARK(B, C)
#endif

// Uncomment and run with --verbose_glog=2 to dump input/output tokens
// #define DEBUG
#ifdef OUTPUT_DEBUG
constexpr int32_t kNUM_DEBUG_TOKEN{128};
#endif

// Enable LLM recorder class by enabling the MACRO
#define ENABLE_RECORDER 0

// Enable CommBuffer debug
#define COMMBUF_TIMER_ON 0

// The order of QSL input.
constexpr int32_t kINPUT_TOKEN_IDX{0};
constexpr int32_t kINPUT_LENGTH_IDX{1};

// Data structure to handle QuerySampleResponse
// Data structure to handle LLM Sample
struct LLMSample
{
    mlperf::QuerySample QS;
    int32_t SampleLen;
    uintptr_t SamplePtr;

    LLMSample(mlperf::QuerySample qs, int32_t l, uintptr_t p)
        : QS(qs)
        , SampleLen(l)
        , SamplePtr(p)
    {
    }
};

struct LLMResponse
{
    mlperf::QuerySampleResponse QSR;
    std::vector<int32_t> outputTokens;
};

// Define a data structure for the inference
struct LLMConfig
{
    // Model details
    int32_t mMaxSumSeqlen = 2048;
    int32_t mMaxInputSeqlen;
    int32_t mMaxOutputSeqlen;
    int32_t mEosId;

    // Deployment details
    // TODO: Support hetero TP
    int32_t mTp = 1;
    int32_t mPp = 1;
    int32_t mWorldSize = 1;
    int32_t mMaxGpuBatchSize = 1;

    // Generation Config
    int32_t mBeamWidth = 1;
    float mTemperature = 1.0;
    int32_t mTopK = 1;
    float mTopP = 0.0;
    uint64_t mRandomSeed;
    int32_t mMinOutputSeqlen = 1;
    bool mIsStreaming = false;
    bool mReportFirstToken = false;
    bool mEnableSort = false;
    bool mExcludeInputInOutput = true;

    // Batch manager parameters
    int32_t mMaxNumSequences = 0;
    int32_t mMaxTokensInPagedKvcache = 0;
    float mKvCacheFreeGpuMemFraction = 0.95;
    bool mEnableTrtOverlap = false;

    // Batching mode and policy from TRTLLM
    TrtGptModelType mBatchMode = TrtGptModelType::InflightFusedBatching;
    trtllmSchedulerPolicy mSchedulerPolicy = trtllmSchedulerPolicy::MAX_UTILIZATION;

    // Print the details of the config
    friend std::ostream& operator<<(std::ostream& os, const LLMConfig& config)
    {
        os << "LLMConfig Details:" << std::endl;
        os << "\tmMaxSumSeqlen: " << config.mMaxSumSeqlen << " mMaxInputSeqlen: " << config.mMaxInputSeqlen
           << " mMaxOutputSeqlen: " << config.mMaxOutputSeqlen << " mEosId: " << config.mEosId << std::endl;
        os << "\ttp: " << config.mTp << " pp: " << config.mPp << " mWorldSize: " << config.mWorldSize << std::endl;
        os << "\tmMaxGpuBatchSize: " << config.mMaxGpuBatchSize << " mBeamWidth: " << config.mBeamWidth << std::endl;
        os << "\tmTemperature: " << config.mTemperature << " mTopK: " << config.mTopK << " mTopP: " << config.mTopP
           << " mMinOutputSeqlen: " << config.mMinOutputSeqlen << " mIsStreaming: " << config.mIsStreaming
           << " mReportFirstToken: " << config.mReportFirstToken << " mEnableSort: " << config.mEnableSort
           << " mExcludeInputInOutput: " << config.mExcludeInputInOutput << std::endl;
        os << "\tmMaxNumSequences: " << config.mMaxNumSequences
           << " mMaxTokensInPagedKvcache: " << config.mMaxTokensInPagedKvcache
           << " mKvCacheFreeGpuMemFraction: " << config.mKvCacheFreeGpuMemFraction
           << " mEnableTrtOverlap: " << config.mEnableTrtOverlap << std::endl;
        os << "\tmBatchMode: " << static_cast<int32_t>(config.mBatchMode)
           << " mSchedulerPolicy: " << static_cast<int32_t>(config.mSchedulerPolicy) << std::endl;
        return os;
    };
};

// commBuffer stores data with unique ID
// Each data may be at most as long as mDataLen
// commBuffer entires may be at most mNumBuffer
// In order to write, first get available entry idx by calling getAvailBufferIdx(),
// then commit the entry by calling commitBuffer(idx, data_len, uniqueID) before write.
// This commitBuffer() returns address pointer for the buffer, i.e. commBuffer[idx].data() FYI
// When the lifetime of the entry is expired, clean buffer by using uniqueID: cleanBuffer(uniqueID)
template <typename T, typename U>
class commBuffer
{
public:
    explicit commBuffer(size_t num_buffer, size_t data_length, std::string name = "commBuffer")
        : mNumBuffer(num_buffer)
        , mDataLen(data_length)
        , mName(name)
#if COMMBUF_TIMER_ON
        , mTotalLifetimeCount(0)
        , mTotalLifetime(0)
        , mAggregatedOccupancy(0)
        , mSize(0)
#endif
    {
        mBuffer.resize(mNumBuffer);
        for (int32_t i = 0; i < mNumBuffer; ++i)
        {
            mBufferIndicesAvailable.push_back(i);
            mBuffer[i].reserve(mDataLen);
            mBuffer[i].resize(0);
        }
    }

    ~commBuffer()
    {
        CHECK_EQ(mBufferIndicesAvailable.size(), mNumBuffer) << "CommBuffer management is defective; missing indices";
#if COMMBUF_TIMER_ON
        CHECK(getOccupancy() == 0) << "CommBuffer management is defective; buffer is not empty";
        auto avgLifetime = mTotalLifetimeCount == 0 ? 0 : mTotalLifetime.count() / mTotalLifetimeCount;
        LOG(INFO) << "CommBuffer[" << mName << "] Lifetime reports " << avgLifetime << " ms per call for " 
                  << mTotalLifetimeCount << " times." << std::endl;
        auto avgOccupancy = mTotalLifetimeCount == 0
            ? 0
            : static_cast<long double>(mAggregatedOccupancy) / static_cast<long double>(mTotalLifetimeCount);
        LOG(INFO) << "CommBuffer[" << mName << "] average occupancy reports " << avgOccupancy << std::endl;
#endif
    }

#if COMMBUF_TIMER_ON
    size_t const getOccupancy() const
    {
        return mSize.load(std::memory_order_relaxed);
    }
#endif

private:
    size_t getIdxFromUniqId(U uniqueId)
    {
        auto& [idx, size] = mBufferMap[uniqueId];
        CHECK(size == mBuffer[idx].size()) << mName << ": commBuffer corrupted - data len seems inconsistent";
        return idx;
    }

    size_t getDataLenFromUniqId(U uniqueId)
    {
        auto& [idx, size] = mBufferMap[uniqueId];
        CHECK(size == mBuffer[idx].size()) << mName << ": commBuffer corrupted - data len seems inconsistent";
        return size;
    }

public:
    T getBufferContent(size_t bufferIdx, size_t contentIdx)
    {
        CHECK_LT(contentIdx, mBuffer[bufferIdx].size()) << mName << ": Trying to access invalid content";
        return mBuffer[bufferIdx][contentIdx];
    }

    size_t getAvailBufferIdx()
    {
        std::unique_lock<std::mutex> lck(mBufferMtx);
        mBufferCV.wait(lck, [&]() { return !mBufferIndicesAvailable.empty(); });
        CHECK(!mBufferIndicesAvailable.empty()) << mName << ": commBuffer overflow; flow control may be defective";
        auto idx = mBufferIndicesAvailable.front();
        mBufferIndicesAvailable.pop_front();
        return idx;
    }

    void* commitBuffer(size_t idx, size_t size, U uniqueId)
    {
        CHECK_LE(size, mDataLen) << mName << ": Asked size is bigger than commBuffer MAX size";
        mBuffer[idx].resize(size);
        std::lock_guard<std::mutex> lock(mBufferMtx);
        mBufferMap[uniqueId] = std::make_tuple(idx, size);
#if COMMBUF_TIMER_ON
        mLifetimeTracker[uniqueId] = std::chrono::high_resolution_clock::now();
        mSize.store(mNumBuffer - mBufferIndicesAvailable.size(), std::memory_order_release);
#endif
        return mBuffer[idx].data();
    }

    void cleanBuffer(U uniqueId)
    {
        std::lock_guard<std::mutex> lock(mBufferMtx);
        auto idx = getIdxFromUniqId(uniqueId);
        CHECK(0 < mBuffer[idx].size()) << mName << ": commBuffer may be corrupted - cleaning 0 size entry";
        mBuffer[idx].resize(0);
        mBufferMap.erase(uniqueId);
        mBufferIndicesAvailable.push_back(idx);
        mBufferCV.notify_one();
#if COMMBUF_TIMER_ON
        ++mTotalLifetimeCount;
        mTotalLifetime += std::chrono::high_resolution_clock::now() - mLifetimeTracker[uniqueId];
        mLifetimeTracker.erase(uniqueId);
        mAggregatedOccupancy += getOccupancy();
        mSize.store(mNumBuffer - mBufferIndicesAvailable.size(), std::memory_order_release);
#endif
    }

private:
    // Store data if input buffer is not available locally and therefore buffering is needed
    // Also set up map to get the index from vector used as buffers
    std::vector<std::vector<T>> mBuffer;
    std::unordered_map<U, std::tuple<size_t, size_t>> mBufferMap;
    std::deque<size_t> mBufferIndicesAvailable;
    size_t mNumBuffer;
    size_t mDataLen;
    std::string mName;
    std::mutex mBufferMtx;
    std::condition_variable mBufferCV;
#if COMMBUF_TIMER_ON
    std::atomic<size_t> mSize;
    std::unordered_map<U, std::chrono::time_point<std::chrono::high_resolution_clock>> mLifetimeTracker;
    size_t mTotalLifetimeCount;
    std::chrono::duration<double, std::milli> mTotalLifetime;
    size_t mAggregatedOccupancy;
#endif

};
