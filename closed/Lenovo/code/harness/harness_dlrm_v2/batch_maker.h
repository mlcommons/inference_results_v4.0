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

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_fp16.h>
#include <nvToolsExt.h>

#include "qsl.hpp"

#include "dlrm_v2_qsl.h"
#include "lwis_buffers.h"

using DLRMv2NumericInputType = half; // int8_t; // float
using DLRMv2CategoricalInputType = int32_t;
using DLRMv2OutputType = float;
using DLRMv2InferCallback = std::function<void(std::vector<mlperf::QuerySampleResponse>&)>;

#define NVTX 0
#if NVTX
#define NVTX_MARK(message) nvtxMarkA(message);
#define NVTX_NAME_THIS_THREAD(name) nvtxNameOsThreadA(pthread_self(), name)
#define NVTX_RANGE_PUSH(message) nvtxRangePushA(message);
#define NVTX_RANGE_POP() nvtxRangePop();
#else
#define NVTX_MARK(message)
#define NVTX_NAME_THIS_THREAD(name)
#define NVTX_RANGE_PUSH(message)
#define NVTX_RANGE_POP()
#endif

constexpr size_t kMIN_NUM_PAIRS = 100;

struct DLRMv2Task
{
    mlperf::QuerySample querySample;
    size_t numIndividualPairs;
};

/* DLRMv2DataBuffer abstracts a host and device buffer and provides H2DAsync and D2HAsync. However, it does not assume
 * that `host` refers to its own internal buffer to allow passing in arbitrary host memory addresses. To use its
 * internal buffer, use `GetHostPtr()`.
 */
template <typename T>
class DLRMv2DataBuffer
{
public:
    DLRMv2DataBuffer(
        const size_t length, const size_t maxBatchSize, const bool allocHost = true, const bool allocDevice = true)
        : mLength(length)
        , mMaxBatchSize(maxBatchSize)
        , mMaxBytes(maxBatchSize * sizeof(T) * mLength)
        , mHostBuffer(allocHost ? mMaxBytes : 0)
        , mDeviceBuffer(allocDevice ? mMaxBytes : 0)
        , mHostPtr(static_cast<T*>(mHostBuffer.data()))
        , mDevicePtr(static_cast<T*>(mDeviceBuffer.data()))
    {
        CHECK_EQ(mMaxBatchSize > 0, true);
        CHECK_EQ(mMaxBytes > 0, true);
        CHECK_EQ(!allocHost || (mHostPtr != nullptr), true);
        CHECK_EQ(!allocDevice || (mDevicePtr != nullptr), true);
    }

    void H2H(const T* hostPtr, const size_t size, const size_t offset = 0) const noexcept
    {
        CHECK_EQ(mHostPtr != nullptr, true);
        memcpy(GetHostPtr() + offset * mLength, hostPtr, ElemByteSize() * size);
    }

    void H2DAsync(const T* hostPtr, const size_t size, const cudaStream_t stream) const noexcept
    {
        CHECK_EQ(mDevicePtr != nullptr, true);
        CHECK_EQ(cudaMemcpyAsync(GetDevicePtr(), hostPtr, ElemByteSize() * size, cudaMemcpyHostToDevice, stream),
            cudaSuccess);
    }

    void D2HAsync(T* hostPtr, const size_t size, const cudaStream_t stream) const noexcept
    {
        CHECK_EQ(mDevicePtr != nullptr, true);
        CHECK_EQ(cudaMemcpyAsync(hostPtr, GetDevicePtr(), ElemByteSize() * size, cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
    }

    size_t ElemByteSize() const noexcept
    {
        return mLength * sizeof(T);
    }

    T* const GetDevicePtr() const noexcept
    {
        return mDevicePtr;
    }

    T* const GetHostPtr() const noexcept
    {
        return mHostPtr;
    }

    void SetHostPtr(const T* ptr) noexcept
    {
        mHostPtr = const_cast<T*>(ptr);
    }

    void ResetHostPtr() noexcept
    {
        mHostPtr = static_cast<T*>(mHostBuffer.data());
    }

    bool isDefaultHostPtr() const noexcept
    {
        return mHostPtr == static_cast<const T*>(mHostBuffer.data());
    }

private:
    size_t mLength;
    size_t mMaxBatchSize;
    size_t mMaxBytes;

    lwis::HostBuffer mHostBuffer;
    lwis::DeviceBuffer mDeviceBuffer;
    T* const mDevicePtr;
    T* mHostPtr;
};

// Aliases for convenience
using DLRMv2NumericBuffer = DLRMv2DataBuffer<DLRMv2NumericInputType>;
using DLRMv2CategoricalBuffer = DLRMv2DataBuffer<DLRMv2CategoricalInputType>;
using DLRMv2OutputBuffer = DLRMv2DataBuffer<DLRMv2OutputType>;

class BatchMaker; // forward declaration

class Batch
{
public:
    Batch(const std::string& id, const size_t maxBatchSize, const size_t minBatchSize, const size_t numericVolume,
        const size_t categoricalVolume, BatchMaker* batchMaker);

    // Copy task data from QSL memory to this Batch's host buffers
    void doCopy(const size_t tasksOffset, const size_t pairsOffset, const size_t pairsToCopy) noexcept;

    // Reset batch state after consumption in inference
    void reset() noexcept;

    void pushTask(const DLRMv2Task& task) noexcept
    {
        mTasks.push_back(task);
    }

    std::vector<DLRMv2Task> const getTasks() const noexcept
    {
        return mTasks;
    }

    void commitCopies(const size_t numCopies) noexcept
    {
        mCommittedCopies += numCopies;
    }

    size_t getCommittedCopies() const noexcept
    {
        return mCommittedCopies;
    }

    void completeCopies(const size_t numCopies) noexcept
    {
        mCompletedCopies += numCopies;
    }

    size_t getCompletedCopies() const noexcept
    {
        return mCompletedCopies;
    }

    void markReadyWhenComplete() noexcept
    {
        mReadyWhenComplete = true;
    }

    bool isReadyWhenComplete() const noexcept
    {
        return mReadyWhenComplete;
    }

    bool isComplete() const noexcept
    {
        return mCompletedCopies == mCommittedCopies && mReadyWhenComplete;
    }

    size_t getFreeSpace() const noexcept
    {
        return mMaxBatchSize - mCommittedCopies;
    }

    DLRMv2NumericInputType* const getNumericHostPtr() const noexcept
    {
        return mNumericInputBuf.GetHostPtr();
    }

    DLRMv2CategoricalInputType* const getCategoricalHostPtr() const noexcept
    {
        return mCategoricalInputBuf.GetHostPtr();
    }

    std::string mDebugId;
    BatchMaker* mBatchMaker;

private:
    void ContiguityAwareH2H(const size_t tasksOffset, const size_t pairsOffset, const size_t pairsToCopy) noexcept;
    void IndividualH2H(size_t tasksOffset, const size_t pairsOffset, const size_t pairsToCopy) noexcept;

    std::vector<DLRMv2Task> mTasks;
    DLRMv2NumericBuffer mNumericInputBuf;
    DLRMv2CategoricalBuffer mCategoricalInputBuf;

    size_t mCommittedCopies;
    size_t mCompletedCopies;
    size_t mMinBatchSize;
    size_t mMaxBatchSize;

    bool mReadyWhenComplete;
};

class BatchMaker
{
public:
    BatchMaker(const size_t numStagingThreads, const size_t numStagingBatches, const size_t maxBatchSize,
        const size_t maxPairsPerThread, const size_t numericVolume, const size_t categoricalVolume,
        const bool checkContiguity, const std::shared_ptr<DLRMv2SampleLibrary> qsl, const int64_t deviceIdx,
        const int64_t numaIdx, const int64_t numaNum, const std::vector<int>& cpus);
    ~BatchMaker();

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples, const int offset, const int count) noexcept;
    void FlushQueries() noexcept;
    void StopWork() noexcept;

    // Interface to fetch the earliest Batch from the queue of Batches ready for inference,
    // blocking if no batch is ready
    Batch* GetBatch() noexcept;

    // Interface to notify that the inference thread has completed H2D transfer
    // of a `batch` and it can be returned to the pool of idle batches
    void NotifyH2D(Batch* batch) noexcept;

    int64_t GetDeviceIdx() const noexcept
    {
        return mDeviceIdx;
    };

    std::shared_ptr<DLRMv2SampleLibrary> mQsl;
    bool mCheckContiguity;

private:
    bool UseNuma() const noexcept
    {
        return mNumaNum > 0;
    };

    // Greedily fetches tasks, and copies them to the active staging batch
    void StageBatch(const int threadIdx) noexcept;

    // Pushes a staged batch to the queue of batches ready for inference
    // NOTE: Should be called under mutex lock
    void HandleReadyBatch(Batch* batch) noexcept;

    // Close this batch for new copies, and handle related bookkeeping
    void CloseBatch(Batch* batch) noexcept;

    // Check if batch satisfies conditions to be marked as ready
    void ReadyBatchIfComplete(Batch* batch) noexcept;

    // All Batches owned by this BatchMaker
    std::vector<Batch> mInputBatches;

    // Pointers to idle empty Batches, which are not ready, not staging, and not being used for inference
    std::vector<Batch*> mIdleBatches;

    // Queue of staged Batches that are ready to be sent for inference
    std::deque<Batch*> mReadyBatches;

    // Pointer to Batch currently being used for inference by all StageBatch threads
    Batch* mStagingBatch;

    // Queue of task IDs coming from LoadGen which are to be staged
    std::deque<DLRMv2Task> mTasksQ;

    // Threads running StageBatch
    std::vector<std::thread> mStagingThreads;

    // Mutex to serialize access mStagingBatch & mIdleBatches
    std::mutex mMutex;

    // Condition variable on which StageBatch will wait to produce batches
    std::condition_variable mProducerCV;

    // Condition variable on which GetBatch will wait to consume batches
    std::condition_variable mConsumerCV;

    // Indicates that there will no new tasks and the worker threads should stop processing samples
    bool mStopWork;

    // Indicates FlushQueries has been called by LoadGen
    bool mFlushQueries;

    // Total number of readied pairs sent for inference
    size_t mReadiedPairs;

    // Total number of pairs issued by LoadGen in IssueQuery
    size_t mIssuedPairs;

    size_t mMinBatchSize;
    size_t mMaxBatchSize;
    size_t mMaxPairsPerThread;
    size_t mWaitingBatches;

    const int64_t mNumaIdx;
    const int64_t mNumaNum;
    const int64_t mDeviceIdx;

    // CPUs on which staging threads should run. Empty implies no constraint.
    std::vector<int> mCpus;
};
