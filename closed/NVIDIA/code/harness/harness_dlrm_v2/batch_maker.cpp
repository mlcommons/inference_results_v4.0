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

#include "batch_maker.h"
#include "utils.hpp"

Batch::Batch(const std::string& id, const size_t minBatchSize, const size_t maxBatchSize, const size_t numericVolume,
    const size_t categoricalVolume, BatchMaker* batchMaker)
    : mDebugId(id)
    , mBatchMaker(batchMaker)
    , mMinBatchSize(minBatchSize)
    , mMaxBatchSize(maxBatchSize)
    , mNumericInputBuf(numericVolume, maxBatchSize, true, false)
    , mCategoricalInputBuf(categoricalVolume, maxBatchSize, true, false)
{
    // Reserve needs to be called only once; reset can just clear() and reserved capacity won't be touched by clear()
    mTasks.reserve(mMaxBatchSize);
    reset();
}

void Batch::ContiguityAwareH2H(const size_t tasksOffset, const size_t pairsOffset, const size_t pairsToCopy) noexcept
{
    // We assume tasks before `tasksOffset` (if any) have already been processed by this function and are contiguous.
    // We check for 2 things:
    //  - internal contiguity: whether all tasks [tasksOffset, end) are contiguous, by comparing the expected vs actual
    //    address of last task
    //  - external contiguity: whether task at `tasksOffset` is contiguous with that at tasksOffset-1
    //
    // If both are contiguous, we skip H2H and point the input buffers to QSL host memory instead.
    CHECK(mBatchMaker->mCheckContiguity);

    NVTX_RANGE_PUSH("Contiguity H2H");
    const auto qsl = mBatchMaker->mQsl;
    const auto numericInputVolume = mNumericInputBuf.ElemByteSize() / sizeof(DLRMv2NumericInputType);

    const DLRMv2Task& firstTask = mTasks[tasksOffset];
    const DLRMv2Task& lastTask = mTasks.back();

    // first address in the contiguous block starting from mTasks[0]
    const DLRMv2NumericInputType* globalContigStart
        = static_cast<DLRMv2NumericInputType*>(qsl->GetSampleAddress(mTasks[0].querySample.index, 0));

    // expected first task addr: right after the end of last contiguous block
    const DLRMv2NumericInputType* expectedFirstTaskStart = globalContigStart + (pairsOffset * numericInputVolume);
    // actual first task addr: first address of current subset of mTasks
    const DLRMv2NumericInputType* actualFirstTaskStart
        = static_cast<DLRMv2NumericInputType*>(qsl->GetSampleAddress(firstTask.querySample.index, 0));

    // expected address end of lastTaskStart, if current mTasks are contiguous
    const DLRMv2NumericInputType* expectedLastTaskStart = actualFirstTaskStart + (pairsToCopy * numericInputVolume);

    // real address of lastTaskStart in this set of mTasks
    const DLRMv2NumericInputType* actualLastTaskStart
        = static_cast<DLRMv2NumericInputType*>(qsl->GetSampleAddress(lastTask.querySample.index, 0))
        + (qsl->GetNumUserItemPairs(lastTask.querySample.index) * numericInputVolume);

    // is there contiguity from mTasks[tasksOffset] to mTasks.back
    const auto internalContiguity = (expectedLastTaskStart == actualLastTaskStart);

    // is there contiguity from mTasks[0] to mTasks[tasksOffset]
    const auto externalContiguity = (actualFirstTaskStart == expectedFirstTaskStart);

    if (!(externalContiguity && internalContiguity))
    {
        if (!mNumericInputBuf.isDefaultHostPtr())
        {
            // might be set elsewhere, reset to default
            // if default, assume continuation
            mNumericInputBuf.ResetHostPtr();
            mCategoricalInputBuf.ResetHostPtr();

            // First transfer existing contiguous block from QSL to own buffer
            mNumericInputBuf.H2H(globalContigStart, pairsOffset, 0);
            const auto catInpAddr = qsl->GetSampleAddress(mTasks[0].querySample.index, 1);
            mCategoricalInputBuf.H2H(static_cast<DLRMv2CategoricalInputType*>(catInpAddr), pairsOffset, 0);
        }

        if (internalContiguity)
        {
            // current set of tasks is contiguous, so can append pairsToCopy to mNumericInputBuf in one go
            mNumericInputBuf.H2H(actualFirstTaskStart, pairsToCopy, pairsOffset);
            const auto catInpAddr = qsl->GetSampleAddress(firstTask.querySample.index, 1);
            mCategoricalInputBuf.H2H(static_cast<DLRMv2CategoricalInputType*>(catInpAddr), pairsToCopy, pairsOffset);
        }

        else
        {
            // discontiguous, but don't know where. Copy tasks one by one
            IndividualH2H(tasksOffset, pairsOffset, pairsToCopy);
        }
    }

    else
    {
        // all contiguous, point to QSL memory
        mNumericInputBuf.SetHostPtr(globalContigStart);
        const auto catInpAddr = qsl->GetSampleAddress(mTasks[0].querySample.index, 1);
        mCategoricalInputBuf.SetHostPtr(static_cast<DLRMv2CategoricalInputType*>(catInpAddr));
    }

    NVTX_RANGE_POP();
}

// copy over batches individually
// for when they are non-contiguous
void Batch::IndividualH2H(size_t tasksOffset, const size_t pairsOffset, const size_t pairsToCopy) noexcept
{
    const auto qsl = mBatchMaker->mQsl;

    size_t copiedPairs = 0;
    while (copiedPairs < pairsToCopy)
    {
        const auto& task = mTasks[tasksOffset++];
        const auto queryIndex = task.querySample.index;
        const auto numIndividualPairs = task.numIndividualPairs;
        const auto numInpAddr = qsl->GetSampleAddress(queryIndex, 0);
        const auto catInpAddr = qsl->GetSampleAddress(queryIndex, 1);

        mNumericInputBuf.H2H(
            static_cast<DLRMv2NumericInputType*>(numInpAddr), numIndividualPairs, pairsOffset + copiedPairs);
        mCategoricalInputBuf.H2H(
            static_cast<DLRMv2CategoricalInputType*>(catInpAddr), numIndividualPairs, pairsOffset + copiedPairs);

        copiedPairs += numIndividualPairs;
    }
}

void Batch::doCopy(const size_t tasksOffset, const size_t pairsOffset, const size_t pairsToCopy) noexcept
{
    NVTX_RANGE_PUSH(("Copying batch " + mDebugId).c_str());
    DLOG(INFO) << mDebugId << " doCopy : " << pairsToCopy;

    if (mBatchMaker->mCheckContiguity)
    {
        ContiguityAwareH2H(tasksOffset, pairsOffset, pairsToCopy);
    }

    else
    {
        IndividualH2H(tasksOffset, pairsOffset, pairsToCopy);
    }

    NVTX_RANGE_POP();
}

void Batch::reset() noexcept
{
    mCommittedCopies = 0;
    mCompletedCopies = 0;
    mReadyWhenComplete = false;
    mTasks.clear();
    mNumericInputBuf.ResetHostPtr();
    mCategoricalInputBuf.ResetHostPtr();
}

BatchMaker::BatchMaker(const size_t numStagingThreads, const size_t numStagingBatches, const size_t maxBatchSize,
    const size_t maxPairsPerThread, const size_t numericVolume, const size_t categoricalVolume,
    const bool checkContiguity, const std::shared_ptr<DLRMv2SampleLibrary> qsl, const int64_t deviceIdx,
    const int64_t numaIdx, const int64_t numaNum, const std::vector<int>& cpus)
    : mMaxBatchSize(maxBatchSize)
    , mMaxPairsPerThread(maxPairsPerThread)
    , mCheckContiguity(checkContiguity)
    , mQsl(qsl)
    , mStagingBatch(nullptr)
    , mIssuedPairs(0)
    , mReadiedPairs(0)
    , mFlushQueries(false)
    , mStopWork(false)
    , mWaitingBatches(0)
    , mDeviceIdx(deviceIdx)
    , mNumaIdx(numaIdx)
    , mNumaNum(numaNum)
    , mCpus(cpus)
{
    if (UseNuma())
    {
        bindNumaMemPolicy(mNumaIdx, mNumaNum);
    }

    DLOG(INFO) << "BatchMaker - numStagingThreads = " << numStagingThreads;
    DLOG(INFO) << "BatchMaker - numStagingBatches = " << numStagingBatches;
    DLOG(INFO) << "BatchMaker - maxPairsPerThread = " << maxPairsPerThread;

    if (mCheckContiguity)
    {
        LOG(INFO) << "Contiguity-Aware H2H : ON";
        mMaxPairsPerThread = mMaxPairsPerThread == 0 ? mMaxBatchSize : mMaxPairsPerThread;
    }
    else
    {
        LOG(INFO) << "Contiguity-Aware H2H : OFF";
        mMaxPairsPerThread = mMaxPairsPerThread == 0 ? mMaxBatchSize / numStagingThreads : mMaxPairsPerThread;
    }

    mInputBatches.reserve(numStagingBatches);
    for (int i = 0; i < numStagingBatches; ++i)
    {
        mInputBatches.emplace_back(
            "Batch#" + std::to_string(i), 1, mMaxBatchSize, numericVolume, categoricalVolume, this);
    }

    // All batches idle initially
    mIdleBatches.reserve(numStagingBatches);
    for (auto& batch : mInputBatches)
    {
        mIdleBatches.push_back(&batch);
    }

    // start StageBatch for numStagingThreads
    mStagingThreads.reserve(numStagingThreads);
    for (int i = 0; i < numStagingThreads; ++i)
    {
        mStagingThreads.emplace_back(&BatchMaker::StageBatch, this, i);
    }

    // Limit the staging threads to the closest CPUs.
    if (UseNuma())
    {
        for (auto& stagingThread : mStagingThreads)
        {
            bindThreadToCpus(stagingThread, mCpus);
        }
    }

    if (UseNuma())
    {
        resetNumaMemPolicy();
    }
}

BatchMaker::~BatchMaker()
{
    DLOG(INFO) << "~BatchMaker";

    StopWork();
    for (auto& stagingThread : mStagingThreads)
    {
        stagingThread.join();
    }
}

void BatchMaker::IssueQuery(const std::vector<mlperf::QuerySample>& samples, const int offset, const int count) noexcept
{
    NVTX_RANGE_PUSH("BatchMaker::IssueQuery");

    mFlushQueries = false;
    for (int i = 0; i < count; ++i)
    {
        {
            const auto& sample = samples[offset + i];
            std::unique_lock<std::mutex> lock(mMutex);

            const size_t numPairs = mQsl->GetNumUserItemPairs(sample.index);
            mTasksQ.push_back({sample, numPairs});
            mIssuedPairs += numPairs;
        }
    }
    mProducerCV.notify_one();

    NVTX_RANGE_POP();
}

void BatchMaker::FlushQueries() noexcept
{
    NVTX_RANGE_PUSH("BatchMaker::FlushQueries");
    DLOG(INFO) << "FlushQueries";

    std::unique_lock<std::mutex> mMutex;
    if (mTasksQ.empty() && mStagingBatch && mStagingBatch->getCommittedCopies() > 0)
    {
        // close this buffer on my own, since no thread will do it until new tasks arrive
        CloseBatch(mStagingBatch);
    }
    mFlushQueries = true;

    NVTX_RANGE_POP();
}

void BatchMaker::StopWork() noexcept
{
    NVTX_RANGE_PUSH("BatchMaker::StopWork");
    DLOG(INFO) << "Stop Work";

    std::unique_lock<std::mutex> lock(mMutex);
    mStopWork = true;
    mProducerCV.notify_all();
    mConsumerCV.notify_all();

    NVTX_RANGE_POP();
}

Batch* BatchMaker::GetBatch() noexcept
{
    NVTX_RANGE_PUSH("BatchMaker::GetBatch");
    DLOG(INFO) << "GetBatch";

    NVTX_RANGE_PUSH("BatchMaker::GetBatch lock acquire");
    std::unique_lock<std::mutex> lock(mMutex);
    NVTX_RANGE_POP();

    if (mReadyBatches.empty())
    {
        NVTX_RANGE_PUSH("Waiting for ready batch");

        if (mStagingBatch) // close previous staged batches
        {
            CloseBatch(mStagingBatch);
        }

        ++mWaitingBatches;
        DLOG(INFO) << "GetBatch Waiting";
        mConsumerCV.wait(lock, [&] { return !mReadyBatches.empty() || mStopWork; });
        DLOG(INFO) << "GetBatch notified";
        --mWaitingBatches;

        NVTX_RANGE_POP();
    }

    if (mStopWork)
    {
        NVTX_RANGE_POP();
        LOG(INFO) << "GetBatch Done";
        return nullptr;
    }

    Batch* readyBatch = mReadyBatches.front();
    NVTX_RANGE_PUSH(("Preparing batch " + std::string(readyBatch->mDebugId)).c_str());
    mReadyBatches.pop_front();
    const auto completedCopies = readyBatch->getCompletedCopies();
    mReadiedPairs += completedCopies;

    DLOG(INFO) << "Sending " << readyBatch->mDebugId << "; batchSize : " << readyBatch->getCompletedCopies()
               << " mReadiedPairs : " << mReadiedPairs << " mIssuedPairs : " << mIssuedPairs;
    CHECK_LE(mReadiedPairs, mIssuedPairs);

    if (!mReadyBatches.empty())
    {
        mConsumerCV.notify_one(); // unblock another batch maker
    }

    NVTX_RANGE_POP();
    NVTX_RANGE_POP();

    return readyBatch;
}

void BatchMaker::HandleReadyBatch(Batch* batch) noexcept
{
    DLOG(INFO) << "Ready : " << batch->mDebugId << " " << batch->getCompletedCopies();

    const auto actualBatchSize = batch->getCompletedCopies();
    CHECK(batch->isComplete());
    CHECK_GT(actualBatchSize, 0);

    mReadyBatches.push_back(batch);
    mConsumerCV.notify_one();
}

void BatchMaker::NotifyH2D(Batch* batch) noexcept
{
    NVTX_RANGE_PUSH(("BatchMaker::NotifyH2D for batch " + batch->mDebugId).c_str());
    DLOG(INFO) << "Notify " << batch->mDebugId << "; flush = " << mFlushQueries;

    // reset and put back on idle batch queue
    batch->reset();
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mIdleBatches.push_back(batch);
    }

    // unblock batch loaders
    mProducerCV.notify_one();
    NVTX_RANGE_POP();
}

void BatchMaker::StageBatch(const int threadIdx) noexcept
{
    if (UseNuma())
    {
        bindNumaMemPolicy(mNumaIdx, mNumaNum);
    }

    while (true)
    {
        NVTX_RANGE_PUSH("BatchMaker::StageBatch iteration");

        Batch* batchPtr = nullptr;
        size_t pairsToCopy = 0;
        size_t tasksOffset = 0;
        size_t pairsOffset = 0;
        std::vector<DLRMv2Task> committedTasks;

        // de-queue tasks and commit copies, under lock
        {
            NVTX_RANGE_PUSH("BatchMaker::StageBatch acquire lock");
            std::unique_lock<std::mutex> lock(mMutex);

            DLOG(INFO) << "StageBatch waiting";
            mProducerCV.wait(
                lock, [&] { return ((mStagingBatch || !mIdleBatches.empty()) && !mTasksQ.empty()) || mStopWork; });
            DLOG(INFO) << "StageBatch notified";
            NVTX_RANGE_POP();

            if (mStopWork)
            {
                NVTX_RANGE_POP();
                return;
            }

            if (!mStagingBatch)
            {
                CHECK(!mIdleBatches.empty());
                mStagingBatch = mIdleBatches.back();
                mIdleBatches.pop_back();
                DLOG(INFO) << "Stage Batch set to " << mStagingBatch->mDebugId;
            }

            // mStagingBatch may change once lock is released, so store a copy
            batchPtr = mStagingBatch;
            tasksOffset = batchPtr->getTasks().size();
            pairsOffset = batchPtr->getCommittedCopies();
            const auto maxPairs = std::min(mMaxPairsPerThread, batchPtr->getFreeSpace());
            NVTX_RANGE_PUSH(
                ("Committing max " + std::to_string(maxPairs) + " pairs to batch " + batchPtr->mDebugId).c_str());
            DLOG(INFO) << "thread " << threadIdx << " with " << batchPtr->mDebugId << ":+" << pairsOffset;

            while (!mTasksQ.empty())
            {
                const auto task = mTasksQ.front();
                const auto numPairs = task.numIndividualPairs;

                if (numPairs + pairsToCopy > maxPairs)
                {
                    if (numPairs >= batchPtr->getFreeSpace())
                    {
                        // batch can't fit next sample
                        DLOG(INFO) << pairsToCopy << " : Break because full";
                        CloseBatch(batchPtr);
                    }

                    else
                    {
                        DLOG(INFO) << pairsToCopy << " : Break because maxPairs";
                    }

                    // Let some other thread commit remaining tasks
                    break;
                }

                pairsToCopy += numPairs;
                batchPtr->commitCopies(numPairs);
                batchPtr->pushTask(task);
                mTasksQ.pop_front();
            }

            if (pairsToCopy == 0)
            {
                NVTX_RANGE_POP();
                NVTX_RANGE_POP();
                continue;
            }

            DLOG(INFO) << "Commit " << pairsToCopy << " pairs in " << committedTasks.size() << " tasks";

            if ((mTasksQ.empty() && mFlushQueries) || mWaitingBatches)
            {
                // no more queries
                CloseBatch(batchPtr);
            }

            mProducerCV.notify_one();
            NVTX_RANGE_POP();
        }

        // do H2H, without lock
        batchPtr->doCopy(tasksOffset, pairsOffset, pairsToCopy);

        // mark copy as complete, under lock
        {
            NVTX_RANGE_PUSH("BatchMaker::StageBatch complete copy aquire lock");
            std::unique_lock<std::mutex> lock(mMutex);
            NVTX_RANGE_POP();

            NVTX_RANGE_PUSH("BatchMaker::StageBatch complete copy");
            batchPtr->completeCopies(pairsToCopy);
            ReadyBatchIfComplete(batchPtr);
            mConsumerCV.notify_one(); // TODO(vir): is this needed?
            NVTX_RANGE_POP();
        }

        NVTX_RANGE_POP();
    }

    if (UseNuma())
    {
        resetNumaMemPolicy();
    }
}

void BatchMaker::CloseBatch(Batch* batch) noexcept
{
    if (!batch->isReadyWhenComplete())
    {
        DLOG(INFO) << batch->mDebugId << " closing";
        // batch will move to mReadyBatches once copies complete
        batch->markReadyWhenComplete();
        // if already complete, move to ready
        ReadyBatchIfComplete(batch);
        // next thread will set new staging batch
        mStagingBatch = nullptr;
    }
}

void BatchMaker::ReadyBatchIfComplete(Batch* batch) noexcept
{
    if (batch->isComplete())
    {
        HandleReadyBatch(batch);
    }
}
