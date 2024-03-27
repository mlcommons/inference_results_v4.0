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

#include "llm_server.hpp"
#include "glog/logging.h"
#include "llm_utils.hpp"
#include "loadgen.h"

#include <fstream>
#include <limits>
#include <set>

std::string const& LLMServer::Name()
{
    return mName;
}

// Waiting for LLMCore to finish executing all the tasks
void LLMServer::waitForLLMCoreComplete()
{
    for (size_t idx = 0; idx < mLLMCoreVec.size(); ++idx)
    {
        for (int32_t llmCoreIdx = 0; llmCoreIdx < mNumLLMCoresPerDevice; ++llmCoreIdx)
        {
            mLLMCoreVec[idx][llmCoreIdx]->waitUntilTerminate();
        }
    }
}

// Send all pending queries in the Response Queue to the LoadGen
void LLMServer::sendQueryComplete(int32_t const threadIdx)
{
    CHECK(mIsResponder) << "Rank " << mMyRank << ":: Query Completions should be initiated from responder process";

    uint32_t nbSamplesProcessed = 0;
    while (true)
    {
        std::deque<LLMResponse> responses;
        {
            std::unique_lock<std::mutex> lck((*mRespQueueMtxs)[threadIdx]);
            // Wait until the response queue is not empty, or the stop signal
            (*mRespQueueCondVars)[threadIdx].wait(
                lck, [&]() { return !(*mRespQueue)[threadIdx].empty() || mServerStopWork; });
            if (mServerStopWork)
            {
                VLOG(1) << "Rank " << mMyRank << ":: " << mName << " exits from sendQueryComplete thread";
                break;
            }

            responses = std::deque<LLMResponse>(std::make_move_iterator((*mRespQueue)[threadIdx].begin()),
                std::make_move_iterator((*mRespQueue)[threadIdx].end()));
            (*mRespQueue)[threadIdx].clear();
        }

        for (auto& resp : responses)
        {
            if (mUseMPI)
            {
                sendQueryCompleteMPI(resp, threadIdx);
            }
            else
            {
                mlperf::QuerySamplesComplete(&(resp.QSR), 1);
            }

            ++nbSamplesProcessed;
            if ((nbSamplesProcessed % 1000 == 0) && !mUseMPI)
            {
                LOG(INFO) << "Rank " << mMyRank << ":: Processed " << nbSamplesProcessed << " samples.";
            }
        }
        VLOG(1) << "Rank " << mMyRank << ":: Response queue idx: " << threadIdx << " finished " << responses.size()
                << " sample. Total finished now: " << nbSamplesProcessed;
    }
    if (!mUseMPI)
    {
        VLOG(1) << "QuerySamplesCompelete threadIdx: " << threadIdx << " completed " << nbSamplesProcessed
                << " samples.";
    }
}

void LLMServer::sendQueryCompleteMPI(LLMResponse resp, int32_t const threadIdx)
{
    auto uniqueId = resp.QSR.id;
    auto count = resp.QSR.size / sizeof(int32_t);
    auto bufPtr = resp.outputTokens.data();
    int32_t idTag = kMPI_ID_TAG;
    int32_t dataTag = kMPI_DATA_TAG;

    // Send through MPI; any send can be blocking
    // Standard MPI send is optimized by backend and is likely be asynchronous regardless of it being blocking
    // NOTE: Sending it two steps enables below while being performant still through backend optimizations
    //       * First msg can be ID and at the same time, a special msg (like shutdown msg)
    //       * First msg can help preparing second msg as these msgs are queued up and not consumed right away
    //       * First msg and second msg can be in different types, and MPI backend takes care of complications,
    //         otherwise, user has to handle byte ordering differences, bit representations etc manually
    //       * No need to copy any long data already stored in memory, in order to pack them together (any transfer
    //         should happen for data in contiguous memory)
    MPICHECK(MPI_Send(&uniqueId, 1, MPI_UINT64_T, 1, idTag, *(mRespComms[threadIdx])));
    MPICHECK(MPI_Send(bufPtr, count, MPI_INT32_T, 1, dataTag, *(mRespComms[threadIdx])));
    VLOG(1) << "Rank " << mMyRank << ":: Sent msg starting with " << resp.outputTokens[0] << " uniqueID: " << uniqueId
            << "; msg -- COUNT: " << count << ", TAG: " << idTag << "+" << dataTag;
}

// Send all pending queries in the Response Queue to the LoadGen
void LLMServer::sendFirstTokComplete(int32_t const threadIdx)
{
    CHECK(mIsResponder) << "Rank " << mMyRank
                        << ":: first token Completions should be initiated from responder process";

    uint32_t nbSamplesProcessed = 0;
    while (true)
    {
        std::deque<LLMResponse> responses;
        {
            std::unique_lock<std::mutex> lck((*mFirstTokQueueMtxs)[threadIdx]);
            // Wait until the first-token queue is not empty, or the stop signal
            (*mFirstTokQueueCondVars)[threadIdx].wait(
                lck, [&]() { return !(*mFirstTokQueue)[threadIdx].empty() || mServerStopWork; });
            if (mServerStopWork)
            {
                VLOG(1) << "Rank " << mMyRank << ":: " << mName << " exits from sendFirstTokComplete thread";
                break;
            }

            responses = std::deque<LLMResponse>(std::make_move_iterator((*mFirstTokQueue)[threadIdx].begin()),
                std::make_move_iterator((*mFirstTokQueue)[threadIdx].end()));
            (*mFirstTokQueue)[threadIdx].clear();
        }

        for (auto& resp : responses)
        {
            if (mUseMPI)
            {
                sendFirstTokCompleteMPI(resp, threadIdx);
            }
            else
            {
                mlperf::FirstTokenComplete(&(resp.QSR), 1);
            }

            ++nbSamplesProcessed;
            if ((nbSamplesProcessed % 1000 == 0) && !mUseMPI)
            {
                LOG(INFO) << "Rank " << mMyRank << ":: Processed " << nbSamplesProcessed << " first tokens.";
            }
        }
        VLOG(1) << "Rank " << mMyRank << ":: first token queue idx: " << threadIdx << " finished " << responses.size()
                << " first tokens. Total finished now: " << nbSamplesProcessed;
    }
    if (!mUseMPI)
    {
        VLOG(1) << "FirstTokenCompelete threadIdx: " << threadIdx << " completed " << nbSamplesProcessed << " samples.";
    }
}

void LLMServer::sendFirstTokCompleteMPI(LLMResponse resp, int32_t const threadIdx)
{
    auto uniqueId = resp.QSR.id;
    auto count = resp.QSR.size / sizeof(int32_t);
    CHECK_EQ(count, 1);
    auto bufPtr = resp.outputTokens.data();
    int32_t idTag = kMPI_ID_TAG;
    int32_t dataTag = kMPI_DATA_TAG;

    MPICHECK(MPI_Send(&uniqueId, 1, MPI_UINT64_T, 1, idTag, *(mFirstTokComms[threadIdx])));
    MPICHECK(MPI_Send(bufPtr, count, MPI_INT32_T, 1, dataTag, *(mFirstTokComms[threadIdx])));
    VLOG(1) << "Rank " << mMyRank << ":: Sent first token starting with " << resp.outputTokens[0]
            << " uniqueID: " << uniqueId << "; msg -- COUNT: " << count << ", TAG: " << idTag << "+" << dataTag;
}

void LLMServer::sendTerminateMPI(int32_t targetRank, MPI_Comm const& comm)
{
    uint64_t id = kTERMINATION_KEY;
    int32_t size = 1;
    int32_t buf = 0;
    int32_t idTag = kMPI_ID_TAG;

    // Send through MPI; any send can be blocking
    MPICHECK(MPI_Send(&id, size, MPI_UINT64_T, targetRank, idTag, comm));
}

void LLMServer::handleReclaimBuffer(int32_t const threadIdx)
{
    CHECK(mIsResponder) << "Rank " << mMyRank << ":: Reclaiming buffer should be initiated from the responder process";
    CHECK(!mIsReceiver) << "Rank " << mMyRank
                        << ":: Reclaiming buffer happens locally if the responder is also the receiver";

    while (true)
    {
        std::deque<uintptr_t> resources;
        {
            std::unique_lock<std::mutex> lck((*mReclaimQueueMtxs)[threadIdx]);
            // Wait until the response queue is not empty, or the stop signal
            (*mReclaimQueueCondVars)[threadIdx].wait(
                lck, [&]() { return !(*mReclaimQueue)[threadIdx].empty() || mServerStopWork; });
            if (mServerStopWork)
            {
                VLOG(1) << "Rank " << mMyRank << ":: " << mName << " exits from handleReclaimBuffer thread";
                break;
            }

            resources = (*mReclaimQueue)[threadIdx];
            (*mReclaimQueue)[threadIdx].clear();
        }

        for (auto& resource : resources)
        {
            uint64_t uniqueId = reinterpret_cast<uint64_t>(resource);
            int32_t idTag = kMPI_ID_TAG;

            // Send through MPI; any send can be blocking
            // Receiver is Leader whose local rank is 0
            MPICHECK(MPI_Send(&uniqueId, 0, MPI_UINT64_T, 0, idTag, *(mReclaimComms[threadIdx])));
        }
    }
}

void LLMServer::waitForTermination(int32_t const threadIdx)
{
    CHECK(mIsResponder) << "Rank " << mMyRank << ":: Only responder is wait for Termination";

    MPI_Message msg;
    MPI_Status status;
    int32_t count;
    uintptr_t uniqueId;
    int32_t idTag = kMPI_ID_TAG;

    // Within the Comm, worker rank is always 0 and LoadGen rank is always 1
    // The only time worker rank receives MPI messages in the respcomm from loadgen is termination
    MPICHECK(MPI_Mprobe(1, idTag, *(mRespComms[threadIdx]), &msg, &status));
    MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
    CHECK_EQ(count, 1) << "Unexpected UniqueID count";
    MPICHECK(MPI_Mrecv(&uniqueId, count, MPI_UINT64_T, &msg, &status));

    // EXIT condition: receiving TERMINATE msg from the loadgen rank
    if (uniqueId == kTERMINATION_KEY)
    {
        VLOG(1) << "Rank " << mMyRank << " Termination msg received";
        // Unblock the LoadGen process RECV blocking for response
        sendTerminateMPI(1, *(mRespComms[threadIdx]));
        if (mLLMConfig.mReportFirstToken)
        {
            sendTerminateMPI(1, *(mFirstTokComms[threadIdx]));
        }
        if (!mIsReceiver)
        {
            // Unblock RECV for reclaim buffer resource
            sendTerminateMPI(0, *(mReclaimComms[threadIdx]));
        }
        stopSUT();

        return;
    }

    CHECK(false) << "Termination failed";
}

// Wait until Server stop signal is set
void LLMServer::waitSutCompletion()
{
    if (mIsReceiver || mIsResponder)
    {
        while (!mServerStopWork)
        {
            sleep(1);
        }
    }
}

// stop SUT
void LLMServer::stopSUT()
{
    VLOG(1) << "Rank " << mMyRank << " stop SUT called";

    mServerStopWork = true;

    for (auto& i : *mRespQueueCondVars)
    {
        i.notify_all();
    }
    for (auto& i : *mFirstTokQueueCondVars)
    {
        i.notify_all();
    }
    for (auto& i : *mRequestQueueCondVars)
    {
        i.notify_all();
    }
    for (auto& i : *mReclaimQueueCondVars)
    {
        i.notify_all();
    }
    VLOG(1) << "Rank " << mMyRank << " stop SUT done";
}

LLMServer::LLMServer(std::string const& name, std::string const& enginePath, DevIdVec const& gpuDeviceIds,
    DevIdVec const& grpDevIds, int32_t numLLMCoresPerDevice, LLMConfig const& LLMConfig, int32_t serverNumWorkerThreads,
    std::vector<std::shared_ptr<MPI_Comm>> respComms, std::vector<std::shared_ptr<MPI_Comm>> firstTokComms,
    std::vector<std::shared_ptr<MPI_Comm>> reclaimComms, bool isReceiver, bool isResponder, bool useMPI, int32_t myRank,
    bool verboseNVTX)
    : mName(name)
    , mNumLLMCoresPerDevice(numLLMCoresPerDevice)
    , mLLMConfig(LLMConfig)
    , mRespComms(respComms)
    , mFirstTokComms(firstTokComms)
    , mReclaimComms(reclaimComms)
    , mIsReceiver(isReceiver)
    , mIsResponder(isResponder)
    , mUseMPI(useMPI)
    , mMyRank(myRank)
    , mServerStopWork(false)
    , mTotalSampleConsumed(0)
{
    // For now, one MPI endpoint per inference group
    CHECK((mUseMPI && (serverNumWorkerThreads == 1)) || !mUseMPI);

    VLOG(1) << "LLMServer[" << mName << "] being constructed...";

    // serverNumWorkerThreads >= 1 means creating multiple RequestQueue/RespQueue and share among LLMCores
    // This is useful when we require high parallelism for issueQuery (when the inference is fast).
    // LLM model has long inference latencies, thus this is temporarily not needed.
    if ((!mUseMPI) && (serverNumWorkerThreads >= 1))
    {
        CHECK(false) << "serverNumIssueQueryThreads >= 1 for SP not implemented!";
    }
    else
    {
        mRequestQueue = std::make_shared<RequestQueueVec>(1);
        mRequestQueueMtxs = std::make_shared<MutexVec>(1);
        mRequestQueueCondVars = std::make_shared<CondVarVec>(1);

        mRespQueue = std::make_shared<RespQueueVec>(1);
        mRespQueueMtxs = std::make_shared<MutexVec>(1);
        mRespQueueCondVars = std::make_shared<CondVarVec>(1);

        mFirstTokQueue = std::make_shared<RespQueueVec>(1);
        mFirstTokQueueMtxs = std::make_shared<MutexVec>(1);
        mFirstTokQueueCondVars = std::make_shared<CondVarVec>(1);

        mReclaimQueue = std::make_shared<ReclaimQueueVec>(1);
        mReclaimQueueMtxs = std::make_shared<MutexVec>(1);
        mReclaimQueueCondVars = std::make_shared<CondVarVec>(1);
    }

    // Start threads
    mRespQueueThreads.clear();
    mFirstTokQueueThreads.clear();
    mReclaimBufferThreads.clear();
    mWaitForTerminationThreads.clear();
    if (mIsResponder)
    {
        mRespQueueThreads.reserve(mRespQueue->size());
        for (int32_t threadIdx = 0; threadIdx < mRespQueue->size(); ++threadIdx)
        {
            mRespQueueThreads.emplace_back(&LLMServer::sendQueryComplete, this, threadIdx);
            if (mUseMPI)
            {
                if (!mIsReceiver)
                {
                    mReclaimBufferThreads.emplace_back(&LLMServer::handleReclaimBuffer, this, threadIdx);
                }
                mWaitForTerminationThreads.emplace_back(&LLMServer::waitForTermination, this, threadIdx);
            }
        }

        if (mLLMConfig.mReportFirstToken)
        {
            mFirstTokQueueThreads.reserve(mFirstTokQueue->size());
            for (int32_t threadIdx = 0; threadIdx < mFirstTokQueue->size(); ++threadIdx)
            {
                mFirstTokQueueThreads.emplace_back(&LLMServer::sendFirstTokComplete, this, threadIdx);
            }
        }
    }

    // Divide all LLMCores among the number of IssueQuery threads
    int32_t LLMCoresPerQThread = (serverNumWorkerThreads == 0)
        ? std::numeric_limits<int32_t>::max()
        : (gpuDeviceIds.size() * mNumLLMCoresPerDevice) / serverNumWorkerThreads;
    int32_t counter = 0;
    int32_t qThreadIdx = 0;

    // Create mNumLLMCoresPerDevice LLMCores on each gpu and store them in a vector
    mLLMCoreVec.resize(gpuDeviceIds.size());
    for (size_t idx = 0; idx < gpuDeviceIds.size(); ++idx)
    {
        for (int32_t llmCoreIdx = 0; llmCoreIdx < mNumLLMCoresPerDevice; ++llmCoreIdx)
        {
            auto deviceId = gpuDeviceIds[idx];
            CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
            auto& groupDeviceIds = mUseMPI ? grpDevIds : DevIdVec{deviceId};
            LOG(INFO) << "LLMServer[" << mName << "] creating LLMCore[" << idx << "] on Device[" << deviceId << "]...";
            mLLMCoreVec[idx].push_back(std::make_shared<LLMCore>(LLMConfig, enginePath, llmCoreIdx, deviceId,
                groupDeviceIds, qThreadIdx, mMyRank, mIsReceiver, mIsResponder, mRequestQueue, mRequestQueueMtxs,
                mRequestQueueCondVars, mRespQueue, mRespQueueMtxs, mRespQueueCondVars, mFirstTokQueue,
                mFirstTokQueueMtxs, mFirstTokQueueCondVars, mReclaimQueue, mReclaimQueueMtxs, mReclaimQueueCondVars));
            LOG(INFO) << "LLMServer[" << mName << "] created LLMCore[" << idx << "] on Device[" << deviceId << "].";

            ++counter;
            if (counter == LLMCoresPerQThread)
            {
                ++qThreadIdx;
                counter = 0;
            }
        }
    }
    VLOG(1) << "LLMServer[" << mName << "] constructed.";
}

LLMServer::~LLMServer()
{
    VLOG(1) << "Destructing LLMServer[" << mName << "]...";
    if (!mUseMPI)
    {
        stopSUT();
    }

    // Notify all LLMCore that the stop signal is received
    for (size_t deviceIdx = 0; deviceIdx < mLLMCoreVec.size(); ++deviceIdx)
    {
        for (size_t LLMCoreIdx = 0; LLMCoreIdx < mLLMCoreVec[deviceIdx].size(); ++LLMCoreIdx)
        {
            mLLMCoreVec[deviceIdx][LLMCoreIdx]->notifyCoreComplete();
        }
    }

    if (!mRespQueueThreads.empty())
    {
        for (auto& thread : mRespQueueThreads)
        {
            thread.join();
        }
    }
    if (!mFirstTokQueueThreads.empty())
    {
        for (auto& thread : mFirstTokQueueThreads)
        {
            thread.join();
        }
    }
    if (!mReclaimBufferThreads.empty())
    {
        for (auto& thread : mReclaimBufferThreads)
        {
            thread.join();
        }
    }
    if (!mWaitForTerminationThreads.empty())
    {
        for (auto& thread : mWaitForTerminationThreads)
        {
            thread.join();
        }
    }

    // Graceful exit of TRTLLM
    waitForLLMCoreComplete();

    VLOG(1) << "Done destructing LLMServer[" << mName << "].";
}

// Sort samples in the descending order of input length
std::vector<std::pair<int32_t, int32_t>> LLMServerLeader::sortSamples(
    std::vector<mlperf::QuerySample> const& samples) const
{
    std::vector<std::pair<int32_t, int32_t>> sequenceSamplePosAndLength(samples.size());
    for (size_t samplePos = 0; samplePos < samples.size(); ++samplePos)
    {
        sequenceSamplePosAndLength[samplePos] = std::make_pair(
            samplePos, *static_cast<int32_t*>(mQsl->GetSampleAddress(samples[samplePos].index, kINPUT_LENGTH_IDX)));
    }

    // Sort the queries in descending order.
    if (mLLMConfig.mEnableSort)
    {
        std::sort(sequenceSamplePosAndLength.begin(), sequenceSamplePosAndLength.end(),
            [](const std::pair<int32_t, int32_t>& a, const std::pair<int32_t, int32_t>& b) -> bool {
                return a.second > b.second;
            });
        VLOG(1) << "Sorted " << samples.size() << " sample(s).";
    }
    return sequenceSamplePosAndLength;
}

// SUT override IssueQuery entrance.
// Loadgen issues queries to RequestQueue
void LLMServerLeader::IssueQuery(std::vector<mlperf::QuerySample> const& samples)
{
    CHECK(!mUseMPI) << "MPI is used -- LoadGen should NOT call this directly";

    VLOG(1) << "Issuing queries for " << samples.size() << " samples.";

    // find which task vector to enqueue tasks
    int32_t qThreadIdx = mIssueQueryThreadMap[std::this_thread::get_id()];

    // push queries from LoadGen to mRequestQueue
    NVTX_MARK("IssueQuery", nvtx::COLOR_BLUE_2);

    // sort samples in the descending order of input length
    // TODO: Investigate load balancing with multiple issue threads/queues
    std::vector<std::pair<int32_t, int32_t>> sequenceSamplePosAndLength = sortSamples(samples);

    // Transfer queries to the RequestQueue to be consumed by the LLMCore.
    for (size_t beginSamplePos = 0; beginSamplePos < sequenceSamplePosAndLength.size(); beginSamplePos += mGpuBatchSize)
    {
        int32_t actualBatchSize
            = std::min(mGpuBatchSize, static_cast<int32_t>(sequenceSamplePosAndLength.size() - beginSamplePos));
        mTotalSampleConsumed += actualBatchSize;
        {
            std::scoped_lock<std::mutex> lck((*mRequestQueueMtxs)[qThreadIdx]);
            std::stringstream seqLenStream;
            seqLenStream << "[";
            for (int32_t i = 0; i < actualBatchSize; ++i)
            {
                auto samplePos = sequenceSamplePosAndLength[beginSamplePos + i].first;
                auto inputSeqLen = sequenceSamplePosAndLength[beginSamplePos + i].second;
                auto inputPtr
                    = reinterpret_cast<uintptr_t>(mQsl->GetSampleAddress(samples[samplePos].index, kINPUT_TOKEN_IDX));
                seqLenStream << inputSeqLen << " ";
                (*mRequestQueue)[qThreadIdx].emplace_back(samples[samplePos], inputSeqLen, inputPtr);
            }
            seqLenStream << "]";
            VLOG(1) << "Thread " << qThreadIdx << " inserted " << actualBatchSize
                    << " queries into mRequestQueue. It has size: " << (*mRequestQueue)[qThreadIdx].size()
                    << " and lengths: " << seqLenStream.str();
            // notify LLMCores to consume tasks
            (*mRequestQueueCondVars)[qThreadIdx].notify_one();
        }
    }
}

// SUT override FlushQueries entrance
void LLMServerLeader::FlushQueries() {}

// SUT register issue thread for server scenario
void LLMServerLeader::startIssueThread(int32_t threadIdx)
{
    // issue query thread
    {
        CHECK_EQ(!mRequestQueueMtxs->empty(), true);
        std::lock_guard<std::mutex> lock((*mRequestQueueMtxs)[threadIdx]);
        mIssueQueryThreadMap[std::this_thread::get_id()] = threadIdx;
    }
    mlperf::RegisterIssueQueryThread();
}

// Waiting for MPI recv to receive samples and then push them to incoming buffer queue
void LLMServerLeader::startRecvMPIThread(int32_t threadIdx)
{
    MPI_Message msg;
    MPI_Status status;
    int32_t count;
    uintptr_t uniqueId;
    size_t bufferIdx;
    void* bufferPtr;
    int32_t idTag = kMPI_ID_TAG;
    int32_t dataTag = kMPI_DATA_TAG;
    while (true)
    {
        // Blocking is okay; terminate message is expected to arrive here
        // Within the Comm, worker rank is always 0 and LoadGen rank is always 1
        MPICHECK(MPI_Mprobe(1, idTag, *(mReqComms[threadIdx]), &msg, &status));
        MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
        CHECK_EQ(count, 1) << "Unexpected UniqueID count";
        MPICHECK(MPI_Mrecv(&uniqueId, count, MPI_UINT64_T, &msg, &status));

        // EXIT condition, from receiving TERMINATE msg
        if (uniqueId == kTERMINATION_KEY)
        {
            LOG(INFO) << "Rank " << mMyRank << ":: " << mName << " unblocked from RECV to shut-down";
            if (!mIsResponder)
            {
                stopSUT();
            }
            break;
        }

        // prep recv'ing data
        MPICHECK(MPI_Mprobe(1, dataTag, *(mReqComms[threadIdx]), &msg, &status));
        MPICHECK(MPI_Get_count(&status, MPI_INT32_T, &count));
        bufferIdx = mRecvBuffer.getAvailBufferIdx();
        bufferPtr = mRecvBuffer.commitBuffer(bufferIdx, count, uniqueId);
        MPICHECK(MPI_Mrecv(bufferPtr, count, MPI_INT32_T, &msg, &status));
        VLOG(1) << "Rank " << mMyRank << ":: Received msg starting with " << mRecvBuffer.getBufferContent(bufferIdx, 0)
                << " uniqueId: " << uniqueId << " from local_comm_rank: " << status.MPI_SOURCE << "; msg -- COUNT: " 
                << count << ", TAG: " << idTag << "+" << dataTag << ", ERR: " << status.MPI_ERROR;

        // make LLMSample and queue it
        mlperf::QuerySample qs;
        qs.id = uniqueId;
        qs.index = 0;
        {
            std::scoped_lock<std::mutex> lck((*mRequestQueueMtxs)[threadIdx]);
            (*mRequestQueue)[threadIdx].emplace_back(
                qs, static_cast<size_t>(count), reinterpret_cast<uintptr_t>(bufferPtr));
            VLOG(1) << "Rank " << mMyRank << " Queue " << threadIdx
                    << " Received a query sample. Current size: " << (*mRequestQueue)[threadIdx].size();
            (*mRequestQueueCondVars)[threadIdx].notify_one();
        }
    }
}

// Reclaim resources used for holding incoming data through MPI
void LLMServerLeader::reclaimResourcesMPI(int32_t const threadIdx)
{
    CHECK(!mIsResponder)
        << "Rank " << mMyRank
        << ":: Reclaiming comm buffer resource should happen through MPI only if the Leader is not the responder";
    CHECK_NE(mResponderSessionRank, mMyRank);

    MPI_Message msg;
    MPI_Status status;
    int32_t count;
    uint64_t uniqueId;
    int32_t idTag = kMPI_ID_TAG;

    while (true)
    {
        // Blocking is okay; terminate message is expected to arrive here
        MPICHECK(MPI_Mprobe(mResponderSessionRank, idTag, *(mReclaimComms[threadIdx]), &msg, &status));
        MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
        CHECK_EQ(count, 1) << "Unexpected UniqueID count";
        MPICHECK(MPI_Mrecv(&uniqueId, count, MPI_UINT64_T, &msg, &status));
        // EXIT condition, from receiving TERMINATE msg
        if (uniqueId == kTERMINATION_KEY)
        {
            LOG(INFO) << "Rank " << mMyRank << ":: " << mName << " unblocked from RECV for shut-down";
            break;
        }

        mRecvBuffer.cleanBuffer(reinterpret_cast<uintptr_t>(uniqueId));
        VLOG(1) << "Rank " << mMyRank << ":: Reclaimed buffer for uniqueId: " << uniqueId;
    }
}

// Reclaim resources for holding incoming data locally
void LLMServerLeader::reclaimResources(int32_t const threadIdx)
{
    CHECK(mIsResponder)
        << "Rank " << mMyRank
        << ":: Reclaiming comm buffer resource should happen locally only if the Leader is also the responder";

    while (true)
    {
        std::deque<uintptr_t> resources;
        {
            std::unique_lock<std::mutex> lck((*mReclaimQueueMtxs)[threadIdx]);
            // Wait until the response queue is not empty, or the stop signal
            (*mReclaimQueueCondVars)[threadIdx].wait(
                lck, [&]() { return !(*mReclaimQueue)[threadIdx].empty() || mServerStopWork; });
            if (mServerStopWork)
            {
                VLOG(1) << "Rank " << mMyRank << ":: " << mName << " exits from reclaimResources thread";
                break;
            }
            resources = (*mReclaimQueue)[threadIdx];
            (*mReclaimQueue)[threadIdx].clear();
        }

        for (auto& resource : resources)
        {
            mRecvBuffer.cleanBuffer(resource);
            VLOG(1) << "Rank " << mMyRank << ":: Reclaimed buffer for uniqueId: " << resource;
        }
    }
}

LLMServerLeader::LLMServerLeader(std::string const& name, std::shared_ptr<qsl::SampleLibrary> qsl,
    std::string const& enginePath, DevIdVec const& gpuDeviceIds, DevIdVec const& grpDevIds,
    int32_t numLLMCoresPerDevice, int32_t gpuBatchSize, LLMConfig const& LLMConfig, double targetLatencyPercentile,
    int32_t serverNumWorkerThreads, std::vector<std::shared_ptr<MPI_Comm>> reqComms,
    std::vector<std::shared_ptr<MPI_Comm>> respComms, std::vector<std::shared_ptr<MPI_Comm>> firstTokComms,
    std::vector<std::shared_ptr<MPI_Comm>> reclaimComms, bool isResponder, bool useMPI, int32_t myRank,
    int32_t responderLocalRank, int32_t numTrxInflight, bool verboseNVTX)
    : LLMServer(name, enginePath, gpuDeviceIds, grpDevIds, numLLMCoresPerDevice, LLMConfig, serverNumWorkerThreads,
        respComms, firstTokComms, reclaimComms, /* isReceiver */ true, isResponder, useMPI, myRank, verboseNVTX)
    , mQsl(qsl)
    , mGpuBatchSize(gpuBatchSize)
    , mTargetLatencyPercentile(targetLatencyPercentile)
    , mReqComms(reqComms)
    , mResponderSessionRank(responderLocalRank)
    , mRecvBuffer(useMPI ? numTrxInflight : 0,
          useMPI ? mLLMConfig.mExcludeInputInOutput ? kMAX_BUFFER_BYTESIZE_PER_TRANSACTION
                                                    : kMAX_BUFFER_BYTESIZE_PER_TRANSACTION_WITH_INPUT
                 : 0,
          name + "-ReqRecvBuf")
{
    VLOG(1) << "LLMServerLeader[" << mName << "] being constructed...";

    // start threads
    mRecvQueryThreads.clear();
    mReclaimResourceThreads.clear();
    mIssueQueryThreads.clear();
    for (int i = 0; i < serverNumWorkerThreads; ++i)
    {
        if (mUseMPI)
        {
            mRecvQueryThreads.emplace_back(&LLMServerLeader::startRecvMPIThread, this, i);
            if (mIsResponder)
            {
                mReclaimResourceThreads.emplace_back(&LLMServerLeader::reclaimResources, this, i);
            }
            else
            {
                mReclaimResourceThreads.emplace_back(&LLMServerLeader::reclaimResourcesMPI, this, i);
            }
        }
        else
        {
            mIssueQueryThreads.emplace_back(&LLMServerLeader::startIssueThread, this, i);
        }
    }

    VLOG(1) << "LLMServerLeader[" << mName << "] constructed.";
}

LLMServerLeader::~LLMServerLeader()
{
    VLOG(1) << "Destructing LLMServerLeader[" << mName << "]...";

    if (!mRecvQueryThreads.empty())
    {
        for (auto& thread : mRecvQueryThreads)
        {
            thread.join();
        }
    }
    if (!mReclaimResourceThreads.empty())
    {
        for (auto& thread : mReclaimResourceThreads)
        {
            thread.join();
        }
    }
    if (!mIssueQueryThreads.empty())
    {
        for (auto& thread : mIssueQueryThreads)
        {
            thread.join();
        }
    }

    VLOG(1) << "Done destructing LLMServerLeader[" << mName << "].";
}

// Sort samples in the descending order of input length
std::vector<std::pair<int32_t, int32_t>> LLMServerLoadGen::sortSamples(
    std::vector<mlperf::QuerySample> const& samples) const
{
    std::vector<std::pair<int32_t, int32_t>> sequenceSamplePosAndLength(samples.size());
    for (size_t samplePos = 0; samplePos < samples.size(); ++samplePos)
    {
        sequenceSamplePosAndLength[samplePos] = std::make_pair(
            samplePos, *static_cast<int32_t*>(mQsl->GetSampleAddress(samples[samplePos].index, kINPUT_LENGTH_IDX)));
    }

    // Sort the queries in descending order.
    if (mEnableSort)
    {
        std::sort(sequenceSamplePosAndLength.begin(), sequenceSamplePosAndLength.end(),
            [](const std::pair<int32_t, int32_t>& a, const std::pair<int32_t, int32_t>& b) -> bool {
                return a.second > b.second;
            });
        VLOG(1) << "Sorted " << samples.size() << " sample(s).";
    }
    return sequenceSamplePosAndLength;
}

int32_t LLMServerLoadGen::getNextQueueIdx(size_t sample_size, int32_t idx)
{
    int32_t qThreadIdx;
    switch (mTrafficDistributionPolicy)
    {
    case (trafficDistributionPolicy::roundRobin): qThreadIdx = (mLastUsedQueueIdx + 1) % mNumWorkerThreads; break;
    case (trafficDistributionPolicy::staticChop):
    {
        qThreadIdx = idx < ((sample_size / mNumWorkerThreads) * mNumWorkerThreads)
            ? idx / (sample_size / mNumWorkerThreads)
            : (idx - ((sample_size / mNumWorkerThreads) * mNumWorkerThreads)) % mNumWorkerThreads;
        break;
    }
    case (trafficDistributionPolicy::loadBalancing): qThreadIdx = getLeastBusyIdx(); break;
    default: CHECK(false) << "Unknown traffic distribution policy: " << mTrafficDistributionPolicy; break;
    }
    mLastUsedQueueIdx = qThreadIdx;

    return qThreadIdx;
}

int32_t LLMServerLoadGen::getLeastBusyIdx()
{
    std::deque<std::atomic<uint32_t>>::iterator res
        = std::min_element(mNumInFlightTransactions.begin(), mNumInFlightTransactions.end());
    int32_t idx = static_cast<int32_t>(std::distance(mNumInFlightTransactions.begin(), res));
    VLOG(1) << "Rank " << mLoadgenRank << ":: LoadBalancingTrafficDistributor found the least busy Idx " << idx
            << " with " << *res << " transactions on the fly";
    return idx;
}

// SUT override IssueQuery entrance. Only used for MPI model.
// Loadgen issues queries to RequestQueue
void LLMServerLoadGen::IssueQuery(std::vector<mlperf::QuerySample> const& samples)
{
    VLOG(1) << "Rank " << mLoadgenRank << ":: Issuing queries for " << samples.size() << " samples.";

#if ENABLE_RECORDER == 1
    if (!mRecorder->mInitialized)
    {
        mRecorder->initialize();
    }
#endif

    // push queries from LoadGen to mRequestQueue
    NVTX_MARK("IssueQuery", nvtx::COLOR_BLUE_2);

    // sort samples in the descending order of input length
    std::vector<std::pair<int32_t, int32_t>> sequenceSamplePosAndLength = sortSamples(samples);

    // Transfer queries to the RequestQueue to be consumed by the LLMCore.
    for (int32_t i = 0; i < sequenceSamplePosAndLength.size(); ++i)
    {
        auto s = sequenceSamplePosAndLength[i];

        // If server scenario, multiple request streams come from LoadGen; else need to distribute
        // TODO: Investigate load balancing with multiple issue threads/queues

        // find which task vector to enqueue tasks
        int32_t qThreadIdx;

        if (mUseMultiIssueThreads)
        {
            qThreadIdx = mIssueQueryThreadMap[std::this_thread::get_id()];
        }
        else
        {
            qThreadIdx = getNextQueueIdx(samples.size(), i);
        }

        std::scoped_lock<std::mutex> lck((*mRequestQueueMtxs)[qThreadIdx]);
        std::stringstream seqLenStream;
        seqLenStream << "[";
        auto samplePos = s.first;
        auto inputSeqLen = s.second;
        auto inputPtr = reinterpret_cast<uintptr_t>(mQsl->GetSampleAddress(samples[samplePos].index, kINPUT_TOKEN_IDX));
        seqLenStream << inputSeqLen << " ";
        (*mRequestQueue)[qThreadIdx].emplace_back(samples[samplePos], inputSeqLen, inputPtr);
        seqLenStream << "]";
        VLOG(1) << "Rank " << mLoadgenRank << ":: Thread " << qThreadIdx << " inserted 1 out of " << samples.size()
                << " samples into mRequestQueue; after insertion, mRequestQueue size became: "
                << (*mRequestQueue)[qThreadIdx].size() << "; inserted sample's uniqueID: " << samples[samplePos].id
                << ", sequence lengths: " << seqLenStream.str();
        // notify insertion
        (*mRequestQueueCondVars)[qThreadIdx].notify_one();
#if ENABLE_RECORDER==1
        mRecorder->recordIssueQuery(samples[samplePos].id, inputSeqLen);
#endif
        mNumInFlightTransactions[qThreadIdx] += 1;
    }
}

// Override FlushQueries entrance
void LLMServerLoadGen::FlushQueries() {}

// Register issue thread for server scenario
void LLMServerLoadGen::startIssueThread(int32_t threadIdx)
{
    // issue query thread
    {
        CHECK_EQ(!mRequestQueueMtxs->empty(), true);
        std::lock_guard<std::mutex> lock((*mRequestQueueMtxs)[threadIdx]);
        mIssueQueryThreadMap[std::this_thread::get_id()] = threadIdx;
    }
    mlperf::RegisterIssueQueryThread();
}

// Sending samples collected from LoadGen's IssueQuery() calls through MPI send
void LLMServerLoadGen::sendQueryMPI(LLMSample sample, int32_t const threadIdx)
{
    auto uniqueId = sample.QS.id;
    auto count = reinterpret_cast<int32_t>(sample.SampleLen);
    auto bufPtr = reinterpret_cast<void*>(sample.SamplePtr);
    int32_t idTag = kMPI_ID_TAG;
    int32_t dataTag = kMPI_DATA_TAG;

    // Send through MPI; any send can be blocking
    // Standard MPI send is optimized by backend and is likely be asynchronous regardless of it being blocking
    // NOTE: Sending it two steps enables below while being performant still through backend optimizations
    //       * First msg can be ID and at the same time, a special msg (like shutdown msg)
    //       * First msg can help preparing second msg as these msgs are queued up and not consumed right away
    //       * First msg and second msg can be in different types, and MPI backend takes care of complications,
    //         otherwise, user has to handle byte ordering differences, bit representations etc manually
    //       * No need to copy any long data already stored in memory, in order to pack them together (any transfer
    //         should happen for data in contiguous memory)
    // Within the query/response Comm, worker rank is always 0 and LoadGen rank is always 1
    MPICHECK(MPI_Send(&uniqueId, 1, MPI_UINT64_T, 0, idTag, *(mReqComms[threadIdx])));
    MPICHECK(MPI_Send(bufPtr, count, MPI_INT32_T, 0, dataTag, *(mReqComms[threadIdx])));
#if ENABLE_RECORDER==1
        // This is a very rough estimation of the start inference time.
        mRecorder->recordStartInference(uniqueId);
#endif
    VLOG(1) << "Rank " << mLoadgenRank << ":: Sent msg starting with " << *reinterpret_cast<int32_t*>(sample.SamplePtr)
            << " uniqueID: " << uniqueId << "; msg -- COUNT: " << count << ", TAG: " << idTag << "+" << dataTag;
}

void LLMServerLoadGen::sendTerminateMPI(int32_t const targetRank, MPI_Comm const& comm)
{
    uint64_t id = kTERMINATION_KEY;
    int32_t size = 1;
    int32_t idTag = kMPI_ID_TAG;

    // Send through MPI; any send can be blocking
    MPICHECK(MPI_Send(&id, size, MPI_UINT64_T, targetRank, idTag, comm));
}

void LLMServerLoadGen::triggerTermination()
{
    // Unblock leaders by sending Termination signal
    for (auto& c : mReqComms)
    {
        sendTerminateMPI(0, *c);
    }
    // unblock my blocked RECV by sending termination signal to responders
    for (auto& c : mRespComms)
    {
        sendTerminateMPI(0, *c);
    }
    for (auto& c : mFirstTokComms)
    {
        sendTerminateMPI(0, *c);
    }
}

// Stop SUT
void LLMServerLoadGen::stopSUT()
{
    std::lock_guard<std::mutex> lock(mExitMtx);
    if ((++mExitCounter) >= mRequiredExitCount)
    {
        LOG(INFO) << "Rank " << mLoadgenRank << ":: " << mName << " Stopping...";
        mServerStopWork = true;
        for (auto& i : *mRespQueueCondVars)
        {
            i.notify_all();
        }
        for (auto& i : *mFirstTokQueueCondVars)
        {
            i.notify_all();
        }
        for (auto& i : *mRequestQueueCondVars)
        {
            i.notify_all();
        }
    }
}

// Wait until Server stop signal is set
void LLMServerLoadGen::waitSutCompletion()
{
    while (!mServerStopWork)
    {
        sleep(1);
    }
}

// sendQueryMPI thread; use one thread per MPI leader for now
void LLMServerLoadGen::sendQueryThread(int32_t const threadIdx)
{
    std::vector<LLMSample> samples;
    while (true)
    {
        {
            std::unique_lock<std::mutex> lck((*mRequestQueueMtxs)[threadIdx]);
            // Wait until the response queue is not empty, or the stop signal
            (*mRequestQueueCondVars)[threadIdx].wait(
                lck, [&]() { return !(*mRequestQueue)[threadIdx].empty() || mServerStopWork; });
            if (mServerStopWork)
            {
                VLOG(1) << "Rank " << mLoadgenRank << ":: " << mName << " exits from sendQueryThread thread";
                break;
            }
            samples = std::vector<LLMSample>(std::make_move_iterator((*mRequestQueue)[threadIdx].begin()),
                std::make_move_iterator((*mRequestQueue)[threadIdx].end()));
            (*mRequestQueue)[threadIdx].clear();
        }
        for (auto& s : samples)
        {
            sendQueryMPI(s, threadIdx);
        }
        samples.clear();
    }
}

// Thread performing MPI Recv responses; use one thread per MPI leader for now
void LLMServerLoadGen::recvResponseMPIThread(int32_t const threadIdx)
{
    MPI_Message msg;
    MPI_Status status;
    int32_t count;
    uintptr_t uniqueId;
    size_t bufferIdx;
    void* bufferPtr;
    int32_t idTag = kMPI_ID_TAG;
    int32_t dataTag = kMPI_DATA_TAG;
    while (true)
    {
        // Blocking receive response message.
        // Within the Comm, worker rank is always 0 and LoadGen rank is always 1
        MPICHECK(MPI_Mprobe(0, idTag, *(mRespComms[threadIdx]), &msg, &status));
        MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
        CHECK_EQ(count, 1) << "Unexpected UniqueID count";
        MPICHECK(MPI_Mrecv(&uniqueId, count, MPI_UINT64_T, &msg, &status));
        // EXIT condition, from receiving TERMINATE msg
        if (uniqueId == kTERMINATION_KEY)
        {
            LOG(INFO) << "Rank " << mLoadgenRank << ":: " << mName << " Port[" << threadIdx
                      << "] unblocked from RECV for shut-down";
#if ENABLE_RECORDER==1
        mRecorder->finalize();
#endif
            stopSUT();
            break;
        }
#if ENABLE_RECORDER==1
        // We assume the finish inference happened at the same time when MPI message is sent
        mRecorder->recordFinishInference(uniqueId);
#endif

        // prep recv'ing data
        MPICHECK(MPI_Mprobe(0, dataTag, *(mRespComms[threadIdx]), &msg, &status));
        MPICHECK(MPI_Get_count(&status, MPI_INT32_T, &count));
        bufferIdx = mRecvBuffer.getAvailBufferIdx();
        bufferPtr = mRecvBuffer.commitBuffer(bufferIdx, count, uniqueId);
        MPICHECK(MPI_Mrecv(bufferPtr, count, MPI_INT32_T, &msg, &status));
        VLOG(1) << "Rank " << mLoadgenRank << ":: Port[" << threadIdx << "]"
                << " Received response msg starting with " << mRecvBuffer.getBufferContent(bufferIdx, 0) 
                << " uniqueId: " << uniqueId << " from local_comm_rank: " << status.MPI_SOURCE 
                << "; msg -- COUNT: " << count << ", TAG: " << idTag << "+" << dataTag 
                << ", ERR: " << status.MPI_ERROR;

        // make QSR and queue it
        mlperf::QuerySampleResponse resp;
        resp.id = uniqueId;
        resp.size = static_cast<size_t>(count) * sizeof(int32_t);
        resp.data = reinterpret_cast<uintptr_t>(bufferPtr);
        resp.n_tokens = count;
        {
            std::scoped_lock<std::mutex> lck((*mRespQueueMtxs)[threadIdx]);
            (*mRespQueue)[threadIdx].emplace_back(std::move(resp));
            (*mRespQueueCondVars)[threadIdx].notify_one();
        }

    }
}

// This completes the responses gathered through MPI, to LoadGen
void LLMServerLoadGen::responseCompleteThread(int32_t const threadIdx)
{
    uint32_t nbSamplesProcessed = 0;
    std::vector<mlperf::QuerySampleResponse> responses;
    while (true)
    {
        {
            std::unique_lock<std::mutex> lck((*mRespQueueMtxs)[threadIdx]);
            // Wait until the response queue is not empty, or the stop signal
            (*mRespQueueCondVars)[threadIdx].wait(
                lck, [&]() { return !(*mRespQueue)[threadIdx].empty() || mServerStopWork; });
            if (mServerStopWork)
            {
                VLOG(1) << "Rank " << mLoadgenRank << ":: " << mName << " exits from responseCompleteThread thread";
                break;
            }

            responses
                = std::vector<mlperf::QuerySampleResponse>(std::make_move_iterator((*mRespQueue)[threadIdx].begin()),
                    std::make_move_iterator((*mRespQueue)[threadIdx].end()));
            (*mRespQueue)[threadIdx].clear();
        }

        mlperf::QuerySamplesComplete(responses.data(), responses.size());
        for (auto& r : responses)
        {
            mRecvBuffer.cleanBuffer(r.id);
            ++nbSamplesProcessed;
            if (nbSamplesProcessed % 1000 == 0)
            {
                LOG(INFO) << "Rank " << mLoadgenRank << ":: Port[" << threadIdx << "] completed " << nbSamplesProcessed
                          << " samples to LoadGen";
            }
#if ENABLE_RECORDER==1
            mRecorder->recordQueryComplete(r.id, r.n_tokens);
#endif
        }
        VLOG(1) << "Rank " << mLoadgenRank << " Port [" << threadIdx << "]"
                << " completed " << responses.size() << " sample(s). Total finished now: " << nbSamplesProcessed;

        mNumInFlightTransactions[threadIdx] -= responses.size();
        responses.clear();
    }
    VLOG(1) << "Rank " << mLoadgenRank << ":: QuerySamplesCompelete Port[" << threadIdx << "] completed total "
            << nbSamplesProcessed << " samples";
}

// Thread performing MPI Recv first tokens;
void LLMServerLoadGen::recvFirstTokMPIThread(int32_t const threadIdx)
{
    MPI_Message msg;
    MPI_Status status;
    int32_t count;
    uintptr_t uniqueId;
    size_t bufferIdx;
    void* bufferPtr;
    int32_t idTag = kMPI_ID_TAG;
    int32_t dataTag = kMPI_DATA_TAG;
    while (true)
    {
        // Blocking receive response message.
        // Within the Comm, worker rank is always 0 and LoadGen rank is always 1
        MPICHECK(MPI_Mprobe(0, idTag, *(mFirstTokComms[threadIdx]), &msg, &status));
        MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
        CHECK_EQ(count, 1) << "Unexpected UniqueID count";
        MPICHECK(MPI_Mrecv(&uniqueId, count, MPI_UINT64_T, &msg, &status));
        // EXIT condition, from receiving TERMINATE msg
        if (uniqueId == kTERMINATION_KEY)
        {
            LOG(INFO) << "Rank " << mLoadgenRank << ":: " << mName << " Port[" << threadIdx
                      << "] unblocked from first-token receive for shut-down";
            stopSUT();
            break;
        }
#if ENABLE_RECORDER==1
        mRecorder->recordFirstToken(uniqueId);
#endif
        // prep recv'ing data
        MPICHECK(MPI_Mprobe(0, dataTag, *(mFirstTokComms[threadIdx]), &msg, &status));
        MPICHECK(MPI_Get_count(&status, MPI_INT32_T, &count));
        CHECK_EQ(count, 1) << "The first token can only be of length 1.";
        bufferIdx = mFirstTokBuffer.getAvailBufferIdx();
        bufferPtr = mFirstTokBuffer.commitBuffer(bufferIdx, count, uniqueId);
        MPICHECK(MPI_Mrecv(bufferPtr, count, MPI_INT32_T, &msg, &status));
        VLOG(1) << "Rank " << mLoadgenRank << ":: Port[" << threadIdx << "]"
                << " Received first token starting with " << mFirstTokBuffer.getBufferContent(bufferIdx, 0)
                << " uniqueId: " << uniqueId << " from local_comm_rank: " << status.MPI_SOURCE
                << "; msg -- COUNT: " << count << ", TAG: " << idTag << "+" << dataTag << ", ERR: " << status.MPI_ERROR;

        // make QSR and queue it
        mlperf::QuerySampleResponse resp;
        resp.id = uniqueId;
        resp.size = static_cast<size_t>(count) * sizeof(int32_t);
        resp.data = reinterpret_cast<uintptr_t>(bufferPtr);
        resp.n_tokens = count;
        {
            std::scoped_lock<std::mutex> lck((*mFirstTokQueueMtxs)[threadIdx]);
            (*mFirstTokQueue)[threadIdx].emplace_back(std::move(resp));
            (*mFirstTokQueueCondVars)[threadIdx].notify_one();
        }
    }
}

// This completes the first tokens gathered through MPI, to LoadGen
void LLMServerLoadGen::firstTokCompleteThread(int32_t const threadIdx)
{
    uint32_t nbSamplesProcessed = 0;
    std::vector<mlperf::QuerySampleResponse> responses;
    while (true)
    {
        {
            std::unique_lock<std::mutex> lck((*mFirstTokQueueMtxs)[threadIdx]);
            // Wait until the response queue is not empty, or the stop signal
            (*mFirstTokQueueCondVars)[threadIdx].wait(
                lck, [&]() { return !(*mFirstTokQueue)[threadIdx].empty() || mServerStopWork; });
            if (mServerStopWork)
            {
                VLOG(1) << "Rank " << mLoadgenRank << ":: " << mName << " exits from responseCompleteThread thread";
                break;
            }

            responses = std::vector<mlperf::QuerySampleResponse>(
                std::make_move_iterator((*mFirstTokQueue)[threadIdx].begin()),
                std::make_move_iterator((*mFirstTokQueue)[threadIdx].end()));
            (*mFirstTokQueue)[threadIdx].clear();
        }

        for (auto& r : responses)
        {
            mlperf::FirstTokenComplete(&r, 1);
            VLOG(1) << "Rank " << mLoadgenRank << ":: Port[" << threadIdx << "] FirstTokenComplete for " << r.id;
            mFirstTokBuffer.cleanBuffer(r.id);
            ++nbSamplesProcessed;
            if (nbSamplesProcessed % 1000 == 0)
            {
                LOG(INFO) << "Rank " << mLoadgenRank << ":: Port[" << threadIdx << "] completed " << nbSamplesProcessed
                          << " first tokens to LoadGen";
            }
        }
        VLOG(1) << "Rank " << mLoadgenRank << " Port [" << threadIdx << "]"
                << " completed " << responses.size() << " first token(s). Total finished now: " << nbSamplesProcessed;

        responses.clear();
    }
    VLOG(1) << "Rank " << mLoadgenRank << ":: FirstTokCompelete Port[" << threadIdx << "] completed total "
            << nbSamplesProcessed << " first tokens";
}

// This is LoadGen side of MPI processes
// Set serverNumWorkerThreads == number of inference groups (number of leaders == number of responders)
LLMServerLoadGen::LLMServerLoadGen(std::string const& name, std::shared_ptr<qsl::SampleLibrary> qsl,
    LLMConfig const& LLMConfig, int32_t serverNumWorkerThreads, bool useMultiIssueThreads, bool excludeInputInOutput,
    bool enableSort, std::vector<std::shared_ptr<MPI_Comm>> reqComms, std::vector<std::shared_ptr<MPI_Comm>> respComms,
    std::vector<std::shared_ptr<MPI_Comm>> firstTokComms, int32_t loadgenRank, int32_t distrPol, int32_t numTrxInflight, 
    bool verboseNVTX)
    : mName(name)
    , mQsl(qsl)
    , mLLMConfig(LLMConfig)
    , mNumWorkerThreads(serverNumWorkerThreads)
    , mUseMultiIssueThreads(useMultiIssueThreads)
    , mExcludeInputInOutput(excludeInputInOutput)
    , mReqComms(reqComms)
    , mRespComms(respComms)
    , mFirstTokComms(firstTokComms)
    , mLoadgenRank(loadgenRank)
    , mEnableSort(enableSort)
    , mServerStopWork(false)
    , mRecvBuffer(numTrxInflight,
          excludeInputInOutput ? kMAX_BUFFER_BYTESIZE_PER_TRANSACTION : kMAX_BUFFER_BYTESIZE_PER_TRANSACTION_WITH_INPUT,
          name + "-SampleRespRecvBuf")
    , mFirstTokBuffer(numTrxInflight, 1, name + "-FirstTokenRecvBuf")
    , mLastUsedQueueIdx(0)
    , mTrafficDistributionPolicy(distrPol)
    , mExitCounter(0)
    , mRequiredExitCount(respComms.size() + firstTokComms.size())
{
    CHECK_GT(serverNumWorkerThreads, 0) << ":: serverNumWorkerThreads should be greater than 0";

    VLOG(1) << "LLMServerLoadGen[" << mName << "] being constructed...";

    mRequestQueue = std::make_unique<RequestQueueVec>(serverNumWorkerThreads);
    mRequestQueueMtxs = std::make_unique<MutexVec>(serverNumWorkerThreads);
    mRequestQueueCondVars = std::make_unique<CondVarVec>(serverNumWorkerThreads);
    mRespQueue = std::make_unique<QSRQueueVec>(serverNumWorkerThreads);
    mRespQueueMtxs = std::make_unique<MutexVec>(serverNumWorkerThreads);
    mRespQueueCondVars = std::make_unique<CondVarVec>(serverNumWorkerThreads);
    mFirstTokQueue = std::make_unique<QSRQueueVec>(serverNumWorkerThreads);
    mFirstTokQueueMtxs = std::make_unique<MutexVec>(serverNumWorkerThreads);
    mFirstTokQueueCondVars = std::make_unique<CondVarVec>(serverNumWorkerThreads);

    // start threads
    mIssueQueryThreads.clear();
    mSendQueryThreads.clear();
    mRecvResponseThreads.clear();
    mResponseCompleteThreads.clear();
    mRecvFirstTokThreads.clear();
    mFirstTokCompleteThreads.clear();
    for (int i = 0; i < serverNumWorkerThreads; ++i)
    {
        if (mUseMultiIssueThreads)
        {
            // NOTE: This should only apply to Server scenario
            mIssueQueryThreads.emplace_back(&LLMServerLoadGen::startIssueThread, this, i);
        }
        mSendQueryThreads.emplace_back(&LLMServerLoadGen::sendQueryThread, this, i);
        mRecvResponseThreads.emplace_back(&LLMServerLoadGen::recvResponseMPIThread, this, i);
        mResponseCompleteThreads.emplace_back(&LLMServerLoadGen::responseCompleteThread, this, i);

        if (mLLMConfig.mReportFirstToken)
        {
            mRecvFirstTokThreads.emplace_back(&LLMServerLoadGen::recvFirstTokMPIThread, this, i);
            mFirstTokCompleteThreads.emplace_back(&LLMServerLoadGen::firstTokCompleteThread, this, i);
        }

        // TODO: add the functionality of counting inflight trx; this may become useful
        mNumInFlightTransactions.emplace_back(0);
    }

#if ENABLE_RECORDER == 1
    // Initiate a recorder class to track the query lifetime
    mRecorder = std::make_unique<Recorder>(mLLMConfig.mReportFirstToken, mName);
#endif

    VLOG(1) << "LLMServerLoadGen[" << mName << "] constructed.";
}

LLMServerLoadGen::~LLMServerLoadGen()
{
#if ENABLE_RECORDER==1
        mRecorder->calculateMetrics();
        mRecorder->report();
#endif
    VLOG(1) << "Destructing LLMServerLoadGen...";
    if (!mIssueQueryThreads.empty())
    {
        for (auto& thread : mIssueQueryThreads)
        {
            thread.join();
        }
    }
    if (!mSendQueryThreads.empty())
    {
        for (auto& thread : mSendQueryThreads)
        {
            thread.join();
        }
    }
    if (!mRecvResponseThreads.empty())
    {
        for (auto& thread : mRecvResponseThreads)
        {
            thread.join();
        }
    }
    if (!mResponseCompleteThreads.empty())
    {
        for (auto& thread : mResponseCompleteThreads)
        {
            thread.join();
        }
    }
    if (!mRecvFirstTokThreads.empty())
    {
        for (auto& thread : mRecvFirstTokThreads)
        {
            thread.join();
        }
    }
    if (!mFirstTokCompleteThreads.empty())
    {
        for (auto& thread : mFirstTokCompleteThreads)
        {
            thread.join();
        }
    }
    for (auto& q : mNumInFlightTransactions)
    {
        CHECK_EQ(q, 0) << "Tracking number of in flight transactions seems seems corrupted";
    }
    VLOG(1) << "Done destructing LLMServerLoadGen[" << mName << "].";
}

std::string const& LLMServerLoadGen::Name()
{
    return mName;
}
