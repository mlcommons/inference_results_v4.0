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

#include "llm_core.hpp"
#include "llm_recorder.hpp"
#include "lwis_buffers.h"
#include "qsl.hpp"
#include "system_under_test.h"
#include "utils.hpp"

#include <filesystem>

// Size to be used for buffering samples through MPI
constexpr int32_t kMAX_TRANSACTIONS_IN_FLIGHT{8192};
constexpr int32_t kMAX_BUFFER_BYTESIZE_PER_TRANSACTION{2048};
constexpr int32_t kMAX_BUFFER_BYTESIZE_PER_TRANSACTION_WITH_INPUT{4096};
constexpr int32_t kMPI_ID_TAG{127};
constexpr int32_t kMPI_DATA_TAG{1023};
constexpr uint64_t kTERMINATION_KEY{UINT64_MAX};

enum trafficDistributionPolicy
{
    roundRobin = 0,
    staticChop = 1,
    loadBalancing = 2,
    unknown,
};

//! This class overrides mlperf SUT class. LLMServerLight initializes TRT LLM engines for worker ranks,
//! in harmony with leader rank fully fledged LLMServer.
class LLMServer : public mlperf::SystemUnderTest
{
public:
    LLMServer(std::string const& name, std::string const& enginePath, DevIdVec const& gpuDeviceIds,
        DevIdVec const& grpDevIds, int32_t numLLMCoresPerDevice, LLMConfig const& LLMConfig,
        int32_t serverNumWorkerThreads, std::vector<std::shared_ptr<MPI_Comm>> respComms,
        std::vector<std::shared_ptr<MPI_Comm>> firstTokComms, std::vector<std::shared_ptr<MPI_Comm>> reclaimComms,
        bool isReceiver, bool isResponder, bool useMPI, int32_t myRank, bool verboseNVTX);

    virtual ~LLMServer();

    std::string const& Name() override;

    virtual void IssueQuery(std::vector<mlperf::QuerySample> const& samples){};
    virtual void FlushQueries(){};

    // Wait until Server stop signal is set
    void waitSutCompletion();

protected:
    virtual void ProcessTasks(std::shared_ptr<LLMCore> llmCore, int32_t deviceId, int32_t qThreadIdx){};

    // Waiting for all LLMCore to complete.
    void waitForLLMCoreComplete();

    // Send the response staged in the respQueue to mlperf::QuerySampleComplete
    void sendQueryComplete(int32_t const threadIdx);

    // Send the response through MPI (when multi-process is enabled)
    void sendQueryCompleteMPI(LLMResponse resp, int32_t const threadIdx);

    // Send the response staged in the respQueue to mlperf::QuerySampleComplete
    void sendFirstTokComplete(int32_t const threadIdx);

    // Send the response through MPI (when multi-process is enabled)
    void sendFirstTokCompleteMPI(LLMResponse resp, int32_t const threadIdx);

    // Send Terminate msg to receiving end through MPI
    void sendTerminateMPI(int32_t const targetRank, MPI_Comm const& comm);

    // Handle reclaim buffer for MPI communication
    void handleReclaimBuffer(int32_t const threadIdx);

    // Wait until termination signal comes
    void waitForTermination(int32_t const threadIdx);

    // Stop SUT
    void stopSUT();

    std::string const mName;
    // If isReceiver == true, this would be the server instance receiving requests from LoadGen
    // This only happens when this was instantiated from LLMServerLeader
    bool const mIsReceiver;
    // If isResponder == true, this would be the server instance sending responses to LoadGen
    bool const mIsResponder;

    // MPI related
    // if useMPI == true, tensor communication happens through MPI
    bool const mUseMPI;
    std::vector<std::shared_ptr<MPI_Comm>> const mRespComms;
    std::vector<std::shared_ptr<MPI_Comm>> const mFirstTokComms;
    std::vector<std::shared_ptr<MPI_Comm>> const mReclaimComms;
    int32_t const mMyRank;

    // SUT attributes
    int32_t mTotalSampleConsumed;
    int32_t mNumLLMCoresPerDevice;

    LLMConfig mLLMConfig;

    std::vector<std::vector<std::shared_ptr<LLMCore>>> mLLMCoreVec;

    // RequestQueue is a deque to hold queries. It will be accessed by the
    // LLMCore to acquire tasks.
    std::shared_ptr<RequestQueueVec> mRequestQueue;
    std::shared_ptr<MutexVec> mRequestQueueMtxs;
    std::shared_ptr<CondVarVec> mRequestQueueCondVars;

    // RespQueue is a deque to temporarily hold queries sent from LLMCore.
    std::shared_ptr<RespQueueVec> mRespQueue;
    std::shared_ptr<MutexVec> mRespQueueMtxs;
    std::shared_ptr<CondVarVec> mRespQueueCondVars;

    // Queue to hold first token responses.
    std::shared_ptr<RespQueueVec> mFirstTokQueue;
    std::shared_ptr<MutexVec> mFirstTokQueueMtxs;
    std::shared_ptr<CondVarVec> mFirstTokQueueCondVars;

    // ReclaimQueue is a deque to hold resources to reclaim when the data under uniqueID is no more required
    std::shared_ptr<ReclaimQueueVec> mReclaimQueue;
    std::shared_ptr<MutexVec> mReclaimQueueMtxs;
    std::shared_ptr<CondVarVec> mReclaimQueueCondVars;

    // one thread to handle each response queue
    std::vector<std::thread> mRespQueueThreads;
    // one thread to handle each first-token response queue
    std::vector<std::thread> mFirstTokQueueThreads;
    // thread to reclaim recv buffer resources
    std::vector<std::thread> mReclaimBufferThreads;
    // thread to wait for termination signal
    std::vector<std::thread> mWaitForTerminationThreads;

    // Indicates that the server should stop working
    std::atomic<bool> mServerStopWork;

    // Benchmarking utils
    // std::shared_ptr<Recorder> mRecorder;
};

//! This class inherits LLMServer and adds issue functionality. This also manages data transfer from the QSL
//! and sends query samples to LLMCore instances.
class LLMServerLeader : public LLMServer
{
public:
    LLMServerLeader(std::string const& name, std::shared_ptr<qsl::SampleLibrary> qsl, std::string const& enginePath,
        DevIdVec const& gpuDeviceIds, DevIdVec const& grpDevIds, int32_t numLLMCoresPerDevice, int32_t gpuBatchSize,
        LLMConfig const& LLMConfig, double targetLatencyPercentile, int32_t serverNumWorkerThreads,
        std::vector<std::shared_ptr<MPI_Comm>> reqComms, std::vector<std::shared_ptr<MPI_Comm>> respComms,
        std::vector<std::shared_ptr<MPI_Comm>> firstTokComms, std::vector<std::shared_ptr<MPI_Comm>> reclaimComms,
        bool isResponder, bool useMPI, int32_t myRank, int32_t responderLocalRank, int32_t numTrxInflight, bool verboseNVTX);

    virtual ~LLMServerLeader();

    void IssueQuery(std::vector<mlperf::QuerySample> const& samples) override;
    void FlushQueries() override;

private:
    std::vector<std::pair<int32_t, int32_t>> sortSamples(std::vector<mlperf::QuerySample> const& samples) const;
    void startIssueThread(int32_t const threadIdx);
    void startRecvMPIThread(int32_t const threadIdx);
    void reclaimResources(int32_t const threadIdx);
    void reclaimResourcesMPI(int32_t const threadIdx);

    std::shared_ptr<qsl::SampleLibrary> mQsl;

    // Attributes for the SUT
    double mTargetLatencyPercentile;
    int32_t mGpuBatchSize;

    commBuffer<int32_t, uintptr_t> mRecvBuffer;
    std::vector<std::shared_ptr<MPI_Comm>> const mReqComms;
    int32_t const mResponderSessionRank;

    std::map<std::thread::id, int32_t> mIssueQueryThreadMap;
    std::vector<std::thread> mIssueQueryThreads; // one issue thread for each taskqueue
    std::vector<std::thread> mRecvQueryThreads;
    std::vector<std::thread> mReclaimResourceThreads;
};

//! This class inherits MLPerf SUT and adds issue functionality. This also manages data transfer from the QSL
//! and sends query samples through MPI
class LLMServerLoadGen : public mlperf::SystemUnderTest
{
public:
    LLMServerLoadGen(std::string const& name, std::shared_ptr<qsl::SampleLibrary> qsl, LLMConfig const& LLMConfig,
        int32_t serverNumWorkerThreads, bool useMultiIssueThreads, bool excludeInputInOutput, bool enableSort,
        std::vector<std::shared_ptr<MPI_Comm>> reqComms, std::vector<std::shared_ptr<MPI_Comm>> respComms,
        std::vector<std::shared_ptr<MPI_Comm>> firstTokComms, int32_t loadgenRank, int32_t distrPol, int32_t numTrxInflight, 
        bool verboseNVTX);

    virtual ~LLMServerLoadGen();
    void IssueQuery(std::vector<mlperf::QuerySample> const& samples) override;
    void FlushQueries() override;
    std::string const& Name() override;

    // Send msgs to prepare shut-down
    void triggerTermination();

    // Wait until Server stop signal is set
    void waitSutCompletion();

private:
    // Send Terminate msg to receiving end through MPI
    void sendTerminateMPI(int32_t const targetRank, MPI_Comm const& comm);

    // Stop SUT
    void stopSUT();

    std::vector<std::pair<int32_t, int32_t>> sortSamples(std::vector<mlperf::QuerySample> const& samples) const;
    void startIssueThread(int32_t const threadIdx);
    void sendQueryThread(int32_t const threadIdx);
    void sendQueryMPI(LLMSample sample, int32_t const threadIdx);
    void recvResponseMPIThread(int32_t const threadIdx);
    void responseCompleteThread(int32_t const threadIdx);
    void recvFirstTokMPIThread(int32_t const threadIdx);
    void firstTokCompleteThread(int32_t const threadIdx);

    // Traffic distribution related
    int32_t getLeastBusyIdx();
    int32_t getNextQueueIdx(size_t sample_size, int32_t idx);

    std::string const mName;
    std::shared_ptr<qsl::SampleLibrary> mQsl;
    LLMConfig mLLMConfig;
    bool mExcludeInputInOutput;
    bool mEnableSort;

    int32_t const mNumWorkerThreads;
    int32_t mLastUsedQueueIdx;
    bool const mUseMultiIssueThreads;
    int32_t const mTrafficDistributionPolicy;

    commBuffer<int32_t, uintptr_t> mRecvBuffer;
    commBuffer<int32_t, uintptr_t> mFirstTokBuffer;

    std::vector<std::shared_ptr<MPI_Comm>> mReqComms;
    std::vector<std::shared_ptr<MPI_Comm>> mRespComms;
    std::vector<std::shared_ptr<MPI_Comm>> mFirstTokComms;
    int32_t const mLoadgenRank;

    std::atomic<int32_t> mExitCounter;
    int32_t const mRequiredExitCount;
    std::mutex mExitMtx;

    std::map<std::thread::id, int32_t> mIssueQueryThreadMap;
    std::vector<std::thread> mIssueQueryThreads; // one issue thread for each taskqueue
    std::vector<std::thread> mSendQueryThreads;
    std::vector<std::thread> mRecvResponseThreads;
    std::vector<std::thread> mResponseCompleteThreads;
    std::vector<std::thread> mRecvFirstTokThreads;
    std::vector<std::thread> mFirstTokCompleteThreads;

    // RequestQueue is a deque to hold queries. It will be accessed by the
    // LLMCore to acquire tasks.
    std::unique_ptr<RequestQueueVec> mRequestQueue;
    std::unique_ptr<MutexVec> mRequestQueueMtxs;
    std::unique_ptr<CondVarVec> mRequestQueueCondVars;

    // RespQueue is a deque to temporarily hold queries sent from LLMCore.
    std::unique_ptr<QSRQueueVec> mRespQueue;
    std::unique_ptr<MutexVec> mRespQueueMtxs;
    std::unique_ptr<CondVarVec> mRespQueueCondVars;

    // RespQueue is a deque to temporarily hold queries sent from LLMCore.
    std::unique_ptr<QSRQueueVec> mFirstTokQueue;
    std::unique_ptr<MutexVec> mFirstTokQueueMtxs;
    std::unique_ptr<CondVarVec> mFirstTokQueueCondVars;

    // Indicates that the server should stop working
    std::atomic<bool> mServerStopWork;

    // track number of transactions per MPI group
    std::deque<std::atomic<uint32_t>> mNumInFlightTransactions;
};