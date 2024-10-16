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

#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"

#include "llm_utils.hpp"
#include "qsl.hpp"
#include "system_under_test.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::mpi;

using LLMBatch = std::vector<LLMSample>;
using MutexVec = std::vector<std::mutex>;
using CondVarVec = std::vector<std::condition_variable>;
using RequestQueueVec = std::vector<std::deque<LLMSample>>;
using RespQueueVec = std::vector<std::deque<LLMResponse>>;
using QSRQueueVec = std::vector<std::deque<mlperf::QuerySampleResponse>>;
using ReclaimQueueVec = std::vector<std::deque<uintptr_t>>;
using DevIdVec = std::vector<int32_t>;

//! This class manages the trtllm batch manager
//! Each LLMCore runs on a single cuda stream
class LLMCore
{
public:
    LLMCore(LLMConfig const& LLMConfig, std::filesystem::path const& enginePath, int32_t llmCoreIdx, int32_t deviceId,
        DevIdVec grpDevIds, int32_t threadIdx, int32_t myRank, bool isReceiver, bool isResponder,
        std::shared_ptr<RequestQueueVec> requestQueue, std::shared_ptr<MutexVec> requestQueueMtxs,
        std::shared_ptr<CondVarVec> requestQueueCondVars, std::shared_ptr<RespQueueVec> respQueue,
        std::shared_ptr<MutexVec> respQueueMtxs, std::shared_ptr<CondVarVec> respQueueCondVars,
        std::shared_ptr<RespQueueVec> firstTokQueue, std::shared_ptr<MutexVec> firstTokQueueMtxs,
        std::shared_ptr<CondVarVec> firstTokQueueCondVars, std::shared_ptr<ReclaimQueueVec> reclaimQueue,
        std::shared_ptr<MutexVec> reclaimQueueMtxs, std::shared_ptr<CondVarVec> reclaimQueueCondVars);
    ~LLMCore() = default;
    void warmUp();

    int32_t const mDeviceId;
    int32_t const mLLMCoreIdx;

    DevIdVec const mGroupDeviceIds;

    int32_t const mMyRank;
    int32_t const mMySessionRank;
    int32_t const mMySessionSize;
    bool const mIsReceiver;
    bool const mIsResponder;

    // Wait until the batch manager is done
    void waitUntilTerminate() const;

    // Notify that the generation session is complete
    void notifyCoreComplete();

private:
    // Below are callback functions to provide to the gptManager
    // Post process and send the response of gptManager to the LLMServer
    void sendResponse(
        uint64_t requestId, std::list<NamedTensor> responseTensors, bool finalResponse, const std::string& errMsg);

    // Acquire queries from the LLMServer RequestQueue
    std::list<std::shared_ptr<InferenceRequest>> getInferenceRequest(const int32_t maxNbRequests);
    void populateInferenceRequest(LLMBatch batch, std::list<std::shared_ptr<InferenceRequest>>& rval);

    LLMConfig mLLMConfig;
    std::filesystem::path const& mEnginePath;
    int32_t mThreadQueueIdx; // The index of the issueQueryThread, and the RequestQueue/ResponseQueue

    std::unique_ptr<GptManager> mBatchManager;
    // store sample's input sequence length and pointer to input ID, by its unique ID (response ID)
    std::unordered_map<uintptr_t, std::tuple<int32_t, uintptr_t>> mIdToInpInfo;
    // Buffer streaming token info
    std::unordered_map<uint64_t, LLMResponse> mStreamResponseBufferMap;

    // threading attributes
    // LLMCore maintains the pointers to the mutex, conditional variable, and the data structure of:
    // 1) requestQueue
    // 2) respQueue
    // 3) firstTokQueue (only for workloads which require first-token latency, e.g. Llama2)
    // from LLMServer. LLMCore will attempt to read inputs from the requestQueue, and send response to respQueue.
    std::shared_ptr<RequestQueueVec> mServerRequestQueue;
    std::shared_ptr<MutexVec> mServerRequestQueueMtxs;
    std::shared_ptr<CondVarVec> mServerRequestQueueCondVars;

    std::shared_ptr<RespQueueVec> mServerRespQueue;
    std::shared_ptr<MutexVec> mServerRespQueueMtxs;
    std::shared_ptr<CondVarVec> mServerRespQueueCondVars;

    std::shared_ptr<RespQueueVec> mServerFirstTokQueue;
    std::shared_ptr<MutexVec> mServerFirstTokQueueMtxs;
    std::shared_ptr<CondVarVec> mServerFirstTokQueueCondVars;

    std::shared_ptr<ReclaimQueueVec> mServerReclaimQueue;
    std::shared_ptr<MutexVec> mServerReclaimQueueMtxs;
    std::shared_ptr<CondVarVec> mServerReclaimQueueCondVars;

    std::atomic<bool> mCoreStopWork;
};
