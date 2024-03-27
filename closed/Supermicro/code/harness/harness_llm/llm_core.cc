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

#include "llm_core.hpp"
#include "glog/logging.h"
#include "llm_server.hpp"
#include "llm_utils.hpp"
#include "loadgen.h"

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <fstream>
#include <list>
#include <numeric>

void logStats(std::string const& s, int32_t const& rank, int32_t const& devid)
{
    VLOG(1) << "Rank " << rank << ":: Device[" << devid << "] TRTLLM Log: " << s.c_str();
}

LLMCore::LLMCore(LLMConfig const& LLMConfig, std::filesystem::path const& enginePath, int32_t llmCoreIdx,
    int32_t deviceId, DevIdVec grpDevIds, int32_t threadIdx, int32_t myRank, bool isReceiver, bool isResponder,
    std::shared_ptr<RequestQueueVec> requestQueue, std::shared_ptr<MutexVec> requestQueueMtxs,
    std::shared_ptr<CondVarVec> requestQueueCondVars, std::shared_ptr<RespQueueVec> respQueue,
    std::shared_ptr<MutexVec> respQueueMtxs, std::shared_ptr<CondVarVec> respQueueCondVars,
    std::shared_ptr<RespQueueVec> firstTokQueue, std::shared_ptr<MutexVec> firstTokQueueMtxs,
    std::shared_ptr<CondVarVec> firstTokQueueCondVars, std::shared_ptr<ReclaimQueueVec> reclaimQueue,
    std::shared_ptr<MutexVec> reclaimQueueMtxs, std::shared_ptr<CondVarVec> reclaimQueueCondVars)
    : mLLMConfig(LLMConfig)
    , mEnginePath(enginePath)
    , mDeviceId(deviceId)
    , mGroupDeviceIds(grpDevIds)
    , mLLMCoreIdx(llmCoreIdx)
    , mThreadQueueIdx(threadIdx)
    , mMyRank(myRank)
    , mIsReceiver(isReceiver)
    , mIsResponder(isResponder)
    , mServerRequestQueue(requestQueue)
    , mServerRequestQueueMtxs(requestQueueMtxs)
    , mServerRequestQueueCondVars(requestQueueCondVars)
    , mServerRespQueue(respQueue)
    , mServerRespQueueMtxs(respQueueMtxs)
    , mServerRespQueueCondVars(respQueueCondVars)
    , mServerFirstTokQueue(firstTokQueue)
    , mServerFirstTokQueueMtxs(firstTokQueueMtxs)
    , mServerFirstTokQueueCondVars(firstTokQueueCondVars)
    , mServerReclaimQueue(reclaimQueue)
    , mServerReclaimQueueMtxs(reclaimQueueMtxs)
    , mServerReclaimQueueCondVars(reclaimQueueCondVars)
    , mCoreStopWork(false)
    , mMySessionRank(COMM_SESSION.getRank())
    , mMySessionSize(COMM_SESSION.getSize())
{
    LOG(INFO) << "Rank " << mMyRank << ":: LLMCore[" << llmCoreIdx << "] at Device Id [" << mDeviceId << "]";
    if (mIsReceiver)
    {
        CHECK_EQ(mMySessionRank, 0) << "Rank " << mMyRank << ":: Receiver is not Rank 0 in the session";
    }

    CHECK_EQ(cudaSetDevice(mDeviceId), cudaSuccess);

    // TODO: Provide a default optional params to the batch manager. Check if we need to parameterize this in the config
    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = mLLMConfig.mEnableTrtOverlap;
    optionalParams.kvCacheConfig.freeGpuMemoryFraction = mLLMConfig.mKvCacheFreeGpuMemFraction;
    optionalParams.deviceIds = mGroupDeviceIds;

    // Setup gptManager
    mBatchManager = std::make_unique<GptManager>(
        enginePath, mLLMConfig.mBatchMode, mLLMConfig.mBeamWidth, mLLMConfig.mSchedulerPolicy,
        [this](int maxNbRequests) { return getInferenceRequest(maxNbRequests); },
        [this](uint64_t requestId, std::list<NamedTensor> responseTensors, bool finalResponse,
            std::string const& errMsg) { return sendResponse(requestId, responseTensors, finalResponse, errMsg); },
        nullptr, [rank=this->mMyRank, devId=this->mDeviceId](std::string const& s) { return logStats(s, rank, devId); }, 
        optionalParams, kTERMINATION_KEY, std::nullopt, mLLMConfig.mExcludeInputInOutput);

    LOG(INFO) << "Rank " << mMyRank << ":: LLMCore Setup complete for device " << mDeviceId;
}

// Convert mlperf::QuerySample to InferenceRequest and populate the list for TRT-LLM
void LLMCore::populateInferenceRequest(
    LLMBatch batch, std::list<std::shared_ptr<InferenceRequest>>& inferenceRequestList)
{
    NVTX_THREAD_START("LLMCore::populateInferenceRequest device-" + mDeviceId, nvtx::COLOR_BLUE_3);
    // Create the shared tensor buffer for the inference requests so that the buffer can be reused
    ITensor::SharedPtr beamWidthBuffer = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    *(bufferCast<SizeType>(*beamWidthBuffer)) = mLLMConfig.mBeamWidth;

    ITensor::SharedPtr minLenBuffer = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    *(bufferCast<SizeType>(*minLenBuffer)) = mLLMConfig.mMinOutputSeqlen;

    ITensor::SharedPtr temperatureBuffer = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kFLOAT);
    *(bufferCast<float>(*temperatureBuffer)) = mLLMConfig.mTemperature;

    ITensor::SharedPtr topKBuffer = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    *(bufferCast<SizeType>(*topKBuffer)) = mLLMConfig.mTopK;

    ITensor::SharedPtr topPBuffer = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kFLOAT);
    *(bufferCast<float>(*topPBuffer)) = mLLMConfig.mTopP;

    ITensor::SharedPtr eosIdBuffer = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    *(bufferCast<SizeType>(*eosIdBuffer)) = mLLMConfig.mEosId;

    for (auto& item : batch)
    {
        auto querySample = item.QS;
        auto request = std::make_shared<InferenceRequest>(static_cast<uint64_t>(querySample.id));
        auto inputLength = item.SampleLen;
        auto const inputPtr = reinterpret_cast<int32_t*>(item.SamplePtr);

        std::vector<int64_t> inputIdsShape{1, static_cast<int64_t>(inputLength)};
        auto inputIdsTensor = NamedTensor(nvinfer1::DataType::kINT32, inputIdsShape, "input_ids", inputPtr);
        int32_t const requestOutputLen = mLLMConfig.mMaxOutputSeqlen;
        auto maxNewTokensTensor = NamedTensor(
            nvinfer1::DataType::kINT32, {1, 1}, inference_request::kMaxNewTokensTensorName, &requestOutputLen);

        // Create tensors using the shared buffer.
        auto beamWidthTensor = NamedTensor(beamWidthBuffer, inference_request::kBeamWidthTensorName);
        auto minLenTensor = NamedTensor(minLenBuffer, inference_request::kMinLengthTensorName);
        auto temperatureTensor = NamedTensor(temperatureBuffer, inference_request::kTemperatureTensorName);
        auto topKTensor = NamedTensor(topKBuffer, inference_request::kRuntimeTopKTensorName);
        auto topPTensor = NamedTensor(topPBuffer, inference_request::kRuntimeTopPTensorName);

        request->emplaceInputTensor(inputIdsTensor.name, std::move(inputIdsTensor.tensor));
        request->emplaceInputTensor(maxNewTokensTensor.name, std::move(maxNewTokensTensor.tensor));
        request->emplaceInputTensor(beamWidthTensor.name, std::move(beamWidthTensor.tensor));
        request->emplaceInputTensor(minLenTensor.name, std::move(minLenTensor.tensor));
        request->emplaceInputTensor(temperatureTensor.name, std::move(temperatureTensor.tensor));
        request->emplaceInputTensor(topKTensor.name, std::move(topKTensor.tensor));
        request->emplaceInputTensor(topPTensor.name, std::move(topPTensor.tensor));
        request->setEndId(eosIdBuffer);
        request->setIsStreaming(mLLMConfig.mIsStreaming);
        inferenceRequestList.emplace_back(std::move(request));
        // Add mapping of unique ID (response ID) to input info
        mIdToInpInfo[querySample.id] = std::make_tuple(inputLength, item.SamplePtr);
        // TODO: Add recorder activity
    }
    NVTX_THREAD_END();
}

// 1. Acquire lock from the LLMServer requestQueue
// 2. Get task up to the max batch size, and form InferenceRequest
std::list<std::shared_ptr<InferenceRequest>> LLMCore::getInferenceRequest(const int32_t maxNbRequests)
{
    // CHECK_EQ(cudaSetDevice(mDeviceId), cudaSuccess);
    std::list<std::shared_ptr<InferenceRequest>> inferenceRequestList;
    inferenceRequestList.clear();
    if (maxNbRequests > 0)
    {
        if (mIsReceiver)
        {
            LLMBatch batch;
            batch.clear();
            batch.reserve(maxNbRequests);
            // Acquire QuerySamples from the requestQueue
            {
                // Acquire the lock of the requestQueue. Wait until the requestQueue is populated or the stop variable
                // is set, or the timeout is hit.
                std::scoped_lock<std::mutex> lck((*mServerRequestQueueMtxs)[mThreadQueueIdx]);

                // Exit condition
                if ((*mServerRequestQueue)[mThreadQueueIdx].empty() && mCoreStopWork)
                {
                    inferenceRequestList.emplace_back(std::make_shared<InferenceRequest>(kTERMINATION_KEY));
                }

                if (!(*mServerRequestQueue)[mThreadQueueIdx].empty())
                {
                    NVTX_THREAD_START("LLMCore::getInferenceRequest device-" + mDeviceId, nvtx::COLOR_BLUE_1);
                    int64_t nbQueries = std::min(static_cast<int64_t>(maxNbRequests),
                        static_cast<int64_t>((*mServerRequestQueue)[mThreadQueueIdx].size()));

                    // Form a local batch of QuerySamples by moving from the requestQueue.
                    int32_t actualBatchSize
                        = std::min(static_cast<int32_t>((*mServerRequestQueue)[mThreadQueueIdx].size()), maxNbRequests);
                    batch.insert(batch.end(), (*mServerRequestQueue)[mThreadQueueIdx].begin(),
                        (*mServerRequestQueue)[mThreadQueueIdx].begin() + actualBatchSize);
                    (*mServerRequestQueue)[mThreadQueueIdx].erase((*mServerRequestQueue)[mThreadQueueIdx].begin(),
                        (*mServerRequestQueue)[mThreadQueueIdx].begin() + actualBatchSize);

                    VLOG(1) << "Rank " << mMyRank << "::  Device[" << mDeviceId << "] forming a batch with "
                            << batch.size() << " sample(s), maxNbRequests: " << maxNbRequests
                            << " RequestQueue size: " << (*mServerRequestQueue)[mThreadQueueIdx].size();

                    NVTX_THREAD_END();
                }
            }
            if (!batch.empty())
            {
                // Create tensors and form InfereceRequest for GPTManager
                populateInferenceRequest(batch, inferenceRequestList);
            }

            // Need comm with followers
            if (mMySessionSize > 1)
            {
                // Be careful not to send unsigned using signed int
                auto num_new_work_items = inferenceRequestList.size();
                COMM_SESSION.bcast(&num_new_work_items, 1, tensorrt_llm::mpi::MpiType::kUINT64, 0);
                if (num_new_work_items > 0)
                {
                    std::vector<int64_t> packed;
                    for (auto const& ir : inferenceRequestList)
                    {
                        auto vpacked = ir->serialize();
                        packed.push_back(static_cast<int64_t>(vpacked.size()));
                        packed.insert(
                            packed.end(), std::move_iterator(vpacked.begin()), std::move_iterator(vpacked.end()));
                    }
                    COMM_SESSION.bcast(packed, 0);
                }
            }
        }
        else
        {
            // Need comm with leader
            // NOTE: This may need to be revised if PP is used
            if (mMySessionSize > 1)
            {
                // subordinate ranks hang until master rank sends work
                uint64_t num_new_work_items;
                COMM_SESSION.bcast(&num_new_work_items, 1, tensorrt_llm::mpi::MpiType::kUINT64, 0);
                if (num_new_work_items > 0)
                {
                    std::vector<int64_t> packed;
                    COMM_SESSION.bcast(packed, 0);
                    int64_t* packed_ptr = packed.data();
                    for (int64_t count = 0; count < num_new_work_items; ++count)
                    {
                        int64_t n = *(packed_ptr++);
                        auto ir = InferenceRequest::deserialize(packed_ptr);
                        packed_ptr += n;
                        inferenceRequestList.emplace_back(ir);
                    }
                }
            }
        }
    }
    return inferenceRequestList;
}

void LLMCore::warmUp()
{
    // TODO: Investigate whether warming up will help TRT-LLM.
    LOG(INFO) << "Rank " << mMyRank << " Device " << mDeviceId << ": Warm up bypassed.";
    return;
}

void LLMCore::waitUntilTerminate() const
{
    mBatchManager->waitUntilTerminate();
}

void LLMCore::notifyCoreComplete()
{
    mCoreStopWork = true;
}

void LLMCore::sendResponse(
    uint64_t requestId, std::list<NamedTensor> responseTensors, bool finalResponse, const std::string& errMsg)
{
    if (!mIsResponder)
    {
        return;
    }

    NVTX_THREAD_START("LLMCore::sendResponse for device " + mDeviceId, nvtx::COLOR_BLUE_2);
    // For non-streaming mode, we only process final response. For streaming mode, we need
    // to buffer each token, report first-token and final token latency.
    if (!finalResponse && !mLLMConfig.mIsStreaming)
    {
        return;
    }

    // If the response tensor list is empty, it indicates that something is broken
    if (responseTensors.empty())
    {
        std::string errStr = std::string("Failed to send response for requestId: ") + std::to_string(requestId)
            + " because of: " + errMsg;
        LOG(ERROR) << errStr;
        CHECK(!responseTensors.empty());
    }

    // The response tensor is already on CPU. Construct the response and send to the respQueue.
    LLMResponse resp;
    int32_t outputSeqLen;
    int32_t* outputIdsBufferPtr;
    for (auto& tensor : responseTensors)
    {
        if (tensor.name == inference_request::kOutputIdsTensorName)
        {
            outputIdsBufferPtr = bufferCast<int32_t>(*(tensor.tensor));
        }
        else if (tensor.name == inference_request::kSequenceLengthTensorName)
        {
            // Tensor of shape {nbBeams}, and we only need the first one
            outputSeqLen = *(bufferCast<int32_t>(*(tensor.tensor)));
        }
        // TODO: We can dump the CumLogProb here for debugging.
    }

    if (mLLMConfig.mIsStreaming)
    {
        // For streaming, only 1 token will be generated in each step.
        // We need to report first token and last token back to the LoadGen.
        // And we buffer tokens in a ID->LLMResponse map instead of handling in the ResponseQueue
        bool isFirstToken = (mStreamResponseBufferMap.find(requestId) == mStreamResponseBufferMap.end());
        VLOG(2) << "Rank " << mMyRank << ":: ID: " << requestId << " Length: " << outputSeqLen << " Streamed token: " << outputIdsBufferPtr[0]
                << " isFinal: " << finalResponse << " isFirst: " << isFirstToken;
        if (isFirstToken)
        {
            mStreamResponseBufferMap.emplace(requestId, LLMResponse());
        }
        else
        {
            // Make sure the requestId is available in the buffer
            CHECK(mStreamResponseBufferMap.find(requestId) != mStreamResponseBufferMap.end())
                << "Rank " << mMyRank << " cannot find " << requestId << " in the mStreamResponseBufferMap.";
        }
        // Push the tokens into the LLMResponse.
        if (outputSeqLen)
        {
            // The seqlen should always be 1, otherwise 0 if it's the EOS token
            CHECK_EQ(outputSeqLen, 1) << "Rank " << mMyRank << " ID: " << requestId << " has invalid response length.";
            ;
            mStreamResponseBufferMap[requestId].outputTokens.push_back(outputIdsBufferPtr[0]);
        }

        // If we hit the final tokens, populate the QSR information
        if (finalResponse)
        {
            int32_t finalSeqLen = mStreamResponseBufferMap[requestId].outputTokens.size();
            // NOTE: The LoadGen doesn't allow 1-token output, so we append EOS to the end if final SeqLen is 1
            if (finalSeqLen == 1)
            {
                finalSeqLen += 1;
                mStreamResponseBufferMap[requestId].outputTokens.push_back(mLLMConfig.mEosId);
            }
            mStreamResponseBufferMap[requestId].QSR.id = requestId;
            mStreamResponseBufferMap[requestId].QSR.data
                = reinterpret_cast<uintptr_t>(mStreamResponseBufferMap[requestId].outputTokens.data());
            mStreamResponseBufferMap[requestId].QSR.size = static_cast<size_t>(finalSeqLen * sizeof(int32_t));
            mStreamResponseBufferMap[requestId].QSR.n_tokens = finalSeqLen;
            // Transfer ownership of the response to the outside variable, and remove from the map
            auto extractedBufferNodeHandle = mStreamResponseBufferMap.extract(requestId);
            CHECK(!extractedBufferNodeHandle.empty());
            resp = std::move(extractedBufferNodeHandle.mapped());
            mIdToInpInfo.erase(requestId);
        }
        else
        {
            // Send first token back to the loadgen
            if (isFirstToken && mLLMConfig.mReportFirstToken)
            {
                CHECK_EQ(mStreamResponseBufferMap[requestId].outputTokens.size(), 1);
                // Make a copy of the response to make sure there's no memory handling issues
                LLMResponse firstTokResp;
                firstTokResp.outputTokens = mStreamResponseBufferMap[requestId].outputTokens;
                firstTokResp.QSR.id = requestId;
                firstTokResp.QSR.data = reinterpret_cast<uintptr_t>(firstTokResp.outputTokens.data());
                firstTokResp.QSR.size = static_cast<size_t>(1 * sizeof(int32_t));
                firstTokResp.QSR.n_tokens = 1;

                std::scoped_lock<std::mutex> lck((*mServerFirstTokQueueMtxs)[mThreadQueueIdx]);
                (*mServerFirstTokQueue)[mThreadQueueIdx].emplace_back(std::move(firstTokResp));
                VLOG(1) << "Rank:" << mMyRank << ", LLMCore sending first-token response for requestId: " << requestId
                        << " RespQueue size: " << (*mServerFirstTokQueue)[mThreadQueueIdx].size();
                (*mServerFirstTokQueueCondVars)[mThreadQueueIdx].notify_one();
            }
            return;
        }
    }
    else
    {
        auto [inputSeqLen, inputIdPtr] = mIdToInpInfo[requestId];
        mIdToInpInfo.erase(requestId);
        auto outputPtr = mLLMConfig.mExcludeInputInOutput ? (outputIdsBufferPtr) : (outputIdsBufferPtr + inputSeqLen);

        // Make a copy of the reponse, because the tensor might be deallocated before the mlperf::QuerySampleComplete()
        resp.outputTokens.resize(outputSeqLen);
        std::copy(outputPtr, outputPtr + outputSeqLen, resp.outputTokens.begin());

#ifdef DEBUG
        // Debug: Print the input and output id
        std::stringstream inputIdStream;
        std::stringstream outputIdStream;
        auto const inputPtr = reinterpret_cast<int32_t*>(inputIdPtr);
        std::vector<int32_t> inputIds(inputPtr, inputPtr + inputSeqLen);
        for (auto id : inputIds)
        {
            inputIdStream << id << ",";
        }
        for (auto id : resp.outputTokens)
        {
            outputIdStream << id << ",";
        }
#endif

        // NOTE: The LoadGen doesn't allow 1-token output, so we append EOS to the end if final SeqLen is 1
        if (outputSeqLen == 1)
        {
            outputSeqLen += 1;
            resp.outputTokens.push_back(mLLMConfig.mEosId);
        }
        resp.QSR.id = requestId;
        resp.QSR.data = reinterpret_cast<uintptr_t>(resp.outputTokens.data());
        resp.QSR.size = static_cast<size_t>(outputSeqLen * sizeof(int32_t));
        // This is not required by GPTJ, but we added here so we can check the token stats.
        resp.QSR.n_tokens = outputSeqLen;
    }

    if (mIsResponder)
    {
        {
            std::scoped_lock<std::mutex> lck((*mServerRespQueueMtxs)[mThreadQueueIdx]);
            (*mServerRespQueue)[mThreadQueueIdx].emplace_back(std::move(resp));
            VLOG(1) << "Rank:" << mMyRank << ", LLMCore sending response for requestId: " << requestId
                    << " RespQueue size: " << (*mServerRespQueue)[mThreadQueueIdx].size();
            (*mServerRespQueueCondVars)[mThreadQueueIdx].notify_one();
        }

        // CommBuffer should be reclaimed if MPI is used
        if (mMySessionSize > 1)
        {
            std::scoped_lock<std::mutex> lck((*mServerReclaimQueueMtxs)[mThreadQueueIdx]);
            (*mServerReclaimQueue)[mThreadQueueIdx].emplace_back(requestId);
            VLOG(1) << "Rank:" << mMyRank << ", LLMCore sending resource reclaim signal for requestId: " << requestId
                    << " ReclaimQueue size: " << (*mServerReclaimQueue)[mThreadQueueIdx].size();
            (*mServerReclaimQueueCondVars)[mThreadQueueIdx].notify_one();
        }
    }

    NVTX_THREAD_END();
}
