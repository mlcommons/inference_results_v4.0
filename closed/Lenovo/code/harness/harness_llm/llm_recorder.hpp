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

#include "tensorrt_llm/batch_manager/inferenceRequest.h"

#include <chrono>
#include <mutex>

using timePoint = std::chrono::time_point<std::chrono::steady_clock>;

struct BenchInfo
{
    BenchInfo() {}

    BenchInfo(int32_t inputSeqLength, timePoint issueQuery)
        : mInputSeqLength(static_cast<uint16_t>(inputSeqLength))
        , tIssueQuery(issueQuery)
    {
    }

    // Use smaller container for the sake of reduced overhead
    uint16_t mInputSeqLength;
    uint16_t mOutputSeqLength;
    timePoint tIssueQuery;      // LoadGen issues queries
    timePoint tStartInference;  // Batch manager acquires queries
    timePoint tFirstToken;      // LoadGen receive first token
    timePoint tFinishInference; // Batch manager finished queries
    timePoint tRecvResponse;    // LoadGen receive responses from the worker
    timePoint tQueryComplete;   // SUT reports query complete
    float mLatency;             // total end to end latency in millisecond
};

/*
A per-query benchmarking tracker.
*/
class Recorder
{
public:
    Recorder(bool reportFirstToken = false) : mReportFirstToken(reportFirstToken) {}

    void initialize()
    {
        mStart = std::chrono::steady_clock::now();
    }

    void finalize()
    {
        mEnd = std::chrono::steady_clock::now();
    }

    void recordIssueQuery(uint64_t requestId, int32_t inputSeqLength)
    {
        mNumSamples += 1;
        auto const tp = std::chrono::steady_clock::now();
        std::scoped_lock<std::mutex> lck(mMtx);
        mRequestBenchInfos[requestId] = BenchInfo(inputSeqLength, tp);
    }

    void recordFirstToken(uint64_t requestId)
    {
        auto const tp = std::chrono::steady_clock::now();
        std::scoped_lock<std::mutex> lck(mMtx);
        mRequestBenchInfos[requestId].tFirstToken = tp;
    }

    void recordStartInference(uint64_t requestId)
    {
        auto const tp = std::chrono::steady_clock::now();
        std::scoped_lock<std::mutex> lck(mMtx);
        mRequestBenchInfos[requestId].tStartInference = tp;
    }

    void recordFinishInference(uint64_t requestId)
    {
        auto const tp = std::chrono::steady_clock::now();
        std::scoped_lock<std::mutex> lck(mMtx);
        mRequestBenchInfos[requestId].tFinishInference = tp;
    }

    void recordQueryComplete(uint64_t requestId, int32_t outputSeqLength)
    {
        auto const tp = std::chrono::steady_clock::now();
        std::scoped_lock<std::mutex> lck(mMtx);
        mRequestBenchInfos[requestId].mOutputSeqLength = static_cast<uint16_t>(outputSeqLength);
        mRequestBenchInfos[requestId].mLatency
            = std::chrono::duration<float, std::milli>(tp - mRequestBenchInfos[requestId].tIssueQuery).count();
        mRequestBenchInfos[requestId].tQueryComplete = tp;
    }

    void calculateMetrics()
    {
        mNumSamples = mRequestBenchInfos.size();
        mTotalLatency = std::chrono::duration<float, std::milli>(mEnd - mStart).count();
        mSeqThroughput = mNumSamples / (mTotalLatency / 1000);
        mAvgSeqLatency = 0;
        int totalOutputTokens = 0;
        for (auto& reqInfo : mRequestBenchInfos)
        {
            mAvgSeqLatency += reqInfo.second.mLatency;
            totalOutputTokens += reqInfo.second.mOutputSeqLength;
        }
        mAvgSeqLatency /= mNumSamples;
        mTokenThroughput = totalOutputTokens / (mTotalLatency / 1000);
    }

    void report()
    {
        printf("[BENCHMARK] num_samples(ms) %d\n", mNumSamples);
        printf("[BENCHMARK] total_latency(ms) %.2f\n", mTotalLatency);
        printf("[BENCHMARK] seq_throughput(seq/sec) %.2f\n", mSeqThroughput);
        printf("[BENCHMARK] avg_sequence_latency(ms) %.2f\n", mAvgSeqLatency);
        printf("[BENCHMARK] token_throughput(token/sec) %.2f\n", mTokenThroughput);

        // TODO: Save the metrics to a csv file
    }

private:
    std::unordered_map<uint64_t, BenchInfo> mRequestBenchInfos;

    timePoint mStart;
    timePoint mEnd;
    int32_t mNumSamples;
    float mTotalLatency;
    float mSeqThroughput;
    float mAvgSeqLatency;
    float mTokenThroughput;

    std::mutex mMtx;
    bool mReportFirstToken = false;
}; // class Recorder
