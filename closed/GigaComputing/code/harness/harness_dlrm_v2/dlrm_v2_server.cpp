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

#include <chrono>
#include <fstream>
#include <set>

#include "glog/logging.h"
#include "loadgen.h"

#include "batch_maker.h"
#include "dlrm_v2_kernels.h"
#include "dlrm_v2_qsl.h"
#include "dlrm_v2_server.h"

inline int32_t getPersistentCacheSizeLimit()
{
    size_t persistentL2CacheSizeLimit = 0;
#if CUDART_VERSION >= 11030
    CHECK(cudaDeviceGetLimit(&persistentL2CacheSizeLimit, cudaLimitPersistingL2CacheSize) == 0);
#else
    persistentL2CacheSizeLimit = 0;
#endif
    return persistentL2CacheSizeLimit;
}

bool operator==(const nvinfer1::Dims& d1, const nvinfer1 ::Dims& d2)
{
    if (d1.nbDims != d2.nbDims)
        return false;

    for (int it = 0; it < d1.nbDims; it++)
    {
        if (d1.d[it] != d2.d[it])
            return false;
    }

    return true;
}

size_t bindingVolume(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int idx)
{
    return lwis::volume(
        engine->getBindingDimensions(idx), engine->getBindingFormat(idx), engine->hasImplicitBatchDimension());
}

size_t GetEffectiveBatchSize(const std::vector<DLRMv2Task>& tasks)
{
    return std::accumulate(tasks.begin(), tasks.end(), 0ULL,
        [](const size_t curr, const DLRMv2Task& t) { return curr + t.numIndividualPairs; });
}

DLRMv2Core::DLRMv2Core(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int maxBatchSize, const int numBundles,
    const int numCompleteThreads, const int profileIdx, const double elRatio, const bool verboseNVTX)
    : mEngine(engine)
    , mMaxBatchSize(maxBatchSize)
    , mResHandlerPool(numCompleteThreads)
    , mBundleCounter(0)
{
    mContext = InferObject(mEngine->createExecutionContext());

    if (verboseNVTX)
    {
        mContext->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    }
    else
    {
        mContext->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kNONE);
    }

    LOG(INFO) << "Setting profile = " << profileIdx;
    mContext->setOptimizationProfile(profileIdx);
    SetBatchSize(maxBatchSize);
    LOG(INFO) << "Context creation complete";

    // eviction last setup
    if (elRatio > 0.0)
    {
        mContext->setPersistentCacheLimit(elRatio * getPersistentCacheSizeLimit());
    }

    CHECK_EQ(cudaStreamCreate(&mH2DStream), cudaSuccess);
    CHECK_EQ(cudaStreamCreate(&mComputeStream), cudaSuccess);
    CHECK_EQ(cudaStreamCreate(&mD2HStream), cudaSuccess);
    LOG(INFO) << "Created streams";

    const size_t numBindings = mEngine->getNbBindings();
    CHECK_EQ(numBindings / mEngine->getNbOptimizationProfiles(), 3) << "Harness expects 3 bindings per engine profile";
    const size_t firstBinding = profileIdx * numBindings / mEngine->getNbOptimizationProfiles();

    mNumInVol = bindingVolume(mEngine, firstBinding + 0);
    LOG(INFO) << "Profile - Numeric Input Volume: " << mNumInVol;

    mCatInVol = bindingVolume(mEngine, firstBinding + 1);
    LOG(INFO) << "Profile - Categorical Input Volume: " << mCatInVol;

    mOutVol = bindingVolume(mEngine, firstBinding + 2);
    LOG(INFO) << "Profile - Output Volume: " << mOutVol;

    // setup IO bindings
    mBindings.resize(numBundles);
    for (int i = 0; i < numBundles; ++i)
    {
        const auto bundle = std::make_shared<DLRMv2EventBufferBundle>(i, mNumInVol, mCatInVol, mOutVol, mMaxBatchSize);
        mEventBufferBundle.push_back(bundle);

        // set this profile's bindings to DevicePtrs, set rest to nullptr
        mBindings[i].assign(numBindings, nullptr);
        mBindings[i][firstBinding + 0] = bundle->numericInputBuf.GetDevicePtr();
        mBindings[i][firstBinding + 1] = bundle->categoricalInputBuf.GetDevicePtr();
        mBindings[i][firstBinding + 2] = bundle->outputBuf.GetDevicePtr();
    }

    LOG(INFO) << "Created copy streams and buffers";
    LOG(INFO) << "Setup complete";
}

void DLRMv2Core::SetBatchSize(int batchSize)
{
    const int profileNum = mContext->getOptimizationProfile();
    CHECK_EQ(profileNum >= 0 && profileNum < mEngine->getNbOptimizationProfiles(), true);

    const int numBindings = mEngine->getNbBindings() / mEngine->getNbOptimizationProfiles();
    for (int i = 0; i < numBindings; i++)
    {
        if (mEngine->bindingIsInput(i))
        {
            const auto bindingIdx = numBindings * profileNum + i;
            auto inputDims = mContext->getBindingDimensions(bindingIdx);

            if (inputDims.d[0] != batchSize)
            {
                inputDims.d[0] = batchSize;
                CHECK_EQ(mContext->setBindingDimensions(bindingIdx, inputDims), true);
            }
        }
    }

    CHECK_EQ(mContext->allInputDimensionsSpecified(), true);
}

void DLRMv2Core::infer(std::shared_ptr<DLRMv2EventBufferBundle> ebBundle,
    const size_t batchSize, // We assume that this batchsize is the effective batch size (padded to even value)
    const std::vector<DLRMv2Task>& tasks, Batch* batch, void (*h2dCallBack)(void*), DLRMv2InferCallback resultCallback,
    const DLRMv2NumericInputType* numericInputPtr, const DLRMv2CategoricalInputType* categoricalInputPtr) noexcept
{
    DLOG(INFO) << "infer() batch = " << batch;
    NVTX_RANGE_PUSH(("DLRMv2Core::infer: batchSize=" + std::to_string(batchSize)).c_str());
    CHECK_EQ(batchSize <= mMaxBatchSize, true);
    SetBatchSize(batchSize);

    // Copy buffers
    ebBundle->numericInputBuf.H2DAsync(numericInputPtr, batchSize, mH2DStream);
    ebBundle->categoricalInputBuf.H2DAsync(categoricalInputPtr, batchSize, mH2DStream);
    ebBundle->recordH2D(mH2DStream);

    // callback upon H2D completion
    CHECK_EQ(cudaLaunchHostFunc(mH2DStream, h2dCallBack, batch), cudaSuccess);

    void** bindings = mBindings[ebBundle->idx].data();

    // Run inference
    ebBundle->makeAwaitH2D(mComputeStream);
    CHECK_EQ(mContext->enqueueV2(bindings, mComputeStream, nullptr), true);
    ebBundle->recordInfer(mComputeStream);

    // Get output
    ebBundle->makeAwaitInfer(mD2HStream);
    ebBundle->outputBuf.D2HAsync(ebBundle->outputBuf.GetHostPtr(), batchSize, mD2HStream);
    ebBundle->recordD2H(mD2HStream);

    // setup result callback
    DLRMv2DeferredResult* deferredResult = new DLRMv2DeferredResult{
        batchSize,
        ebBundle->outputBuf.GetHostPtr(),
        std::move(tasks),
        resultCallback,
        [this](const DLRMv2Result& r) { mResHandlerPool.Enqueue(r); },
    };
    CHECK_EQ(cudaLaunchHostFunc(
                 mD2HStream,
                 [](void* deferredResult_ptr) -> void
                 {
                     NVTX_RANGE_PUSH("deferredResult processing");

                     const DLRMv2DeferredResult* deferredResult
                         = reinterpret_cast<DLRMv2DeferredResult*>(deferredResult_ptr);
                     const DLRMv2Result result = {
                         std::make_shared<std::vector<DLRMv2OutputType>>(
                             deferredResult->outputs, deferredResult->outputs + deferredResult->batchSize),
                         std::move(deferredResult->tasks),
                         deferredResult->callback,
                     };

                     deferredResult->resultCallback(result);
                     delete deferredResult;

                     NVTX_RANGE_POP();
                 },
                 deferredResult),
        cudaSuccess);

    NVTX_RANGE_POP();
}

void DLRMv2Core::inferFromDevice(std::shared_ptr<DLRMv2EventBufferBundle> ebBundle, const size_t batchSize,
    const std::vector<DLRMv2Task>& tasks, const DLRMv2InferCallback resultCallback) noexcept
{
    NVTX_RANGE_PUSH(("DLRMv2Core::inferFromDevice: batchSize=" + std::to_string(batchSize)).c_str());

    CHECK_EQ((batchSize % 2 == 0) && (batchSize <= mMaxBatchSize), true);
    SetBatchSize(batchSize);

    bool contiguousData = true;
    for (size_t i = 1; (i < tasks.size()) && contiguousData; ++i)
    {
        contiguousData &= ebBundle->numericInputPtrBuf.GetHostPtr()[i]
            == ebBundle->numericInputPtrBuf.GetHostPtr()[i - 1]
                + ebBundle->sampleSizesBuf.GetHostPtr()[i - 1] * mNumInVol;

        contiguousData &= ebBundle->categoricalInputPtrBuf.GetHostPtr()[i]
            == ebBundle->categoricalInputPtrBuf.GetHostPtr()[i - 1]
                + ebBundle->sampleSizesBuf.GetHostPtr()[i - 1] * mCatInVol;
    }

    // bring data in from host, contiguous
    if (!contiguousData)
    {
        ebBundle->numericInputPtrBuf.H2DAsync(ebBundle->numericInputPtrBuf.GetHostPtr(), tasks.size(), mH2DStream);
        ebBundle->categoricalInputPtrBuf.H2DAsync(
            ebBundle->categoricalInputPtrBuf.GetHostPtr(), tasks.size(), mH2DStream);
        ebBundle->sampleSizesBuf.H2DAsync(ebBundle->sampleSizesBuf.GetHostPtr(), tasks.size(), mH2DStream);
        ebBundle->sampleOffsetsBuf.H2DAsync(ebBundle->sampleOffsetsBuf.GetHostPtr(), tasks.size(), mH2DStream);
    }

    ebBundle->recordH2D(mH2DStream);

    // Run inference
    ebBundle->makeAwaitH2D(mComputeStream);

    // Run gather kernel to load input data into bindings
    if (!contiguousData)
    {
        runGatherKernel((const DLRMv2NumericInputType**) (ebBundle->numericInputPtrBuf.GetDevicePtr()),
            (const DLRMv2CategoricalInputType**) (ebBundle->categoricalInputPtrBuf.GetDevicePtr()),
            (const size_t*) (ebBundle->sampleSizesBuf.GetDevicePtr()),
            (const size_t*) (ebBundle->sampleOffsetsBuf.GetDevicePtr()), ebBundle->numericInputBuf.GetDevicePtr(),
            ebBundle->categoricalInputBuf.GetDevicePtr(), static_cast<int>(tasks.size()), static_cast<int>(mNumInVol),
            static_cast<int>(mCatInVol), mComputeStream);
    }

    void** bindings = mBindings[ebBundle->idx].data();

    // use data source as is, contiguous
    std::array<void*, 3> actualBindings;
    if (contiguousData)
    {
        actualBindings[0] = ebBundle->numericInputPtrBuf.GetHostPtr()[0];
        actualBindings[1] = ebBundle->categoricalInputPtrBuf.GetHostPtr()[0];
        actualBindings[2] = bindings[2];
        bindings = actualBindings.data();
    }

    CHECK_EQ(mContext->enqueueV2(bindings, mComputeStream, nullptr), true);
    ebBundle->recordInfer(mComputeStream);

    // Get output
    ebBundle->makeAwaitInfer(mD2HStream);
    ebBundle->outputBuf.D2HAsync(ebBundle->outputBuf.GetHostPtr(), batchSize, mD2HStream);
    ebBundle->recordD2H(mD2HStream);

    // setup result callback
    DLRMv2DeferredResult* deferredResult = new DLRMv2DeferredResult{
        batchSize,
        ebBundle->outputBuf.GetHostPtr(),
        std::move(tasks),
        resultCallback,
        [this](const DLRMv2Result& r) { mResHandlerPool.Enqueue(r); },
    };
    CHECK_EQ(cudaLaunchHostFunc(
                 mD2HStream,
                 [](void* deferredResult_ptr) -> void
                 {
                     NVTX_RANGE_PUSH("deferredResult processing");

                     const DLRMv2DeferredResult* deferredResult
                         = reinterpret_cast<DLRMv2DeferredResult*>(deferredResult_ptr);
                     const DLRMv2Result result = {
                         std::make_shared<std::vector<DLRMv2OutputType>>(
                             deferredResult->outputs, deferredResult->outputs + deferredResult->batchSize),
                         std::move(deferredResult->tasks),
                         deferredResult->callback,
                     };

                     deferredResult->resultCallback(result);
                     delete deferredResult;

                     NVTX_RANGE_POP();
                 },
                 deferredResult),
        cudaSuccess);

    NVTX_RANGE_POP();
}

void DLRMv2Core::WarmUp(const double duration) noexcept
{
    double elapsed = 0.0;
    const auto tStart = std::chrono::high_resolution_clock::now();

    std::vector<DLRMv2Task> dummyTasks(mMaxBatchSize, {{0, 0}, 1});
    std::vector<DLRMv2NumericInputType> dummyNumIn(mMaxBatchSize * mNumInVol);
    std::vector<DLRMv2CategoricalInputType> dummyCatIn(mMaxBatchSize * mCatInVol);

    LOG(INFO) << "Running warmup for " << duration << "s.";
    do
    {
        const auto bundle = NextForegroundBundle();
        bundle->syncD2H();
        infer(
            bundle, mMaxBatchSize, dummyTasks, nullptr, [](void* batch) -> void { return; },
            [](std::vector<mlperf::QuerySampleResponse>&) { return; }, dummyNumIn.data(), dummyCatIn.data());
        elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - tStart).count();

    } while (elapsed < duration);

    for (size_t i = 0; i < mEventBufferBundle.size(); ++i)
        NextForegroundBundle()->syncD2H();

    LOG(INFO) << "Warmup complete, ran for " << elapsed << "s.";
}

std::shared_ptr<DLRMv2EventBufferBundle> DLRMv2Core::NextForegroundBundle() noexcept
{
    const size_t idx = mBundleCounter;
    mBundleCounter = (mBundleCounter + 1) % mEventBufferBundle.size();
    return mEventBufferBundle[idx];
}

DLRMv2Core::~DLRMv2Core()
{
    CHECK_EQ(cudaStreamDestroy(mH2DStream), cudaSuccess);
    CHECK_EQ(cudaStreamDestroy(mComputeStream), cudaSuccess);
    CHECK_EQ(cudaStreamDestroy(mD2HStream), cudaSuccess);
}

DLRMv2Server::DLRMv2Server(const std::string name, const std::string enginePath,
    const std::vector<DLRMv2SampleLibraryPtr_t> qsls, const std::vector<int>& gpus, const int maxBatchSize,
    const int numBundles, const int numCompleteThreads, const int numDLRMv2Cores, const double warmupDuration,
    const int numStagingThreads, const int numStagingBatches, const int maxPairsPerThread, const bool checkContiguity,
    const bool startFromDevice, const NumaConfig numaConfig, const uint64_t serverNumIssueQueryThreads,
    const bool useBatcherPerGpu, const double elRatio, const NumaToQSLMap numaToQslMap, const bool verbostNVTX)
    : mName{name}
    , mQsls{qsls}
    , mStartFromDevice(startFromDevice)
    , mStopWork{false}
    , mDLRMv2Cores{gpus.size() * numDLRMv2Cores}
    , mNumInVol(0)
    , mCatInVol(0)
    , mNumaConfig(numaConfig)
    , mGpuToNumaMap(getGpuToNumaMap(mNumaConfig))
    , mServerNumIssueQueryThreads(serverNumIssueQueryThreads)
    , mUseBatcherPerGpu(useBatcherPerGpu)
    , mElRatio(elRatio)
    , mNumaToQslMap(numaToQslMap)
    , mVerboseNVTX(verbostNVTX)
{
    NVTX_NAME_THIS_THREAD("DLRMv2Server");
    LOG(INFO) << "Using " << numDLRMv2Cores << " DLRMv2 Core(s) per Device";

    if (UseNuma())
    {
        LOG(INFO) << "Using NUMA nodes";
        if (numaToQslMap.empty())
        {
            CHECK_EQ(mNumaConfig.size(), qsls.size()) << "Number of QSLs should match number of NUMA nodes!";
        }
    }

    mMaxBatchSize = maxBatchSize;
    mMaxBatchSize = mMaxBatchSize % 2 == 1 ? mMaxBatchSize - 1 : mMaxBatchSize;

    // setup all devices
    {
        std::vector<std::thread> setupThreads;
        for (const auto& deviceId : gpus)
        {
            setupThreads.emplace_back(&DLRMv2Server::SetupDevice, this, enginePath, numBundles, numCompleteThreads,
                numDLRMv2Cores, warmupDuration, deviceId);
        }

        for (auto& t : setupThreads)
        {
            t.join();
        }
    }

    if (mServerNumIssueQueryThreads > 0)
    {
        CHECK_EQ(mServerNumIssueQueryThreads, gpus.size());

        // when start_from_device is used, use only one issue query thread
        CHECK(!startFromDevice);

        LOG(INFO) << "Use number of server IssueQuery threads = " << mServerNumIssueQueryThreads;
        for (int deviceId = 0; deviceId < mServerNumIssueQueryThreads; ++deviceId)
        {
            if (UseNuma())
            {
                CHECK_LE(mNumaConfig.size(), mServerNumIssueQueryThreads);
                bindNumaMemPolicy(mGpuToNumaMap[deviceId], mNumaConfig.size());
            }

            mIssueQueryThreads.emplace_back(&DLRMv2Server::StartIssueThread, this, deviceId);

            if (UseNuma())
            {
                bindThreadToCpus(mIssueQueryThreads.back(), mNumaConfig[mGpuToNumaMap[deviceId]].second);
                resetNumaMemPolicy();
            }

            mThreadMap[mIssueQueryThreads.back().get_id()] = deviceId;
        }
    }

    if (!startFromDevice)
    {

        // clang-format off
        const int numBatchMakers = mServerNumIssueQueryThreads > 0 ?
                                    mServerNumIssueQueryThreads :
                                    mUseBatcherPerGpu ? gpus.size() : 1;
        // clang-format on

        // Construct BatchMaker, in the order of device it works with
        for (int i = 0; i < numBatchMakers; ++i)
        {
            const auto devId = i;
            const auto numaNum = UseNuma() ? mNumaConfig.size() : 0;
            const auto numaIdx = UseNuma() ? mGpuToNumaMap[devId] : -1;
            const auto cpus = UseNuma() ? mNumaConfig[numaIdx].second : std::vector<int>();
            const auto qslIdx = UseNuma() ? (mNumaToQslMap.empty() ? numaIdx : mNumaToQslMap[numaIdx]) : 0;

            mBatchMakers.emplace_back(std::make_shared<BatchMaker>(numStagingThreads, numStagingBatches, maxBatchSize,
                maxPairsPerThread, mNumInVol, mCatInVol, checkContiguity, qsls[qslIdx], devId, numaIdx, numaNum, cpus));
        }
    }

    for (int deviceId = 0; deviceId < gpus.size(); ++deviceId)
    {
        // Bind the worker to the NUMA memory if needed
        if (UseNuma())
        {
            bindNumaMemPolicy(mGpuToNumaMap[deviceId], mNumaConfig.size());
        }

        mWorkerThreads.reserve(numDLRMv2Cores);
        for (size_t profileIdx = 0; profileIdx < numDLRMv2Cores; ++profileIdx)
        {
            const auto dlrmv2Core = mDLRMv2Cores[deviceId * numDLRMv2Cores + profileIdx];
            mWorkerThreads.emplace_back(
                startFromDevice ? &DLRMv2Server::ProcessTasksFromDevice : &DLRMv2Server::ProcessTasks, this, dlrmv2Core,
                deviceId, profileIdx);

            // Limit the worker thread to the closest CPUs.
            if (UseNuma())
            {
                bindThreadToCpus(mWorkerThreads.back(), mNumaConfig[mGpuToNumaMap[deviceId]].second);
            }
        }

        // Reset memory allocation setting
        if (UseNuma())
        {
            resetNumaMemPolicy();
        }
    }
}

void DLRMv2Server::SetupDevice(const std::string enginePath, const int numBundles, const int numCompleteThreads,
    const int numDLRMv2Cores, const int warmupDuration, const int deviceId) noexcept
{
    if (UseNuma())
    {
        bindNumaMemPolicy(mGpuToNumaMap[deviceId], mNumaConfig.size());
    }

    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
    const auto engine = DeserializeEngine(enginePath);
    CHECK_LE(numDLRMv2Cores, engine->getNbOptimizationProfiles());

    mNumInVol = bindingVolume(engine, 0);
    mCatInVol = bindingVolume(engine, 1);

    // create DLRMv2Core(s)
    for (size_t profileIdx = 0; profileIdx < numDLRMv2Cores; ++profileIdx)
    {
        const auto index = deviceId * numDLRMv2Cores + profileIdx;
        mDLRMv2Cores[index] = std::make_shared<DLRMv2Core>(
            engine, mMaxBatchSize, numBundles, numCompleteThreads, profileIdx, mElRatio, mVerboseNVTX);

        CHECK_LE(mMaxBatchSize, mDLRMv2Cores[index]->GetMaxBatchSize());
        mDLRMv2Cores[index]->WarmUp(warmupDuration);
    }

    // Reset memory allocation setting
    if (UseNuma())
    {
        resetNumaMemPolicy();
    }
};

DLRMv2Server::~DLRMv2Server()
{
    DLOG(INFO) << "~DLRMv2Server";

    {
        std::unique_lock<std::mutex> lck(mMtx);
        mStopWork = true;
        mCondVar.notify_all();
    }

    for (auto batchMaker : mBatchMakers)
    {
        if (batchMaker)
        {
            batchMaker->StopWork();
        }
    }

    for (auto& workerThread : mWorkerThreads)
    {
        workerThread.join();
    }

    for (auto& issueQueryThread : mIssueQueryThreads)
    {
        issueQueryThread.join();
    }
}

const std::string& DLRMv2Server::Name()
{
    return mName;
}

void DLRMv2Server::StartIssueThread(const int tId) const noexcept
{
    LOG(INFO) << "Registering IssueQueryThread of " << std::this_thread::get_id() << " for Device[" << tId << "]";
    mlperf::RegisterIssueQueryThread();
}

void DLRMv2Server::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    NVTX_RANGE_PUSH(("DLRMv2Server::IssueQuery for " + std::to_string(samples.size()) + " samples").c_str());

    if (mStartFromDevice)
    {
        std::vector<size_t> numPairs(samples.size());
        std::transform(samples.begin(), samples.end(), numPairs.begin(),
            [&](const mlperf::QuerySample& x) { return mQsls[0]->GetNumUserItemPairs(x.index); });

        // push to task queue from device
        {
            std::unique_lock<std::mutex> lck(mMtx);
            for (size_t i = 0; i < samples.size(); ++i)
                mTasks.push_back({samples[i], numPairs[i]});
            mCondVar.notify_one();
        }
    }

    else
    {
        if (mServerNumIssueQueryThreads > 0)
        {
            // tId is device Id
            const int tId = mThreadMap[std::this_thread::get_id()];
            mBatchMakers[tId]->IssueQuery(samples, 0, samples.size());
        }

        else if (samples.size() < SPLIT_THRESHOLD)
        {
            const int nextBatchMakerIdx = (mPrevBatchMakerIdx + 1) % mBatchMakers.size();
            mBatchMakers[nextBatchMakerIdx]->IssueQuery(samples, 0, samples.size());
            mPrevBatchMakerIdx = nextBatchMakerIdx;
        }

        else
        {
            // Offline case where LoadGen won't use multiple issue threads
            // One time sample distribution - NUMA thread binding won't help (syscall outweighs the possible upside)
            const int64_t numSamplesTotal = samples.size();
            const int64_t numBatchMakers = mBatchMakers.size();

            // round robin to BatchMakers
            const int64_t numSamplesPerBM = numSamplesTotal / numBatchMakers;
            const int64_t remainder = numSamplesTotal % numBatchMakers;

            // Use a thread per batchMaker to issue queries in parallel. Each batchMaker has its own lock, so running
            // with multiple threads should not cause lock contention.
            const auto issueQueryOneBatchMaker
                = [](std::shared_ptr<BatchMaker> batchMaker, const std::vector<mlperf::QuerySample>& samples,
                      const int64_t offset, const int64_t size) { batchMaker->IssueQuery(samples, offset, size); };
            std::vector<std::thread> issueThreads;
            issueThreads.reserve(numBatchMakers);

            int64_t offset = 0;
            for (int64_t myIdx = 0; myIdx < numBatchMakers; ++myIdx)
            {
                const int64_t size = numSamplesPerBM + (myIdx < remainder ? 1 : 0);
                if (size == 0)
                {
                    break;
                }

                issueThreads.emplace_back(
                    issueQueryOneBatchMaker, mBatchMakers[myIdx], std::ref(samples), offset, size);
                offset += size;
            }

            CHECK_EQ(offset, numSamplesTotal);
            for (auto& th : issueThreads)
            {
                th.join();
            }
        }
    }

    NVTX_RANGE_POP();
}

void DLRMv2Server::FlushQueries()
{
    NVTX_RANGE_PUSH("DLRMv2Server::FlushQueries");

    if (!mStartFromDevice)
    {
        for (auto batchMaker : mBatchMakers)
        {
            batchMaker->FlushQueries();
        }
    }

    NVTX_RANGE_POP();
}

void DLRMv2Server::ProcessTasks(
    std::shared_ptr<DLRMv2Core> dlrmv2Core, const int deviceId, const int profileIdx) noexcept
{
    NVTX_NAME_THIS_THREAD(("ProcessTasks" + std::to_string(profileIdx)).c_str());
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);

    if (UseNuma())
    {
        bindNumaMemPolicy(mGpuToNumaMap[deviceId], mNumaConfig.size());
    }

    // Process samples in batches
    while (true)
    {
        const auto ebBundle = dlrmv2Core->NextForegroundBundle();

        // Only grab tasks when the copy stream is ready. This allows `tasks` to fill up with more batches while the
        // stream is still working. The GPU should still be busy with inference on other bundle(s)
        NVTX_RANGE_PUSH("DLRMv2Server::ProcessTasks syncing for the foreground bundle to complete");
        ebBundle->syncD2H();
        NVTX_RANGE_POP();
        NVTX_RANGE_PUSH(("DLRMv2Server::ProcessTasks iteration, profile" + std::to_string(profileIdx)).c_str());

        const int batchMakerIdx = mUseBatcherPerGpu ? deviceId : 0;
        Batch* batch = mBatchMakers[batchMakerIdx]->GetBatch();

        if (!batch)
        {
            NVTX_RANGE_POP();
            break;
        }

        const size_t actualBatchSize = batch->getCommittedCopies();
        const auto tasks = batch->getTasks();
        const auto numericHostPtr = batch->getNumericHostPtr();
        const auto categoricalHostPtr = batch->getCategoricalHostPtr();
        DLOG(INFO) << "Batch Size : " << actualBatchSize;

        dlrmv2Core->infer(
            ebBundle, actualBatchSize, tasks, batch,
            [](void* batch) -> void
            { reinterpret_cast<Batch*>(batch)->mBatchMaker->NotifyH2D(reinterpret_cast<Batch*>(batch)); },
            [=](std::vector<mlperf::QuerySampleResponse>& responses)
            { mlperf::QuerySamplesComplete(responses.data(), responses.size()); },
            numericHostPtr, categoricalHostPtr);

        NVTX_RANGE_POP();
    }

    // Reset memory allocation setting
    if (UseNuma())
    {
        resetNumaMemPolicy();
    }
}

void DLRMv2Server::ProcessTasksFromDevice(
    std::shared_ptr<DLRMv2Core> dlrmv2Core, const int deviceId, const int profileIdx) noexcept
{
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);

    // Process samples in batches
    while (true)
    {
        auto ebBundle = dlrmv2Core->NextForegroundBundle();

        // Only grab tasks when the copy stream is ready. This allows `tasks` to fill up with more batches while the
        // stream is still working. The GPU should still be busy with inference on other bundle(s)
        NVTX_RANGE_PUSH("DLRMv2Server::ProcessTasksFromDevice syncing for the foreground bundle to complete");
        ebBundle->syncD2H();
        NVTX_RANGE_POP();
        NVTX_RANGE_PUSH(
            ("DLRMv2Server::ProcessTasksFromDevice iteration, profile" + std::to_string(profileIdx)).c_str());

        auto tasks = GetBatch();
        if (tasks.empty())
        {
            NVTX_RANGE_POP();
            break;
        }

        NVTX_RANGE_PUSH("DLRMv2Server::ProcessTasksFromDevice preparing batch host buffers");
        const size_t originalBatchSize = std::accumulate(tasks.begin(), tasks.end(), (size_t) 0,
            [](size_t x, const DLRMv2Task& y) { return x + y.numIndividualPairs; });

        // Pad the tasks so that the batch size is even
        const bool isBatchPadded = (originalBatchSize % 2) == 1;
        if (isBatchPadded)
        {
            tasks.push_back({tasks.back().querySample, 1});
        }

        NVTX_RANGE_PUSH("DLRMv2Server::ProcessTasksFromDevice preparing numericInputPtrBuf");
        std::transform(tasks.begin(), tasks.end(), ebBundle->numericInputPtrBuf.GetHostPtr(),
            [&](const DLRMv2Task& x) {
                return reinterpret_cast<DLRMv2NumericInputType*>(
                    mQsls[0]->GetSampleAddress(x.querySample.index, 0, 0, deviceId));
            });
        NVTX_RANGE_POP();

        NVTX_RANGE_PUSH("DLRMv2Server::ProcessTasksFromDevice preparing categoricalInputPtrBuf");
        std::transform(tasks.begin(), tasks.end(), ebBundle->categoricalInputPtrBuf.GetHostPtr(),
            [&](const DLRMv2Task& x)
            {
                return reinterpret_cast<DLRMv2CategoricalInputType*>(
                    mQsls[0]->GetSampleAddress(x.querySample.index, 1, 0, deviceId));
            });
        NVTX_RANGE_POP();

        NVTX_RANGE_PUSH("DLRMv2Server::ProcessTasksFromDevice preparing sampleSizesBuf");
        std::transform(tasks.begin(), tasks.end(), ebBundle->sampleSizesBuf.GetHostPtr(),
            [&](const DLRMv2Task& x) { return x.numIndividualPairs; });
        NVTX_RANGE_POP();

        NVTX_RANGE_PUSH("DLRMv2Server::ProcessTasksFromDevice preparing sampleOffsetsBuf");
        ebBundle->sampleOffsetsBuf.GetHostPtr()[0] = 0;
        std::partial_sum(ebBundle->sampleSizesBuf.GetHostPtr(),
            ebBundle->sampleSizesBuf.GetHostPtr() + tasks.size() - 1, ebBundle->sampleOffsetsBuf.GetHostPtr() + 1);
        NVTX_RANGE_POP();

        const size_t batchSize = std::accumulate(
            ebBundle->sampleSizesBuf.GetHostPtr(), ebBundle->sampleSizesBuf.GetHostPtr() + tasks.size(), (size_t) 0);
        NVTX_RANGE_POP();

        dlrmv2Core->inferFromDevice(ebBundle, batchSize, tasks,
            [=](std::vector<mlperf::QuerySampleResponse>& responses)
            {
                if (isBatchPadded)
                {
                    responses.pop_back();
                }

                mlperf::QuerySamplesComplete(responses.data(), responses.size());
            });

        NVTX_RANGE_POP();
    }
}

std::shared_ptr<nvinfer1::ICudaEngine> DLRMv2Server::DeserializeEngine(const std::string& enginePath) noexcept
{
    int device;
    CHECK_EQ(cudaGetDevice(&device), cudaSuccess);
    LOG(INFO) << "Deserializing Engine on GPU#" << device;

    std::vector<char> trtModelStream{};
    const auto size = lwis::GetModelStream(trtModelStream, enginePath);
    const auto runtime = InferObject(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));
    const auto engine = InferObject(runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr));
    LOG(INFO) << "Engine - Device Memory requirements: " << engine->getDeviceMemorySize();
    LOG(INFO) << "Engine - Number of Optimization Profiles: " << engine->getNbOptimizationProfiles();

    return engine;
}

std::vector<DLRMv2Task> DLRMv2Server::GetBatch() noexcept
{
    NVTX_RANGE_PUSH("DLRMv2Server::GetBatch");

    std::vector<DLRMv2Task> currentTasks;

    // Wait for the new work to arrive
    std::unique_lock<std::mutex> lck(mMtx);
    mCondVar.wait(lck, [&] { return (!mTasks.empty()) || mStopWork; });
    NVTX_RANGE_PUSH(("Extracting tasks from queue with length " + std::to_string(mTasks.size())).c_str());

    // Consume up to mMaxBatchSize pairs
    int currentBatchSize = 0;

    currentTasks.reserve(mTasks.size());
    while (!mTasks.empty())
    {
        const auto& topTask = mTasks.front();
        currentBatchSize += topTask.numIndividualPairs;

        if (currentBatchSize > mMaxBatchSize)
            break;

        currentTasks.push_back(topTask);
        mTasks.pop_front();
    }

    // Let some other thread to consume more tasks
    if (!mTasks.empty())
        mCondVar.notify_one();

    NVTX_RANGE_POP();
    NVTX_RANGE_POP();

    return currentTasks;
}
