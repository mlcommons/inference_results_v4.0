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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include <cudnn.h>
#include <numpy.hpp>

#include "dlrmv2EmbeddingLookupPlugin.h"
#include "dlrmv2Helper.h"

#include "embedding_gather_launch.hpp"
#include "embedding_remap_launch.h"

/*
 * ========== CHANGELOG ==========
 * 1.0 : fp16, fp32 embedding table support, useRowFrequencies optimization
 * 1.1 : int8 support added, multi-gpu bug fixes
 * ===============================
 */

namespace
{
const char* DLRMv2_INTERACTIONS_PLUGIN_VERSION = "1.1";
const char* DLRMv2_INTERACTIONS_PLUGIN_NAME = "DLRMv2_EMBEDDING_LOOKUP_TRT";
constexpr int MAX_FILEPATH_LENGTH = 1024;
constexpr int NUM_CATEGORICAL_FEATURES = 26;
} // namespace

using namespace nvinfer1;
using namespace nvinfer1::plugin;

/* static initializations */
std::mutex DLRMv2EmbeddingLookupPlugin::mSharedDataMutex{};
DLRMv2EmbeddingLookupPlugin::HostData DLRMv2EmbeddingLookupPlugin::mHostData{};
std::map<int, DLRMv2EmbeddingLookupPlugin::DeviceData*> DLRMv2EmbeddingLookupPlugin::mDeviceData{};

PluginFieldCollection DLRMv2EmbeddingLookupPluginCreator::mFC{};
std::vector<PluginField> DLRMv2EmbeddingLookupPluginCreator::mPluginAttributes{};
REGISTER_TENSORRT_PLUGIN(DLRMv2EmbeddingLookupPluginCreator);

DLRMv2EmbeddingLookupPlugin::DLRMv2EmbeddingLookupPlugin(const int embeddingSize, const int embeddingRows,
    const float embeddingWeightsOnGpuPart, const std::vector<int>& tableHotness, const std::vector<int>& tableOffsets,
    const int batchSize, const int embedHotnessTotal, const std::string& embeddingWeightsFilepath,
    const std::string& rowFrequenciesFilepath, const std::string& embeddingScalesFilepath, const int reducedPrecisionIO)
    : mInitialized(false)
    , mLocalDeviceData(nullptr)
    , mEmbeddingSize(embeddingSize)
    , mEmbeddingRows(embeddingRows)
    , mEmbeddingWeightsOnGpuPart(embeddingWeightsOnGpuPart)
    , mTableHotness(tableHotness)
    , mTableOffsets(tableOffsets)
    , mBatchSize(batchSize)
    , mEmbedHotnessTotal(embedHotnessTotal)
    , mEmbeddingWeightsFilepath(embeddingWeightsFilepath)
    , mRowFrequenciesFilepath(rowFrequenciesFilepath)
    , mEmbeddingScalesFilepath(embeddingScalesFilepath)
    , mReducedPrecisionIO(reducedPrecisionIO)
{
}

DLRMv2EmbeddingLookupPlugin::DLRMv2EmbeddingLookupPlugin(const void* data, size_t length)
    : mInitialized(false)
    , mLocalDeviceData(nullptr)
{
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;

    mDenseInScale = read<float>(d);
    mOutScale = read<float>(d);

    mBatchSize = read<int>(d);
    mEmbedHotnessTotal = read<int>(d);
    mEmbeddingSize = read<int>(d);
    mEmbeddingRows = read<size_t>(d);
    mEmbeddingWeightsOnGpuPart = read<float>(d);
    mReducedPrecisionIO = read<int>(d);

    int tableHotnessSize = read<int>(d);
    mTableHotness.resize(tableHotnessSize);
    std::copy((const int*) d, (const int*) d + tableHotnessSize, mTableHotness.data());
    d += NUM_CATEGORICAL_FEATURES * sizeof(int);

    int tableOffsetsSize = read<int>(d);
    mTableOffsets.resize(tableOffsetsSize);
    std::copy((const int*) d, (const int*) d + tableOffsetsSize, mTableOffsets.data());
    d += NUM_CATEGORICAL_FEATURES * sizeof(int);

    int embeddingWeightsFilepathSize = read<int>(d);
    mEmbeddingWeightsFilepath = std::string(d, embeddingWeightsFilepathSize);
    d += MAX_FILEPATH_LENGTH;

    int rowFrequenciesFilepathSize = read<int>(d);
    mRowFrequenciesFilepath = std::string(d, rowFrequenciesFilepathSize);
    d += MAX_FILEPATH_LENGTH;

    int embeddingScalesFilepathSize = read<int>(d);
    mEmbeddingScalesFilepath = std::string(d, embeddingScalesFilepathSize);
    d += MAX_FILEPATH_LENGTH;

    ASSERT(d == a + length);
}

int DLRMv2EmbeddingLookupPlugin::initialize() noexcept
{
    if (!mInitialized)
    {
        std::lock_guard<std::mutex> lck(mSharedDataMutex);
        std::cout << "Starting plugin init..." << std::endl;

        // setup types
        std::cout << "IO Config: " << std::endl;
        size_t embed_elem_size;
        switch (mReducedPrecisionIO)
        {
        case 1: /* fp16 */
            mDenseKernelType = IOType::HALF;
            mEmbedKernelType = IOType::HALF;
            embed_elem_size = sizeof(half);
            std::cout << "\tDense Input:     HALF" << '\n';
            std::cout << "\tSparse Input:    INT32" << '\n';
            std::cout << "\tEmbedding Table: HALF" << '\n';
            std::cout << "\tOutput:          HALF" << std::endl;

            break;

        case 2: /* int8 */
            mDenseKernelType = IOType::HALF;
            mEmbedKernelType = IOType::INT8;
            embed_elem_size = sizeof(int8_t);
            std::cout << "\tDense Input:     HALF" << '\n';
            std::cout << "\tSparse Input:    INT32" << '\n';
            std::cout << "\tEmbedding Table: INT8" << '\n';
            std::cout << "\tOutput:          INT8" << std::endl;

            break;

        case 0:  /* fp32 */
        case -1: /* calibration */
        default:
            mDenseKernelType = IOType::FLOAT;
            mEmbedKernelType = IOType::FLOAT;
            embed_elem_size = sizeof(float);
            std::cout << "\tDense Input:     FLOAT" << '\n';
            std::cout << "\tSparse Input:    INT32" << '\n';
            std::cout << "\tEmbedding Table: FLOAT" << '\n';
            std::cout << "\tOutput:          FLOAT" << std::endl;

            break;
        }

        CUDA_ASSERT(cudaGetDevice(&mDeviceId));
        if (const auto it = mDeviceData.find(mDeviceId); it != mDeviceData.end())
        {
            // the device data was already loaded on this GPU
            mLocalDeviceData = it->second;
            mLocalDeviceData->mCounter++;
        }

        // current GPU doesn't have device data initialized, but some other GPU has.
        else if (mDeviceData.size() > 0)
        {
            // Host data should be initialized by this point.
            ASSERT(mHostData.mCounter > 0);

            const auto otherDeviceData = mDeviceData.begin()->second;
            mLocalDeviceData = new DeviceData();
            mDeviceData.insert(std::make_pair(mDeviceId, mLocalDeviceData));
            mLocalDeviceData->mCounter++;

            // copy index mapping over
            CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mIndexRemapping, mEmbeddingRows * sizeof(int)));
            CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mIndexRemapping, otherDeviceData->mIndexRemapping,
                mEmbeddingRows * sizeof(int), cudaMemcpyDeviceToDevice));

            // copy embedding table over
            CUDA_ASSERT(cudaMalloc(
                &mLocalDeviceData->mEmbeddings, mEmbeddingSize * mHostData.mEmbeddingRowsOnDevice * embed_elem_size));
            CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mEmbeddings, otherDeviceData->mEmbeddings,
                mEmbeddingSize * mHostData.mEmbeddingRowsOnDevice * embed_elem_size, cudaMemcpyDeviceToDevice));

            // copy feature-hotness list over
            CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mTableHotness, NUM_CATEGORICAL_FEATURES * sizeof(int)));
            CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mTableHotness, otherDeviceData->mTableHotness,
                NUM_CATEGORICAL_FEATURES * sizeof(int), cudaMemcpyDeviceToDevice));

            // copy mega table offets over
            CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mTableOffsets, NUM_CATEGORICAL_FEATURES * sizeof(int)));
            CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mTableOffsets, otherDeviceData->mTableOffsets,
                NUM_CATEGORICAL_FEATURES * sizeof(int), cudaMemcpyDeviceToDevice));

            // copy int8 scales over
            if (mReducedPrecisionIO == 2)
            {
                CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mScalesGpu, NUM_CATEGORICAL_FEATURES * sizeof(float)));
                CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mScalesGpu, otherDeviceData->mScalesGpu,
                    NUM_CATEGORICAL_FEATURES * sizeof(float), cudaMemcpyDeviceToDevice));

                // NOTE(vir): mScalesGpuInv[-1] := fp32 dense_input inv-scale
                CUDA_ASSERT(
                    cudaMalloc(&mLocalDeviceData->mScalesGpuInv, (NUM_CATEGORICAL_FEATURES + 1) * sizeof(float)));
                CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mScalesGpuInv, otherDeviceData->mScalesGpuInv,
                    (NUM_CATEGORICAL_FEATURES + 1) * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }

        // No GPUs have device data initialized
        else
        {
            // insert device data entry for current GPU into map.
            mLocalDeviceData = new DeviceData();
            mDeviceData.insert(std::make_pair(mDeviceId, mLocalDeviceData));
            mLocalDeviceData->mCounter++;

            // no host data should be initialized at this point
            ASSERT(mHostData.mCounter == 0);

            // number of embedding rows to keep on GPU and to keep on host.
            const int embeddingRowsOnHost = static_cast<int>(mEmbeddingRows * (1.0F - mEmbeddingWeightsOnGpuPart));
            mHostData.mEmbeddingRowsOnDevice = mEmbeddingRows - embeddingRowsOnHost;

            // remapping positionByActualRowId[i] -> actual row index of embedding row i
            // NOTE(vir): WAR for calibration: dont use rowFrequenciesOpt
            const bool useRowFrequencies
                = !mRowFrequenciesFilepath.empty() and (embeddingRowsOnHost > 0) and (mReducedPrecisionIO != -1);
            std::vector<int> positionByActualRowIdList(mEmbeddingRows);

            {
                // allocate memory for GPU embeddings
                CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mEmbeddings,
                    (size_t) mEmbeddingSize * (size_t) mHostData.mEmbeddingRowsOnDevice * embed_elem_size));

                // allocate memory for host embeddings
                if (embeddingRowsOnHost > 0)
                {
                    CUDA_ASSERT(cudaHostAlloc(&mHostData.mHostEmbeddings,
                        (size_t) mEmbeddingSize * (size_t) embeddingRowsOnHost * embed_elem_size, cudaHostAllocMapped));
                    CUDA_ASSERT(
                        cudaHostGetDevicePointer(&mHostData.mHostEmbeddingsDevicePtr, mHostData.mHostEmbeddings, 0));
                }

                if (useRowFrequencies)
                {
                    std::vector<char> input;
                    npy::NpyFile rowFrequenciesFile(mRowFrequenciesFilepath);
                    rowFrequenciesFile.loadAll(input);

                    // criteo test-dataset
                    constexpr std::array<int, NUM_CATEGORICAL_FEATURES> numRowsPerFeat
                        = {28489200, 35398, 15263, 7363, 20262, 3, 6801, 1341, 61, 33253891, 1711988, 400235, 10, 2209,
                            11938, 154, 4, 960, 14, 39765799, 39998723, 39995760, 586768, 12667, 89, 33};

                    // initialize row id : frequency mapping
                    std::vector<std::pair<size_t, int>> rowIdFreqList(mEmbeddingRows);
                    for (size_t i = 0; i < rowIdFreqList.size(); i += 1)
                        rowIdFreqList[i].first = i;

                    // read data from file
                    size_t fileOffset = 0;
                    for (int feat = 0; feat < NUM_CATEGORICAL_FEATURES; feat += 1)
                    {
                        for (int row = 0; row < numRowsPerFeat[feat]; row += 1)
                        {
                            const int rowId = *((int*) (input.data() + sizeof(int) * (fileOffset + (row * 2) + 0)));
                            const int freq = *((int*) (input.data() + sizeof(int) * (fileOffset + (row * 2) + 1)));
                            const size_t embeddingRowId = mTableOffsets[feat] + rowId;

                            rowIdFreqList[embeddingRowId].first = embeddingRowId;
                            rowIdFreqList[embeddingRowId].second = freq;
                        }

                        fileOffset += numRowsPerFeat[feat] * 2;
                    }

                    // sort by decreasing order of freq
                    std::sort(rowIdFreqList.begin(), rowIdFreqList.end(),
                        [](const std::pair<size_t, int>& x, const std::pair<size_t, int>& y)
                        {
                            // clang-format off
                            if      (x.second > y.second) return true;
                            else if (x.second < y.second) return false;
                            else                          return (x.first < y.first);
                            // clang-format on
                        });

                    // fill index remap
                    for (size_t position_id = 0; position_id < mEmbeddingRows; ++position_id)
                        positionByActualRowIdList[rowIdFreqList[position_id].first] = position_id;

                    CUDA_ASSERT(
                        cudaMalloc(&mLocalDeviceData->mIndexRemapping, sizeof(int) * positionByActualRowIdList.size()));
                    CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mIndexRemapping, positionByActualRowIdList.data(),
                        sizeof(int) * positionByActualRowIdList.size(), cudaMemcpyHostToDevice));
                }

                std::cout << "Loading embedding weights..." << std::endl;

                // load embeddings from file to GPU memory in batches to reduce peak usage of host memory
                npy::NpyFile embeddingWeightsFile(mEmbeddingWeightsFilepath);
                const size_t rowsInBatch = std::min((size_t) mEmbeddingRows, (size_t) 1024 * 1024);

                // staging area to use when using row frequencies opt
                void* embeddingsStaged = nullptr;
                if (useRowFrequencies)
                    CUDA_ASSERT(cudaMalloc(&embeddingsStaged, embed_elem_size * rowsInBatch * mEmbeddingSize));

                // copy data to device and host
                size_t startOffset = 0;
                size_t rowsLoaded = 0;
                while (rowsLoaded < mEmbeddingRows)
                {
                    if (useRowFrequencies)
                    {
                        const size_t rowsToLoad = std::min(rowsInBatch, mEmbeddingRows - rowsLoaded);
                        const size_t bytesToLoad = embed_elem_size * (rowsToLoad * mEmbeddingSize);

                        std::vector<char> input;
                        embeddingWeightsFile.streamBytes(input, startOffset, bytesToLoad);

                        // copy rows to staging area
                        // clang-format off
                        CUDA_ASSERT(cudaMemcpy(
                          embeddingsStaged,
                          input.data(),
                          embed_elem_size * (rowsToLoad * mEmbeddingSize),
                          cudaMemcpyHostToDevice
                        ));
                        // clang-format on

                        // fill in rows on device
                        // clang-format off
                        remapEmbeddingRows(
                          0,
                          (const void*) embeddingsStaged,
                          (void*) mLocalDeviceData->mEmbeddings,
                          ((const int*) mLocalDeviceData->mIndexRemapping) + rowsLoaded,
                          (unsigned int) mEmbeddingSize,
                          (unsigned int) rowsToLoad,
                          (unsigned int) mHostData.mEmbeddingRowsOnDevice,
                          (unsigned int) embed_elem_size
                        );
                        // clang-format on

                        // fill in rows on host
                        for (size_t rowInBatch = 0; rowInBatch < rowsToLoad; ++rowInBatch)
                        {
                            const size_t actualRowPos = positionByActualRowIdList[rowsLoaded + rowInBatch];
                            if (actualRowPos >= mHostData.mEmbeddingRowsOnDevice)
                            {
                                const size_t rowOffset = embed_elem_size * (actualRowPos * mEmbeddingSize);
                                const size_t bytesToLoad = embed_elem_size * mEmbeddingSize;

                                std::vector<char> input;
                                embeddingWeightsFile.streamBytes(input, rowOffset, bytesToLoad);

                                // clang-format off
                                memcpy(
                                  (char*) mHostData.mHostEmbeddings + embed_elem_size * ((actualRowPos - mHostData.mEmbeddingRowsOnDevice) * mEmbeddingSize),
                                  (char*) input.data(),
                                  bytesToLoad
                                );
                                // clang-format on
                            }
                        }

                        startOffset += embed_elem_size * (rowsToLoad * mEmbeddingSize);
                        rowsLoaded += rowsToLoad;
                    }

                    else
                    {
                        // device rows
                        if (rowsLoaded < mHostData.mEmbeddingRowsOnDevice)
                        {
                            const size_t rowsToLoad
                                = std::min(rowsInBatch, mHostData.mEmbeddingRowsOnDevice - rowsLoaded);
                            const size_t bytesToLoad = embed_elem_size * (rowsToLoad * mEmbeddingSize);

                            std::vector<char> input;
                            embeddingWeightsFile.streamBytes(input, startOffset, bytesToLoad);

                            // clang-format off
                            CUDA_ASSERT(cudaMemcpy(
                              (char*) mLocalDeviceData->mEmbeddings + embed_elem_size * (rowsLoaded * mEmbeddingSize),
                              (char*) input.data(),
                              embed_elem_size * (rowsToLoad * mEmbeddingSize),
                              cudaMemcpyHostToDevice
                            ));
                            // clang-format on

                            startOffset += bytesToLoad;
                            rowsLoaded += rowsToLoad;
                        }

                        // host rows
                        else
                        {
                            const size_t rowsToLoad = std::min(rowsInBatch, mEmbeddingRows - rowsLoaded);
                            for (size_t rowInBatch = 0; rowInBatch < rowsToLoad; ++rowInBatch)
                            {
                                const size_t actualRowPos = rowsLoaded + rowInBatch;
                                if (actualRowPos >= mHostData.mEmbeddingRowsOnDevice)
                                {
                                    const size_t rowOffset = embed_elem_size * (actualRowPos * mEmbeddingSize);
                                    const size_t bytesToLoad = embed_elem_size * mEmbeddingSize;

                                    std::vector<char> input;
                                    embeddingWeightsFile.streamBytes(input, rowOffset, bytesToLoad);

                                    // clang-format off
                                    memcpy(
                                      (char*) mHostData.mHostEmbeddings + embed_elem_size * ((actualRowPos - mHostData.mEmbeddingRowsOnDevice) * mEmbeddingSize),
                                      (char*) input.data(),
                                      bytesToLoad
                                    );
                                    // clang-format on
                                }
                            }

                            rowsLoaded += rowsToLoad;
                        }
                    }
                }

                ASSERT(rowsLoaded == mEmbeddingRows);
                if (useRowFrequencies)
                    CUDA_ASSERT(cudaFree(embeddingsStaged));

                std::cout << "Completed loading embedding weights..." << std::endl;
            }

            {
                if (!useRowFrequencies)
                {
                    // rows [0 ... mHostData.mEmbeddingRowsOnDevice] are on device
                    // rest [mHostData.mEmbeddingRowsOnDevice ... ] are on host
                    std::vector<int> indexRemap(mEmbeddingRows);
                    std::iota(indexRemap.begin(), indexRemap.end(), 0);
                    CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mIndexRemapping, mEmbeddingRows * sizeof(int)));
                    CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mIndexRemapping, indexRemap.data(),
                        mEmbeddingRows * sizeof(int), cudaMemcpyHostToDevice));
                }

                CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mTableHotness, mTableHotness.size() * sizeof(int)));
                CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mTableHotness, mTableHotness.data(),
                    mTableHotness.size() * sizeof(int), cudaMemcpyHostToDevice));

                CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mTableOffsets, mTableOffsets.size() * sizeof(int)));
                CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mTableOffsets, mTableOffsets.data(),
                    mTableOffsets.size() * sizeof(int), cudaMemcpyHostToDevice));

                /* load int8 scales */
                if (mReducedPrecisionIO == 2)
                {
                    mScalesGpuInv.resize(NUM_CATEGORICAL_FEATURES + 1);
                    mScalesGpu.resize(NUM_CATEGORICAL_FEATURES);

                    std::vector<char> data;
                    npy::NpyFile embeddingScalesFile(mEmbeddingScalesFilepath);
                    embeddingScalesFile.loadAll(data);

                    for (int i = 0; i < NUM_CATEGORICAL_FEATURES; i += 1)
                    {
                        const float scale = *(float*) (data.data() + i * sizeof(float));
                        mScalesGpu[i] = scale;
                        mScalesGpuInv[i] = 1.0f / mOutScale;
                    }

                    // NOTE(vir): mScalesGpuInv[-1] := fp32 dense_input inv-scale
                    mScalesGpuInv[NUM_CATEGORICAL_FEATURES] = 1.0f / mOutScale;

                    CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mScalesGpu, mScalesGpu.size() * sizeof(float)));
                    CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mScalesGpu, mScalesGpu.data(),
                        mScalesGpu.size() * sizeof(float), cudaMemcpyHostToDevice));

                    CUDA_ASSERT(cudaMalloc(&mLocalDeviceData->mScalesGpuInv, mScalesGpuInv.size() * sizeof(float)));
                    CUDA_ASSERT(cudaMemcpy(mLocalDeviceData->mScalesGpuInv, mScalesGpuInv.data(),
                        mScalesGpuInv.size() * sizeof(float), cudaMemcpyHostToDevice));
                }
            }
        }

        std::cout << "Completed plugin init" << std::endl;
        CUDA_ASSERT(cudaStreamSynchronize(0));

        mInitialized = true;
        mHostData.mCounter++;
    }

    return 0;
}

int DLRMv2EmbeddingLookupPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int res = -1;

    // clang-format off
    res = run_mega_embedding_gather(
      stream,
      (const int*) inputs[1],
      (const int*) mLocalDeviceData->mIndexRemapping,
      (const int*) mLocalDeviceData->mTableHotness,
      (const int*) mLocalDeviceData->mTableOffsets,
      mEmbedKernelType,
      mDenseKernelType,
      inputs[0],
      mLocalDeviceData->mEmbeddings,
      mHostData.mHostEmbeddingsDevicePtr,
      outputs[0],
      mBatchSize,
      mEmbeddingSize,
      NUM_CATEGORICAL_FEATURES,
      mEmbedHotnessTotal,
      mHostData.mEmbeddingRowsOnDevice,
      mReducedPrecisionIO == 2 ? (const float*) mLocalDeviceData->mScalesGpu : nullptr,
      mReducedPrecisionIO == 2 ? (const float*) mLocalDeviceData->mScalesGpuInv : nullptr
    );
    // clang-format on

    ASSERT(res == 0);
    return 0;
}

size_t DLRMv2EmbeddingLookupPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return inputs[0].dims.d[0] * 214 * 4;
}

size_t DLRMv2EmbeddingLookupPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) * 12 + sizeof(size_t) + sizeof(int) * NUM_CATEGORICAL_FEATURES * 2 + MAX_FILEPATH_LENGTH * 3;
}

void DLRMv2EmbeddingLookupPlugin::terminate() noexcept
{

    if (mInitialized)
    {
        std::lock_guard<std::mutex> lck(mSharedDataMutex);

        mHostData.mCounter--;
        if (mHostData.mCounter == 0)
        {
            if (mHostData.mHostEmbeddings != nullptr)
            {
                cudaFreeHost(mHostData.mHostEmbeddings);
                mHostData.mHostEmbeddings = nullptr;
                mHostData.mHostEmbeddingsDevicePtr = nullptr;
            }
        }

        mLocalDeviceData->mCounter--;
        if (mLocalDeviceData->mCounter == 0)
        {
            // check and free cuda-mem ptr
            const auto guardCudaFree = [](void** ptr)
            {
                if (*ptr == nullptr)
                    return;

                cudaFree(*ptr);
                *ptr = nullptr;
            };

            guardCudaFree(&mLocalDeviceData->mEmbeddings);
            guardCudaFree(&mLocalDeviceData->mIndexRemapping);
            guardCudaFree(&mLocalDeviceData->mTableHotness);
            guardCudaFree(&mLocalDeviceData->mTableOffsets);
            guardCudaFree(&mLocalDeviceData->mScalesGpu);
            guardCudaFree(&mLocalDeviceData->mScalesGpuInv);

            delete mLocalDeviceData;
            mLocalDeviceData = nullptr;
            mDeviceData.erase(mDeviceId);
        }

        mInitialized = false;
    }
}

void DLRMv2EmbeddingLookupPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    ASSERT(in && nbInputs == 2);
    ASSERT(out && nbOutputs == 1);

    ASSERT(in[0].desc.dims.nbDims == 2); // dense features
    ASSERT(in[1].desc.dims.nbDims == 2); // sparse features

    // NOTE(vir): sprase in is int32 always, so no need for scale (1.0f)
    mDenseInScale = in[0].desc.scale;
    mOutScale = out[0].desc.scale;
}

bool DLRMv2EmbeddingLookupPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);

    // NOTE(vir):
    //
    // ----- support matrix -----
    // |       mReducedPrecisionIO | dense | sparse | embeddings | status              |
    // |-------------------------------------------------------------------------------|
    // |                         0 | FLOAT | INT32  | FLOAT      | PASS                |
    // |                         1 | HALF  | INT32  | HALF       | PASS                |
    // |                         2 | FLOAT | INT32  | INT8       | PASS                |
    // |          [in use]       2 | HALF  | INT32  | INT8       | PASS                |
    // |                         2 | INT8  | INT32  | INT8       | FAIL                |
    // |          [calibration] -1 | FLOAT | INT32  | FLOAT      | PASS                |

    switch (pos)
    {
    /* dense */
    case 0:
        // clang-format off
        return (((inOut[pos].format == TensorFormat::kLINEAR) && (inOut[pos].type == DataType::kFLOAT) && (mReducedPrecisionIO ==  0))
             or ((inOut[pos].format == TensorFormat::kLINEAR) && (inOut[pos].type == DataType::kHALF)  && (mReducedPrecisionIO ==  1))
             or ((inOut[pos].format == TensorFormat::kLINEAR) && (inOut[pos].type == DataType::kHALF)  && (mReducedPrecisionIO ==  2))
             or ((inOut[pos].format == TensorFormat::kLINEAR) && (inOut[pos].type == DataType::kFLOAT) && (mReducedPrecisionIO == -1)));
        break;
        // clang-format on

    /* sparse */
    case 1: return ((inOut[pos].format == TensorFormat::kLINEAR) && (inOut[pos].type == DataType::kINT32)); break;

    /* output */
    case 2:
        // clang-format off
        return (((inOut[pos].format == TensorFormat::kLINEAR) && (inOut[pos].type == DataType::kFLOAT) && (mReducedPrecisionIO ==  0))
             or ((inOut[pos].format == TensorFormat::kLINEAR) && (inOut[pos].type == DataType::kHALF)  && (mReducedPrecisionIO ==  1))
             or ((inOut[pos].format == TensorFormat::kLINEAR) && (inOut[pos].type == DataType::kINT8)  && (mReducedPrecisionIO ==  2))
             or ((inOut[pos].format == TensorFormat::kLINEAR) && (inOut[pos].type == DataType::kFLOAT) && (mReducedPrecisionIO == -1)));
        break;
        // clang-format on
    }

    return false;
}

const char* DLRMv2EmbeddingLookupPlugin::getPluginType() const noexcept
{
    return DLRMv2_INTERACTIONS_PLUGIN_NAME;
}

const char* DLRMv2EmbeddingLookupPlugin::getPluginVersion() const noexcept
{
    return DLRMv2_INTERACTIONS_PLUGIN_VERSION;
}

int DLRMv2EmbeddingLookupPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType DLRMv2EmbeddingLookupPlugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    ASSERT(nbInputs == 2);
    ASSERT(index == 0);

    return inputTypes[0];
}

DimsExprs DLRMv2EmbeddingLookupPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    ASSERT(outputIndex == 0);
    ASSERT(nbInputs == 2);

    return DimsExprs{
        2,
        {
            inputs[0].d[0], exprBuilder.constant((NUM_CATEGORICAL_FEATURES + 1) * mEmbeddingSize)
            // exprBuilder.constant(1),
            // exprBuilder.constant(1),
        },
    };
}

IPluginV2DynamicExt* DLRMv2EmbeddingLookupPlugin::clone() const noexcept
{
    IPluginV2DynamicExt* plugin = new DLRMv2EmbeddingLookupPlugin(*this);
    return plugin;
}

void DLRMv2EmbeddingLookupPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    const char* a = d;

    write(d, mDenseInScale);
    write(d, mOutScale);

    write(d, mBatchSize);
    write(d, mEmbedHotnessTotal);
    write(d, mEmbeddingSize);
    write(d, mEmbeddingRows);
    write(d, mEmbeddingWeightsOnGpuPart);
    write(d, mReducedPrecisionIO);

    ASSERT(mTableHotness.size() == NUM_CATEGORICAL_FEATURES);
    write(d, (int) (mTableHotness.size()));
    std::copy(mTableHotness.data(), mTableHotness.data() + mTableHotness.size(), (int*) d);
    d += NUM_CATEGORICAL_FEATURES * sizeof(int);

    ASSERT(mTableOffsets.size() == NUM_CATEGORICAL_FEATURES);
    write(d, (int) (mTableOffsets.size()));
    std::copy(mTableOffsets.data(), mTableOffsets.data() + mTableOffsets.size(), (int*) d);
    d += NUM_CATEGORICAL_FEATURES * sizeof(int);

    ASSERT(mEmbeddingWeightsFilepath.size() <= MAX_FILEPATH_LENGTH);
    write(d, (int) (mEmbeddingWeightsFilepath.size()));
    std::copy(mEmbeddingWeightsFilepath.data(), mEmbeddingWeightsFilepath.data() + mEmbeddingWeightsFilepath.size(), d);
    d += MAX_FILEPATH_LENGTH;

    ASSERT(mRowFrequenciesFilepath.size() <= MAX_FILEPATH_LENGTH);
    write(d, (int) (mRowFrequenciesFilepath.size()));
    std::copy(mRowFrequenciesFilepath.data(), mRowFrequenciesFilepath.data() + mRowFrequenciesFilepath.size(), d);
    d += MAX_FILEPATH_LENGTH;

    ASSERT(mEmbeddingScalesFilepath.size() <= MAX_FILEPATH_LENGTH);
    write(d, (int) (mEmbeddingScalesFilepath.size()));
    std::copy(mEmbeddingScalesFilepath.data(), mEmbeddingScalesFilepath.data() + mEmbeddingScalesFilepath.size(), d);
    d += MAX_FILEPATH_LENGTH;

    ASSERT(d == a + getSerializationSize());
}

void DLRMv2EmbeddingLookupPlugin::destroy() noexcept
{
    delete this;
}

// {{{ DLRMv2EmbeddingLookupPluginCreator
DLRMv2EmbeddingLookupPluginCreator::DLRMv2EmbeddingLookupPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("embeddingSize", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("embeddingRows", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("embeddingWeightsOnGpuPart", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("tableHotness", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("tableOffsets", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("batchSize", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("embedHotnessTotal", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("embeddingWeightsFilepath", nullptr, PluginFieldType::kCHAR));
    mPluginAttributes.emplace_back(PluginField("rowFrequenciesFilepath", nullptr, PluginFieldType::kCHAR));
    mPluginAttributes.emplace_back(PluginField("embeddingScalesFilepath", nullptr, PluginFieldType::kCHAR));
    mPluginAttributes.emplace_back(PluginField("reducedPrecisionIO", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

IPluginV2* DLRMv2EmbeddingLookupPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;

    int embeddingSize = 0;
    int embeddingRows = 0;
    int batchSize = 0;
    int embedHotnessTotal = 0;
    float embeddingWeightsOnGpuPart = 0.0f;
    int reducedPrecisionIO = 0;
    std::vector<int> tableHotness;
    std::vector<int> tableOffsets;
    std::string embeddingWeightsFilepath;
    std::string rowFrequenciesFilepath;
    std::string embeddingScalesFilepath;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "embeddingRows"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            embeddingRows = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "embeddingSize"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            embeddingSize = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "embeddingWeightsOnGpuPart"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            embeddingWeightsOnGpuPart = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "tableHotness"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            tableHotness.resize(fields[i].length);
            std::copy(
                (const int*) (fields[i].data), (const int*) (fields[i].data) + fields[i].length, tableHotness.data());
        }
        else if (!strcmp(attrName, "tableOffsets"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            tableOffsets.resize(fields[i].length);
            std::copy(
                (const int*) (fields[i].data), (const int*) (fields[i].data) + fields[i].length, tableOffsets.data());
        }
        else if (!strcmp(attrName, "batchSize"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            batchSize = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "embedHotnessTotal"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            embedHotnessTotal = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "embeddingWeightsFilepath"))
        {
            ASSERT(fields[i].type == PluginFieldType::kCHAR);
            embeddingWeightsFilepath = std::string((const char*) (fields[i].data), fields[i].length);
        }
        else if (!strcmp(attrName, "rowFrequenciesFilepath"))
        {
            ASSERT(fields[i].type == PluginFieldType::kCHAR);
            rowFrequenciesFilepath = std::string((const char*) (fields[i].data), fields[i].length);
        }
        else if (!strcmp(attrName, "embeddingScalesFilepath"))
        {
            ASSERT(fields[i].type == PluginFieldType::kCHAR);
            embeddingScalesFilepath = std::string((const char*) (fields[i].data), fields[i].length);
        }
        else if (!strcmp(attrName, "reducedPrecisionIO"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            reducedPrecisionIO = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
    }

    // clang-format off
    return new DLRMv2EmbeddingLookupPlugin(
      embeddingSize,
      embeddingRows,
      embeddingWeightsOnGpuPart,
      tableHotness,
      tableOffsets,
      batchSize,
      embedHotnessTotal,
      embeddingWeightsFilepath,
      rowFrequenciesFilepath,
      embeddingScalesFilepath,
      reducedPrecisionIO
    );
    // clang-format on
}

const char* DLRMv2EmbeddingLookupPluginCreator::getPluginName() const noexcept
{
    return DLRMv2_INTERACTIONS_PLUGIN_NAME;
}

const char* DLRMv2EmbeddingLookupPluginCreator::getPluginVersion() const noexcept
{
    return DLRMv2_INTERACTIONS_PLUGIN_VERSION;
}

IPluginV2* DLRMv2EmbeddingLookupPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new DLRMv2EmbeddingLookupPlugin(serialData, serialLength);
}
// }}}
