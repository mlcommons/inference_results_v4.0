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
#pragma once

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "NvInferPlugin.h"

#include "embedding_gather_launch.hpp"

namespace nvinfer1::plugin
{

class DLRMv2EmbeddingLookupPlugin : public IPluginV2DynamicExt
{
public:
    DLRMv2EmbeddingLookupPlugin(const int embeddingSize, const int embeddingRows, const float embeddingWeightsOnGpuPart,
        const std::vector<int>& tableHotness, const std::vector<int>& tableOffsets, const int batchSize,
        const int embedHotnessTotal, const std::string& embeddingWeightsFilepath,
        const std::string& rowFrequenciesFilepath, const std::string& embeddingScalesFilepath,
        const int reducedPrecisionIO);
    DLRMv2EmbeddingLookupPlugin(const void* data, size_t length);

    DLRMv2EmbeddingLookupPlugin(const DLRMv2EmbeddingLookupPlugin&) = default;
    ~DLRMv2EmbeddingLookupPlugin() override = default;

    // IPluginV2Ext
    int initialize() noexcept override;
    void destroy() noexcept override;
    void terminate() noexcept override;

    IPluginV2DynamicExt* clone() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    size_t getSerializationSize() const noexcept override;
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;

    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out,
        int nbOutputs) noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }
    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    // Kernel data on host
    struct HostData
    {
        HostData()
            : mHostEmbeddings(nullptr)
            , mHostEmbeddingsDevicePtr(nullptr)
            , mEmbeddingRowsOnDevice(0)
            , mCounter(0)
        {
        }

        void* mHostEmbeddings;
        void* mHostEmbeddingsDevicePtr;
        size_t mEmbeddingRowsOnDevice;
        int mCounter;
    };

    // Embedding and related data stored on a single GPU
    struct DeviceData
    {
        DeviceData()
            : mEmbeddings(nullptr)
            , mIndexRemapping(nullptr)
            , mTableHotness(nullptr)
            , mTableOffsets(nullptr)
            , mScalesGpu(nullptr)
            , mScalesGpuInv(nullptr)
            , mCounter(0)
        {
        }

        void* mEmbeddings;     // All rows of embedding values.
        void* mIndexRemapping; // Used to figure out if row is on device or on host
        void* mTableHotness;   // Hotness of each embedding table
        void* mTableOffsets;   // Offsets of each table's rows in global row list.
        void* mScalesGpu;      // scale in per embedding table
        void* mScalesGpuInv;   // scale out per embedding table
        int mCounter;          // Ref count
    };

private:
    std::string mNamespace;

    bool mInitialized;
    int mDeviceId;

    static std::mutex mSharedDataMutex;
    static HostData mHostData;
    static std::map<int, DeviceData*> mDeviceData; // Map of GPU device to its local embedding data.
    DeviceData* mLocalDeviceData;

    float mDenseInScale;
    float mOutScale;
    IOType mEmbedKernelType; // kernel precision of embedding table
    IOType mDenseKernelType; // kernel precision of dense inputs

    int mEmbeddingSize;               // Number of embedding values in one row
    size_t mEmbeddingRows;            // Number of embedding rows
    float mEmbeddingWeightsOnGpuPart; // Fraction of embedding rows to keep on GPU.
    std::vector<int> mTableHotness;   // Hotness for each feature
    std::vector<int> mTableOffsets;   // Index offsets into global rows for each table.
    std::vector<float> mScalesGpu;    // Scale In per embedding table
    std::vector<float> mScalesGpuInv; // scale Out per embedding table
    int mBatchSize;                   // Batch Size
    int mEmbedHotnessTotal;           // Sum of Hotness of all features
    std::string mEmbeddingWeightsFilepath;
    std::string mRowFrequenciesFilepath;
    std::string mEmbeddingScalesFilepath;
    int mReducedPrecisionIO; // -1: calibration, 0: fp32, 1: fp16, 2 : int8
};

class DLRMv2EmbeddingLookupPluginCreator : public IPluginCreator
{
public:
    DLRMv2EmbeddingLookupPluginCreator();
    ~DLRMv2EmbeddingLookupPluginCreator() noexcept override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }
    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace nvinfer1::plugin
