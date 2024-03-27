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

#ifndef _RETINANET_NMS_PLUGIN_H_
#define _RETINANET_NMS_PLUGIN_H_

// Basic cpp headers
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Cuda math constants header
#include "math_constants.h"

// TensorRT Headers based on safety flag
#ifdef SAFE
#include "NvInferRuntimeCommon.h"
#include "NvInferSafeRuntime.h"
#else
#include "NvInfer.h"
#endif
#include "NvInferVersion.h"

// PVA Kernel Specific headers
#include "RetinaNetNMSAppSync.hpp"
//!
//! \file retinaNetNMSPVAPlugin.h
//!
//! This is the top-level API file for RetinaNet NMS PVA plugin
//!

typedef enum
{
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

//!
//! \brief The TensorRT API version 1 namespace.
//!
namespace nvinfer1
{

//!
//! \brief Sample level namespace
//!
namespace retinanetnms
{

//!
//! \brief Custom RetinaNetNMSPlugin implementation class
//!
class RetinaNetNMSPlugin : public IPluginV2IOExt
{
public:
    //!
    //! \brief Constructor for RetinaNetNMSPlugin. Called by RetinaNetNMSPluginCreator class
    //!
    //! \param fc Stores PluginField Collection
    //!
    RetinaNetNMSPlugin(PluginFieldCollection const& fc) noexcept;

    //!
    //! \brief Constructor used during deserialization
    //!
    //! \param name Plugin name
    //! \param data Serialized plugin data
    //! \param length Length of serialized data
    //!
    RetinaNetNMSPlugin(const std::string name, const void* serialData, size_t serialLength);

    //!
    //! \brief Constructor
    //!
    //! It makes no sense to construct RetinaNetNMSPlugin without arguments.
    //!
    RetinaNetNMSPlugin() = delete;

    //!
    //! \brief Destructor override as default
    //!
    ~RetinaNetNMSPlugin() override = default;

    //!
    //! \brief Number of outputs from the plugin class
    //!
    int getNbOutputs() const noexcept override;

    //!
    //! \brief Output dimensions
    //!
    //! \param index Output tensor index
    //! \param inputs Input tensor dimensions
    //! \param nbInputDims Number of input tensors
    //!
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputs) noexcept override;

    //!
    //! \brief Initialize called by TensorRT builder and executor
    //!
    int initialize() noexcept override;

    //!
    //! \brief Terminate called by TensorRT builder and executor
    //!
    void terminate() noexcept override;

    //!
    //! \brief Get workspace size required by plugin
    //!
    //! \param maxBatchSize Maximum batch size configured
    //!
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

    //!
    //! \brief Plugin execution API
    //!
    //! \param batchSize Batch size
    //! \param inputs Input tensor to the plugin layer
    //! \param outputs Output tensor from the plugin layer
    //! \param workspace Workspace
    //! \param stream CUDA stream used by the TensorRT executor
    //!
#if NV_TENSORRT_MAJOR == 7
    // In TRT 7.x use the following
    int32_t enqueue(
        int32_t batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
#else
    // For other versions (Above and below 7.x)
    int32_t enqueue(int32_t batchSize, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;
#endif

    //!
    //! \brief Compute serialized plugin size
    //!
    size_t getSerializationSize() const noexcept override;

    //!
    //! \brief Performs serialization of RetinaNetNMSPlugin class
    //!
    //! \param buffer Destination buffer to write serialized plugin class
    //!
    void serialize(void* buffer) const noexcept override;

    //!
    //! \brief Return true if plugin supports the format and datatype for the input/output
    //!
    //! \param pos Index for input/output
    //! \param inOut Input/output tensor information
    //! \param nbInputs Number of input
    //! \param nbInputs Number of output
    //!
    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept override;

    //!
    //! \brief Implementation method for returning plugin type
    //!
    const char* getPluginType() const noexcept override;

    //!
    //! \brief Implementation method for returning plugin version
    //!
    const char* getPluginVersion() const noexcept override;

    //!
    //! \brief Destructor for plugin class
    //!
    void destroy() noexcept override;

    //!
    //! \brief Returns the cloned copy of plugin
    //!
    IPluginV2IOExt* clone() const noexcept override;

    //!
    //! \brief Implementation for setting plugin namespace
    //!
    //! \param pluginNamespace Configures plugin namespace for RetinaNetNMSPlugin
    //!
    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    //!
    //! \brief Implementation method for returning plugin namespace
    //!
    const char* getPluginNamespace() const noexcept override;

    //!
    //! \brief Get output tensor's datatype
    //!
    //! \param index Output tensor index
    //! \param inputTypes Input Tensor data type
    //! \param nbInputs Number of inputs
    //!
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;

    //!
    //! \brief Configuration of plugin called by TensorRT builder
    //! \param in Input tensor description
    //! \param nbInput Number of input tensors
    //! \param out Output tensor description
    //! \param nbOutput Number of output tensors
    //!
    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept override;

    //!
    //! \brief Checks if output is broadcasted across batch dimension
    //!
    //! \param outputIndex Output tensor index
    //! \param inputIsBroadcasted The ith element is true if the tensor for the ith input is broadcast across a batch
    //! \param nbInputs Number of inputs
    //!
    bool isOutputBroadcastAcrossBatch(
        int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

    //!
    //! \brief Checks if plugin can broadcast input across batch dimensions
    //!
    //! \param inputIndex Input tensor index
    //!
    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

private:
    //!
    //! \brief Utiliy function to write into buffer
    //!
    //! \param buffer buffer to write with given value
    //! \param val Value to write
    //!
    //! \return status
    //!
    template <typename T>
    //! \brief Utility to write
    bool write(char* buffer, const T* val) const noexcept
    {
        bool retVal = false;
        if (val != nullptr)
        {
            *reinterpret_cast<T*>(buffer) = *val;
            retVal = true;
        }
        buffer += sizeof(T);
        return retVal;
    }

    //!
    //! \brief Utiliy function to read from the given buffer
    //!
    //! \param buffer buffer to read from
    //!
    //! \return value read from the buffer
    //!
    template <typename T>
    //! \brief Utility to read
    auto read(const char*& buffer) const noexcept -> T
    {
        T val = *reinterpret_cast<T const*>(buffer);
        buffer = &(buffer[sizeof(T)]);
        return val;
    }

    //!
    //! \brief Plugin namespace
    //!
    const char* mPluginNamespace;
    const std::string mName;

    //!
    //! \brief Params for the RetinaNetNMS Plugin
    //!
    RetinaNetNMSPluginParams mParams;

    RetinaNetNMSAppSync* mRetinaNetNMSAppSyncObj;

}; // class RetinaNetNMSPlugin

//!
//! \brief RetinaNetNMSPlugin Creator for RetinaNetNMSPlugin class
//!
class RetinaNetNMSPluginCreator : public IPluginCreator
{
public:
    //!
    //! \brief Constructor
    //!
    RetinaNetNMSPluginCreator();

    //!
    //! \brief Destructor override as default
    //!
    ~RetinaNetNMSPluginCreator() override = default;

    //!
    //! \brief Get plugin name
    //!
    const char* getPluginName() const noexcept override;

    //!
    //! \brief Get plugin version
    //!
    const char* getPluginVersion() const noexcept override;

    //!
    //! \brief Get plugin field names
    //!
    const PluginFieldCollection* getFieldNames() noexcept override;

    //!
    //! \brief Instantiates RetinaNetNMSPlugin class
    //!
    //! \param name Plugin name
    //! \param fc PluginField collection
    //!
    IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    //!
    //! \brief Deserialze a serialized plugin
    //!
    //! \param name Plugin name
    //! \param serialData Serialized plugin data
    //! \param serialLength Length of serialized data
    //!
    IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    //!
    //! \brief Sets plugin namespace
    //!
    //! \param pluginNamespace Plugin namespace to set
    //!
    void setPluginNamespace(const char* libNamespace) noexcept override;

    //!
    //! \brief Gets plugin namespace
    //!
    const char* getPluginNamespace() const noexcept override;

private:
    //!
    //! \brief Plugin field names
    //!
    static PluginFieldCollection mFC;

    //!
    //! \brief Plugin field attributes
    //!
    static std::vector<PluginField> mPluginAttributes;

    //!
    //! \brief Plugin namespace
    //!
    std::string mNamespace;
}; // class RetinaNetNMSPluginCreator

// Register custom plugin creator for RetinaNet NMS
REGISTER_TENSORRT_PLUGIN(RetinaNetNMSPluginCreator);

} // namespace retinanetnms
} // namespace nvinfer1

#endif // _RETINANET_NMS_PLUGIN_H_
