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

// Plugin Debug Enable / Disable MACRO
#ifndef DEBUG_RETINANET_NMS_PLUGIN
#define DEBUG_RETINANET_NMS_PLUGIN 0 // set debug mode, if you want to print various values and debug, set to 1
#endif

// Whether to read scales from PluginTensorDesc in configurePlugin
// When set to 1, will read dequantization scales from PluginTensorDesc
// When set to 0, will use default scales set in RetinaNetNMSPlugin::configurePlugin
#ifndef READ_SCALES_FROM_DESC
#define READ_SCALES_FROM_DESC 1
#endif

#include "retinanetNMSPVAPlugin.h"
#include "plugin_helper.h"

//!
//! Declaration of CUDA kernels Initializer
//!

// nvinfer namespace
namespace nvinfer1
{
// plugin's name space
namespace retinanetnms
{
// standalone namespace for plugin attributes (name & version)
namespace
{
// Version and name of the plugin to be registered
static const char* RETINANET_NMS_PLUGIN_VERSION{"1"};
static const char* RETINANET_NMS_PLUGIN_NAME{"RetinaNetNMSPVATRT"};
} // namespace

// Static class fields initialization
PluginFieldCollection RetinaNetNMSPluginCreator::mFC{};
std::vector<PluginField> RetinaNetNMSPluginCreator::mPluginAttributes{};

// Div up function
uint32_t div_up(uint32_t m, uint32_t n)
{
    return (m + n - 1) / n;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// Plugin Methods ///////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor which takes plugin field collection as input
RetinaNetNMSPlugin::RetinaNetNMSPlugin(const nvinfer1::PluginFieldCollection& fc) noexcept
{

#if DEBUG_RETINANET_NMS_PLUGIN
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: Inside RetinaNetNMSPlugin::RetinaNetNMSPlugin\n");
#endif

    const PluginField* fields = fc.fields;

    for (int ii = 0; ii < fc.nbFields; ++ii)
    {

        const char* attrName = fields[ii].name;

        if (!strcmp(attrName, "nmsThreshold"))
        {
            TRT_ASSERT(fields[ii].type == PluginFieldType::kFLOAT32);
            mParams.nms_threshold = static_cast<float>(*(static_cast<const float*>(fields[ii].data)));
#if DEBUG_RETINANET_NMS_PLUGIN
            printf("[ RetinaNetNMSPVATRT ]....: mParams.nms_threshold set to %.6f\n", mParams.nms_threshold);
#endif
        }

        else if (!strcmp(attrName, "score0Thresh"))
        {
            TRT_ASSERT(fields[ii].type == PluginFieldType::kFLOAT32);
            mParams.score_thresholds[0] = static_cast<float>(*(static_cast<const float*>(fields[ii].data)));
#if DEBUG_RETINANET_NMS_PLUGIN
            printf(
                "[ RetinaNetNMSPVATRT ]....: mParams.score_thresholds[0] set to %.6f\n", mParams.score_thresholds[0]);
#endif
        }

        else if (!strcmp(attrName, "score1Thresh"))
        {
            TRT_ASSERT(fields[ii].type == PluginFieldType::kFLOAT32);
            mParams.score_thresholds[1] = static_cast<float>(*(static_cast<const float*>(fields[ii].data)));
#if DEBUG_RETINANET_NMS_PLUGIN
            printf(
                "[ RetinaNetNMSPVATRT ]....: mParams.score_thresholds[1] set to %.6f\n", mParams.score_thresholds[1]);
#endif
        }

        else if (!strcmp(attrName, "score2Thresh"))
        {
            TRT_ASSERT(fields[ii].type == PluginFieldType::kFLOAT32);
            mParams.score_thresholds[2] = static_cast<float>(*(static_cast<const float*>(fields[ii].data)));
#if DEBUG_RETINANET_NMS_PLUGIN
            printf(
                "[ RetinaNetNMSPVATRT ]....: mParams.score_thresholds[2] set to %.6f\n", mParams.score_thresholds[2]);
#endif
        }

        else if (!strcmp(attrName, "score3Thresh"))
        {
            TRT_ASSERT(fields[ii].type == PluginFieldType::kFLOAT32);
            mParams.score_thresholds[3] = static_cast<float>(*(static_cast<const float*>(fields[ii].data)));
#if DEBUG_RETINANET_NMS_PLUGIN
            printf(
                "[ RetinaNetNMSPVATRT ]....: mParams.score_thresholds[3] set to %.6f\n", mParams.score_thresholds[3]);
#endif
        }

        else if (!strcmp(attrName, "score4Thresh"))
        {
            TRT_ASSERT(fields[ii].type == PluginFieldType::kFLOAT32);
            mParams.score_thresholds[4] = static_cast<float>(*(static_cast<const float*>(fields[ii].data)));
#if DEBUG_RETINANET_NMS_PLUGIN
            printf(
                "[ RetinaNetNMSPVATRT ]....: mParams.score_thresholds[4] set to %.6f\n", mParams.score_thresholds[4]);
#endif
        }
    } // iterate over fields / node attributes
}

// Constructor which takes serialized buffer as input
RetinaNetNMSPlugin::RetinaNetNMSPlugin(const std::string name, const void* data, size_t length)
    : mName(name)
{

    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mParams = read<RetinaNetNMSPluginParams>(d);
#if DEBUG_RETINANET_NMS_PLUGIN
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: Inside RetinaNetNMSPlugin::RetinaNetNMSPlugin with Serialized Buffer\n");
    printf("[ RetinaNetNMSPVATRT ]....: nms_threshold{ %f }\n", mParams.nms_threshold);
    printf("[ RetinaNetNMSPVATRT ]....: Score thresholds{ %f, %f, %f, %f, %f }\n", mParams.score_thresholds[0],
        mParams.score_thresholds[1], mParams.score_thresholds[2], mParams.score_thresholds[3],
        mParams.score_thresholds[4]);
#endif
    TRT_ASSERT(d == (a + length));
}

// Returns number of outputs of the plugin == 1
int RetinaNetNMSPlugin::getNbOutputs() const noexcept
{
    return 1;
}

// Returns output dimensions based on Input Dimensions
Dims RetinaNetNMSPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputs) noexcept
{

#if DEBUG_RETINANET_NMS_PLUGIN // Set to 1 at top of the file to print input dimensions
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: Inside RetinaNetNMSPlugin::getOutputDimensions\n");
    int32_t nbInputIter = 0, dimsIter = 0;
    for (nbInputIter = 0; nbInputIter < nbInputs; nbInputIter++)
    {
        printf("[ RetinaNetNMSPVATRT ]....: Dims Input %2d  { ", nbInputIter);
        for (dimsIter = 0; dimsIter < inputs[nbInputIter].nbDims; dimsIter++)
        {
            printf("%4d ", inputs[nbInputIter].d[dimsIter]);
        }
        printf("}\n");
    }
#endif

    // Check index of output
    TRT_ASSERT(index == 0);
    // Check number of inputs
    TRT_ASSERT(nbInputs == 11);
    // Assert dims of inputs

    // Variable for Output Dims
    Dims outDims;
    outDims.nbDims = 3;
    outDims.d[0] = 1;
    outDims.d[1] = 1;
    outDims.d[2] = 7001;

#if DEBUG_RETINANET_NMS_PLUGIN // Set to 1 to print output dimensions
    int32_t outIter = 0;
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: Dims Output %d  { ", (int32_t) 1);
    for (outIter = 0; outIter < outDims.nbDims; outIter++)
    {
        printf("%4d ", outDims.d[outIter]);
    }
    printf("}\n");
#endif

    return (outDims);
}

int RetinaNetNMSPlugin::initialize() noexcept
{
#if DEBUG_RETINANET_NMS_PLUGIN
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: initialize Plugin\n");
#endif

    mRetinaNetNMSAppSyncObj = new RetinaNetNMSAppSync();
#if DEBUG_RETINANET_NMS_PLUGIN
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: mRetinaNetNMSAppSyncObj created!!\n");
#endif

    mRetinaNetNMSAppSyncObj->initialize(mParams);
#if DEBUG_RETINANET_NMS_PLUGIN
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: mRetinaNetNMSAppSyncObj initialized \n");
#endif

    return STATUS_SUCCESS;
}

void RetinaNetNMSPlugin::terminate() noexcept
{
#if DEBUG_RETINANET_NMS_PLUGIN
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: terminate Plugin\n");
#endif

    mRetinaNetNMSAppSyncObj->terminate();
#if DEBUG_RETINANET_NMS_PLUGIN
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: mRetinaNetNMSAppSyncObj terminated\n");
#endif

    delete mRetinaNetNMSAppSyncObj;
#if DEBUG_RETINANET_NMS_PLUGIN
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: mRetinaNetNMSAppSyncObj deleted\n");
#endif
}

size_t RetinaNetNMSPlugin::getWorkspaceSize(int maxBatchSize) const noexcept
{
    // Return workspace size
    return (0);
}

// Return serialization size
size_t RetinaNetNMSPlugin::getSerializationSize() const noexcept
{

    size_t serializationSize{0UL};
    try
    {
        serializationSize = helper::addPositiveRangeCheck(serializationSize, sizeof(RetinaNetNMSPluginParams));
    }
    catch (std::runtime_error& e)
    {
        printf("--------------------------:\n");
        printf("[ RetinaNetNMSPVATRT ]....: Exception caught while computing serialization size\n");
    }
    return serializationSize;
}

// Serialize the parameters to the plugin
void RetinaNetNMSPlugin::serialize(void* buffer) const noexcept
{
#if DEBUG_RETINANET_NMS_PLUGIN
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: Serializing Plugin\n");
#endif

    try
    {
        char* d = reinterpret_cast<char*>(buffer);
        const char* a = d;
        *reinterpret_cast<RetinaNetNMSPluginParams*>(d) = mParams;
        d += sizeof(mParams);
        TRT_ASSERT(d == a + getSerializationSize());

#if DEBUG_RETINANET_NMS_PLUGIN
        printf("--------------------------:\n");
        printf("[ RetinaNetNMSPVATRT ]....: nms_threshold{ %f } serialized\n", mParams.nms_threshold);
        printf("[ RetinaNetNMSPVATRT ]....: Score thresholds{ %f, %f, %f, %f, %f }\n", mParams.score_thresholds[0],
            mParams.score_thresholds[1], mParams.score_thresholds[2], mParams.score_thresholds[3],
            mParams.score_thresholds[4]);
#endif
    }
    catch (std::exception& e)
    {
        printf("--------------------------:\n");
        printf("[ RetinaNetNMSPVATRT ]....: Exception caught during serialization\n");
    }
}

// Configure Plugin Data Members using parameters from tensor descriptors
void RetinaNetNMSPlugin::configurePlugin(
    PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out, int32_t nbOutput) noexcept
{
#if DEBUG_RETINANET_NMS_PLUGIN
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: configurePlugin\n");
#endif

    TRT_ASSERT((in != nullptr) && (nbInput == 11));
    TRT_ASSERT((out != nullptr) && (nbOutput == 1));

    TRT_ASSERT(in[0].type == in[1].type);
    TRT_ASSERT(in[0].type == in[2].type);
    TRT_ASSERT(in[0].type == in[3].type);
    TRT_ASSERT(in[0].type == in[4].type);
    TRT_ASSERT(in[0].type == in[5].type);
    TRT_ASSERT(in[0].type == in[6].type);
    TRT_ASSERT(in[0].type == in[7].type);
    TRT_ASSERT(in[0].type == in[8].type);
    TRT_ASSERT(in[0].type == in[9].type);

    TRT_ASSERT(in[10].type == out[0].type);

#if DEBUG_RETINANET_NMS_PLUGIN // Set to 1 at top of the file to print input dimensions
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: RetinaNetNMSPlugin::configurePlugin\n");
    int32_t nbInputIter = 0, dimsIter = 0;
    for (nbInputIter = 0; nbInputIter < nbInput; nbInputIter++)
    {
        printf("--------------------------:\n");
        printf("[ RetinaNetNMSPVATRT ]....: Input %2d  { ", nbInputIter);
        for (dimsIter = 0; dimsIter < in[nbInputIter].dims.nbDims; dimsIter++)
        {
            printf("%4d ", in[nbInputIter].dims.d[dimsIter]);
        }
        printf("}\n");
    }

    int32_t nbOutputIter = 0;
    dimsIter = 0;
    for (nbOutputIter = 0; nbOutputIter < nbOutput; nbOutputIter++)
    {
        printf("[ RetinaNetNMSPVATRT ]....: Output %d  { ", nbOutputIter);
        for (dimsIter = 0; dimsIter < out[nbOutputIter].dims.nbDims; dimsIter++)
        {
            printf("%4d ", out[nbOutputIter].dims.d[dimsIter]);
        }
        printf("}\n");
    }
#endif
#if READ_SCALES_FROM_DESC

    // Configure Scales for the Int8 Tensors here
    mParams.score_scales[0] = in[0].scale;
    mParams.score_scales[1] = in[1].scale;
    mParams.score_scales[2] = in[2].scale;
    mParams.score_scales[3] = in[3].scale;
    mParams.score_scales[4] = in[4].scale;
    mParams.box_scales[0] = in[5].scale;
    mParams.box_scales[1] = in[6].scale;
    mParams.box_scales[2] = in[7].scale;
    mParams.box_scales[3] = in[8].scale;
    mParams.box_scales[4] = in[9].scale;

#else // Scales Set Manually by User
    mParams.score_scales[0] = 0.157048374414f;
    mParams.score_scales[1] = 0.153431400657f;
    mParams.score_scales[2] = 0.170486032963f;
    mParams.score_scales[3] = 0.201211079955f;
    mParams.score_scales[4] = 0.192310780287f;
    mParams.box_scales[0] = 0.0165229439735f;
    mParams.box_scales[1] = 0.0134333372116f;
    mParams.box_scales[2] = 0.00943253282458f;
    mParams.box_scales[3] = 0.00758471013978f;
    mParams.box_scales[4] = 0.00626726821065f;
#endif

#if DEBUG_RETINANET_NMS_PLUGIN // Set to 1 at top of the file to print input dimensions
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: Plugin Configured As :-\n");
    printf("[ RetinaNetNMSPVATRT ]....: nms_threshold { %f }\n", mParams.nms_threshold);
    printf("[ RetinaNetNMSPVATRT ]....: Score Scales { %f, %f, %f, %f, %f }\n", mParams.score_scales[0],
        mParams.score_scales[1], mParams.score_scales[2], mParams.score_scales[3], mParams.score_scales[4]);
    printf("[ RetinaNetNMSPVATRT ]....: Box Scales { %f, %f, %f, %f, %f }\n", mParams.box_scales[0],
        mParams.box_scales[1], mParams.box_scales[2], mParams.box_scales[3], mParams.box_scales[4]);
    printf("[ RetinaNetNMSPVATRT ]....: Score thresholds{ %f, %f, %f, %f, %f }\n", mParams.score_thresholds[0],
        mParams.score_thresholds[1], mParams.score_thresholds[2], mParams.score_thresholds[3],
        mParams.score_thresholds[4]);
#endif
}

// Return boolean for whether plugin supports type and format combinations
bool RetinaNetNMSPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept
{
    // error/type checks
    TRT_ASSERT((nbInputs == 11) && (nbOutputs == 1) && (pos < nbInputs + nbOutputs));
    bool isSupportedFormat{false};
    bool isSupportedType{false};

    // Ensure that TensorFormat is Supported
    if (pos < 10)
    {
        if ((inOut[pos].format == TensorFormat::kCHW32))
        {
            isSupportedFormat = true;
        }
        if (inOut[pos].type == nvinfer1::DataType::kINT8)
        {
            isSupportedType = true;
        }
    }
    else
    {
        if ((inOut[pos].format == TensorFormat::kLINEAR))
        {
            isSupportedFormat = true;
        }
        if (inOut[pos].type == nvinfer1::DataType::kFLOAT)
        {
            isSupportedType = true;
        }
    }

    bool finalBool = (isSupportedType && isSupportedFormat);

#if DEBUG_RETINANET_NMS_PLUGIN // Set to 1 at top of the file to print input dimensions
    if (finalBool)
    {
        printf("--------------------------:\n");
        printf(
            "[ RetinaNetNMSPVATRT ]....: Pos { %d } ==> SupportedType { %s }, SupportedFormat { %s }, Final { %s }\n",
            pos, isSupportedType ? (" True") : ("False"), isSupportedFormat ? (" True") : ("False"),
            finalBool ? (" True") : ("False"));
    }
#endif

    return (finalBool);
}

// Get the data type of the output from input types
DataType RetinaNetNMSPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    TRT_ASSERT(nbInputs == 11);
    return (nvinfer1::DataType::kFLOAT);
}

// Get plugin/op name
const char* RetinaNetNMSPlugin::getPluginType() const noexcept
{
    return RETINANET_NMS_PLUGIN_NAME;
}

// Get plugin version
const char* RetinaNetNMSPlugin::getPluginVersion() const noexcept
{
    return RETINANET_NMS_PLUGIN_VERSION;
}

// Destroy plugin object
void RetinaNetNMSPlugin::destroy() noexcept
{
    delete this;
}

// Clone plugin object
IPluginV2IOExt* RetinaNetNMSPlugin::clone() const noexcept
{
#if DEBUG_RETINANET_NMS_PLUGIN
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: Inside RetinaNetNMSPlugin::clone\n");
#endif

    IPluginV2IOExt* plugin{nullptr};
    try
    {
        plugin = new RetinaNetNMSPlugin(*this);
        plugin->setPluginNamespace(mPluginNamespace);
    }
    catch (std::exception& e)
    {
        printf("--------------------------:\n");
        printf("[ RetinaNetNMSPVATRT ]....: Exception caught while instantiating RetinaNetNMSPlugin in clone\n");
    }
    return plugin;
}

// Set plugin namespace
void RetinaNetNMSPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

// Get plugin namespace
const char* RetinaNetNMSPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

// Whether output broadcasts across batch
bool RetinaNetNMSPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Whether input can broadcast across batch
bool RetinaNetNMSPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

#if NV_TENSORRT_MAJOR == 7
int32_t RetinaNetNMSPlugin::enqueue(
    int32_t batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
#else
int32_t RetinaNetNMSPlugin::enqueue(
    int32_t batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
#endif
{
#if DEBUG_RETINANET_NMS_PLUGIN // Set to 1 at top of the file to print input dimensions
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: RetinaNetNMSPlugin::enqueue\n");
#endif

    mRetinaNetNMSAppSyncObj->launch(batchSize, inputs, outputs, stream);

    TRT_ASSERT(cudaPeekAtLastError() == 0);

#if DEBUG_RETINANET_NMS_PLUGIN // Set to 1 at top of the file to print input dimensions
    printf("--------------------------:\n");
    printf("[ RetinaNetNMSPVATRT ]....: RetinaNetNMSPlugin::enqueue, returning from enqueue\n");
#endif

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// Plugin Creator ///////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor for the plugin creator
RetinaNetNMSPluginCreator::RetinaNetNMSPluginCreator()
{
    mPluginAttributes.clear();

    mPluginAttributes.emplace_back(PluginField("nmsThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("score0Thresh", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("score1Thresh", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("score2Thresh", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("score3Thresh", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("score4Thresh", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = helper::numericCast<int32_t, size_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

// Get the name of the op/plugin
const char* RetinaNetNMSPluginCreator::getPluginName() const noexcept
{
    return RETINANET_NMS_PLUGIN_NAME;
}

// Get the plugin version
const char* RetinaNetNMSPluginCreator::getPluginVersion() const noexcept
{
    return RETINANET_NMS_PLUGIN_VERSION;
}

// Set plugin namespace
void RetinaNetNMSPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

// Get plugin namespace
const char* RetinaNetNMSPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// Get the names of the plugin fields
const PluginFieldCollection* RetinaNetNMSPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

// Create plugin using plugin name and field collection
IPluginV2IOExt* RetinaNetNMSPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    // Pointer to plugin
    IPluginV2IOExt* plugin{nullptr};
    // Field collection is not NULL
    if (fc != nullptr)
    {
        try
        {
            // Generate the new plugin and set its members
            plugin = new RetinaNetNMSPlugin(*fc);
            plugin->setPluginNamespace(mNamespace.c_str());
        }
        catch (std::exception& e)
        {
            printf("--------------------------:\n");
            printf(
                "[ RetinaNetNMSPVATRT ]....: Exception caught while instantiating RetinaNetNMSPlugin in "
                "createPlugin\n");
        }
    }
    return plugin;
}

// Create plugin using plugin name and serial data buffer
IPluginV2IOExt* RetinaNetNMSPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // Pointer to plugin object
    IPluginV2IOExt* plugin{nullptr};
    try
    {
        plugin = new RetinaNetNMSPlugin(name, serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
    }
    catch (std::exception& e)
    {
        printf("--------------------------:\n");
        printf(
            "[ RetinaNetNMSPVATRT ]....: Exception caught while instantiating RetinaNetNMSPlugin in "
            "deserializePlugin\n");
    }
    return plugin;
}

} // namespace retinanetnms
} // namespace nvinfer1
