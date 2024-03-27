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

#include <cudnn.h>
#include <cub/cub.cuh>
#include "ssdOpt.h"
#include "ssdOptMacros.h"

// debug
#include "nms_common.h"
#include <vector>

#define CUDA_MEM_ALIGN 256

// ALIGNPTR {{{
int8_t* alignPtr(int8_t* ptr, uintptr_t to)
{
    uintptr_t addr = (uintptr_t) ptr;
    if (addr % to)
    {
        addr += to - addr % to;
    }
    return (int8_t*) addr;
}
// }}}

// NEXTWORKSPACEPTR {{{
int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t*) addr, CUDA_MEM_ALIGN);
}
// }}}

// CALCULATE TOTAL WORKSPACE SIZE {{{
size_t calculateTotalWorkspaceSize(size_t* workspaces, int count)
{
    size_t total = 0;
    for (int i = 0; i < count; i++)
    {
        total += workspaces[i];
        if (workspaces[i] % CUDA_MEM_ALIGN)
            {
            total += CUDA_MEM_ALIGN - (workspaces[i] % CUDA_MEM_ALIGN);
        }
    }
    return total;
}
// }}}

namespace nvinfer1
{
namespace plugin
{

ssdStatus_t detectionInferenceOpt(
    cudaStream_t stream,
    const int N,
    const int C1,
    const int C2,
    const bool shareLocation,
    const bool varianceEncodedInTarget,
    const int backgroundLabelId,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const float confidenceThreshold,
    const float nmsThreshold,
    const CodeTypeSSD codeType,
    const DType_t DT_BBOX,
    const void* const* locData,
    const void* priorData,
    const DType_t DT_SCORE,
    const void* const* confData,
    void* topDetections,
    void* workspace,
    bool isNormalized,
    bool confSigmoid,
    bool confSoftmax,
    bool permuteBeforeReshape,
    bool concatInputs,
    const int numLayers,
    const int* featureSize,
    const int* numAnchors,
    const int* boxChannels,
    const int* confChannels,
    const bool packed32NCHW,
    const bool packedConf32NCHW,
    const float* inScales,
    cudnnHandle_t cudnnHandle,
    cudnnTensorDescriptor_t inScoreDesc,
    cudnnTensorDescriptor_t outScoreDesc)
{
    // if we want to clip bbox output to [0,1]
    bool clip = true;

    const int locCount = N * C1;
    const bool clipBBox = false;
    const int numLocClasses = shareLocation ? 1 : numClasses;

    size_t bboxDataSize = detectionForwardBBoxDataSize(N, C1, DataType::kFLOAT);
    void* bboxDataRaw = workspace;

    ssdStatus_t status;

    auto dataTypeLoc =  DataType::kFLOAT;

    auto dataTypeConf =  DataType::kFLOAT;

    status = decodeBBoxesOpt(stream,
                                      locCount,
                                      codeType,
                                      varianceEncodedInTarget,
                                      numPredsPerClass,
                                      shareLocation,
                                      numLocClasses,
                                      backgroundLabelId,
                                      clipBBox,
                                      dataTypeLoc,
                                      locData,
                                      priorData,
                                      bboxDataRaw,
                                      numLayers,
                                      featureSize,
                                      numAnchors,
                                      boxChannels,
                                      confChannels,
                                      packed32NCHW,
                                      !permuteBeforeReshape /*softmax means reshape_before_permute*/,
                                      concatInputs);

    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);

    // float for now
    void* bboxData;
    size_t bboxPermuteSize = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DataType::kFLOAT);
    void* bboxPermute = nextWorkspacePtr((int8_t*) bboxDataRaw, bboxDataSize);

    SSD_ASSERT_FAILURE(shareLocation);
    bboxData = bboxDataRaw;

    size_t temp_active_counts_size, temp_sort_scores_size, temp_sort_indicies_size, temp_cub_storage_bytes;
    temp_cub_storage_bytes = topKScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, topK, DT_SCORE,
                                    temp_active_counts_size, temp_sort_scores_size, temp_sort_indicies_size);
    temp_cub_storage_bytes = temp_cub_storage_bytes - temp_active_counts_size - temp_sort_scores_size - temp_sort_indicies_size;

    const int numScores = N * C2;
    size_t scoresSize = std::max(detectionForwardPreNMSSize(N, C2), temp_sort_scores_size);
    void* scores = nextWorkspacePtr((int8_t*) bboxPermute, bboxPermuteSize);
    void* softmaxScores = nextWorkspacePtr((int8_t*) scores, scoresSize);
    void* temp_scores = nullptr; 

    size_t indicesSize = std::max(detectionForwardPreNMSSize(N, C2), temp_sort_indicies_size);
    void* indices = nextWorkspacePtr((int8_t*) softmaxScores, scoresSize);

    void* temp_indices = nextWorkspacePtr((int8_t*) indices, indicesSize);
    void* temp_active_counts = nextWorkspacePtr((int8_t*) temp_indices, indicesSize);
    void* temp_cub_storage = nextWorkspacePtr((int8_t*) temp_active_counts, temp_active_counts_size);

    size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topK);
    size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topK);
    void* postNMSScores = nextWorkspacePtr((int8_t*) temp_cub_storage, temp_cub_storage_bytes);
    void* postNMSIndices = nextWorkspacePtr((int8_t*) postNMSScores, postNMSScoresSize);

    size_t numSegments = N * numClasses;
    size_t activeCountSize = numSegments * sizeof(int);
    void *activeCount = nextWorkspacePtr((int8_t*) postNMSIndices, postNMSIndicesSize);

    // to reduce work we want to know the amount of active elements per class after allClassNMS
    size_t activeCountPerBatchSize = N * sizeof(int);
    void *activeCountPerBatch = nextWorkspacePtr((int8_t*) activeCount, activeCountSize);

    void* sortingWorkspace = nextWorkspacePtr((int8_t*) activeCountPerBatch, activeCountPerBatchSize);

    // TODO Add support for both permutations
    if(confSoftmax && SSD_RETINA_NET == 0) 
    { // confSoftmax 
        DEBUG_PRINTF("Forward: permuteBeforeReshape = %d\n", permuteBeforeReshape);

        status = permuteConfData(stream,
                             numScores,
                             numClasses,
                             numPredsPerClass,
                             1,
                             dataTypeConf,
                             confSigmoid,
                             confData,
                             scores,
                             activeCount,
                             numLayers,
                             featureSize,
                             numAnchors,
                             boxChannels,
                             permuteBeforeReshape,
                             concatInputs,
                             packed32NCHW);
        SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);
    }


    if(confSoftmax) {
        status = softmaxScore(stream,
                        N,
                        numClasses,
                        numPredsPerClass,
                        1,
                        DataType::kFLOAT,
                        scores,
                        softmaxScores,
                        cudnnHandle,
                        inScoreDesc,
                        outScoreDesc); 
        SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);
    
        temp_scores = scores;
        scores = softmaxScores;
    } else {
        temp_scores = softmaxScores;
    }

    //if(!permuteBeforeReshape || confSoftmax)
    if (SSD_RETINA_NET == 1 || confSoftmax)
    { 
        status = topKScoresPerClass(stream,
                                    N,
                                    numClasses,
                                    numPredsPerClass,
                                    topK,
                                    backgroundLabelId,
                                    confidenceThreshold,
                                    DataType::kFLOAT,
                                    scores,
                                    indices,
                                    activeCount,
                                    activeCountPerBatch,
                                    temp_scores,
                                    temp_indices,
                                    temp_active_counts,
                                    temp_cub_storage_bytes,
                                    temp_cub_storage,
                                    numPredsPerClass,
                                    1,
                                    confSigmoid,
                                    confData,
                                    numLayers,
                                    featureSize,
                                    numAnchors,
                                    boxChannels,
                                    packed32NCHW,
                                    packedConf32NCHW,
                                    inScales);
                                    //sortingWorkspace);
    }
    else {
        assert(concatInputs == true);
        status = topKScoresPerClassFusedPermute(stream,
                                    N,
                                    numClasses,
                                    numPredsPerClass,
                                    topK,
                                    backgroundLabelId,
                                    confidenceThreshold,
                                    DataType::kFLOAT,
                                    scores,
                                    indices,
                                    activeCount,
                                    activeCountPerBatch,
                                    sortingWorkspace,
                                    numPredsPerClass,
                                    1,
                                    confSigmoid,
                                    confData,
                                    numLayers,
                                    featureSize,
                                    numAnchors,
                                    boxChannels,
                                    packed32NCHW);
    }

    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);

    status = allClassNMSOpt(stream,
                         N,
                         numClasses,
                         numPredsPerClass,
                         topK,
                         nmsThreshold,
                         shareLocation,
                         isNormalized,
                         DataType::kFLOAT,
                         DataType::kFLOAT,
                         bboxData,
                         scores,
                         indices,
                         postNMSScores,
                         postNMSIndices,
                         activeCount,
                         activeCountPerBatch,
                         false);
    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);
    
    status = topKScoresPerImage(stream,
                                N,
                                numClasses * topK,
                                topK,
                                DataType::kFLOAT,
                                postNMSScores,
                                postNMSIndices,
                                scores,
                                indices,
                                activeCount,
                                activeCountPerBatch,
                                temp_active_counts,
                                temp_cub_storage_bytes,
                                temp_cub_storage);
                                //sortingWorkspace);
    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);

    status = gatherTopDetectionsOpt(stream,
                                 shareLocation,
                                 clip,
                                 N,
                                 numPredsPerClass,
                                 numClasses,
                                 topK,
                                 keepTopK,
                                 DataType::kFLOAT,
                                 DataType::kFLOAT,
                                 indices,
                                 scores,
                                 bboxData,
                                 topDetections);
    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

} // namespace plugin
} // namespace nvinfer1
