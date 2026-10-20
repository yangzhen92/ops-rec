/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
============================================================================== */

#ifndef ACLNN_CREATE_ATTN_METADATA_H
#define ACLNN_CREATE_ATTN_METADATA_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnCreateAttnMetadataGetWorkspaceSize(
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKvOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKvOptional, const aclTensor* maskBlockCntOptional, const aclTensor* fullBlockCntOptional,
    int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
    int64_t maskMode, int64_t winLeft, int64_t winRight, const char* layoutQ, const char* layoutKv,
    const char* layoutOut, int64_t packedRowAxis, int64_t packedMaskHeads, int64_t l2Size, int64_t qBlockSize,
    int64_t kvBlockSize, bool deterministic, const aclTensor* metaData, uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnCreateAttnMetadata(void* workspace, uint64_t workspaceSize,
                                                                           aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_CREATE_ATTN_METADATA_H
