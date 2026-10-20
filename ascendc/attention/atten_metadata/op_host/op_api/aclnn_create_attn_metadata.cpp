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

#include <string>

#include "aclnn_create_attn_metadata.h"
#include "l0_create_attn_metadata.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

#include "../create_attn_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnCreateAttnMetadataGetWorkspaceSize(
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKvOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKvOptional, const aclTensor* maskBlockCntOptional, const aclTensor* fullBlockCntOptional,
    int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
    int64_t maskMode, int64_t winLeft, int64_t winRight, const char* layoutQ, const char* layoutKv,
    const char* layoutOut, int64_t packedRowAxis, int64_t packedMaskHeads, int64_t l2Size, int64_t qBlockSize,
    int64_t kvBlockSize, bool deterministic, const aclTensor* metaData, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnCreateAttnMetadata,
                   DFX_IN(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional,
                          maskBlockCntOptional, fullBlockCntOptional, batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ,
                          numHeadsKv, headDim, maskMode, winLeft, winRight, layoutQ, layoutKv, layoutOut, packedRowAxis,
                          packedMaskHeads, l2Size, qBlockSize, kvBlockSize, deterministic),
                   DFX_OUT(metaData));

    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CreateAttnMetadataCheck::ParamsCheck(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional,
                                                    sequsedKvOptional, batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ,
                                                    numHeadsKv, headDim, maskMode, winLeft, winRight, layoutQ, layoutKv,
                                                    layoutOut, metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo& npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const std::string socVersion = npuInfo.GetSocLongVersion();

    auto output = l0op::CreateAttnMetadata(
        cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, maskBlockCntOptional,
        fullBlockCntOptional, batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, maskMode, winLeft,
        winRight, layoutQ, layoutKv, layoutOut, socVersion.c_str(), aicCoreNum, aivCoreNum, packedRowAxis,
        packedMaskHeads, l2Size, qBlockSize, kvBlockSize, deterministic, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCreateAttnMetadata(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCreateAttnMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
