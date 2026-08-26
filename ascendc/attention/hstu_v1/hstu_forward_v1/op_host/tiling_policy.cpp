/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include "tiling_policy.h"
#include <cstdint>
#include "register/op_def_registry.h"
#include "matmul_check.h"
#include "tiling_policy_define.h"

using namespace MatmulTilingCheck;

namespace HstuForward {

ShapeRange::ShapeRange(int64_t lbound, int64_t ubound, int64_t multiple, const char* name)
{
    this->lbound = lbound;
    this->ubound = ubound;
    this->multiple = multiple;
    this->name = name;
}

bool ShapeRange::Check(int64_t val) const
{
    OPS_CHECK((val < lbound || val > ubound || val % multiple != 0),
              OPS_LOG_E("", "%s must meet range[%lld %lld] and multiple of [%lld]. but get value %lld\n", name, lbound,
                        ubound, multiple, val),
              return false);
    return true;
}

ge::graphStatus TilingPolicy::InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* queryShape = context->GetInputShape(INDEX_T::INDEX_0);
    OPS_CHECK_PTR_NULL(queryShape, return ge::GRAPH_FAILED);

    gert::Shape* attenOutputShape = context->GetOutputShape(INDEX_T::INDEX_0);
    OPS_CHECK_PTR_NULL(attenOutputShape, return ge::GRAPH_FAILED);
    *attenOutputShape = *queryShape;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPolicy::InferDtype(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPolicy::TilingProcess(gert::TilingContext* context)
{
    return ge::GRAPH_SUCCESS;
}

bool TilingPolicy::GeneralShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim, bool dimAlign)
{
    auto dimStep = dimAlign ? 16 : 1;  // 16 means align to C0_size 1 means not align
    const ShapeRange SEQ_RANGE(1, 20480, 1, "seq size");
    const ShapeRange BATCH_RANGE(1, MAX_BATCH_SIZE, 1, "batch size");
    const ShapeRange DIM_RANGE(dimStep, 512, dimStep, "dim size");  // 非对齐情况下dim范围[1,512]
    const ShapeRange HEAD_RANGE(1, 16, 1, "head num");

    if (!SEQ_RANGE.Check(seqLen)) {
        return false;
    }

    if (!BATCH_RANGE.Check(batchSize)) {
        return false;
    }

    if (!HEAD_RANGE.Check(headNum)) {
        return false;
    }

    if (!DIM_RANGE.Check(dim)) {
        return false;
    }

    return true;
}

bool TilingPolicy::TilingWorkSpace(gert::TilingContext* context)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return false);

    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    size_t coreNum = ascendPlatform.GetCoreNumAic();

    int64_t oneBlockMidElem = BLOCK_HEIGHT * BLOCK_HEIGHT * COMPUTE_PIPE_NUM;
    int64_t oneCoreMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

    int64_t oneBlockMidTransElem = BLOCK_HEIGHT * MAX_DIM * TRANS_PIPE_NUM;
    int64_t oneCoreTransMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidTransElem;

    int64_t workspaceSize = (oneCoreMidElem + oneCoreTransMidElem) * sizeof(float);
    // 架构相关额外 workspace（A5 平台的 FP8 workspace，见各架构 tiling_policy_arch.cpp）
    workspaceSize += GetArchWorkspaceExtraSize(coreNum);
    int64_t syncSize = coreNum * VCORE_NUM_IN_ONE_AIC * DATA_ALIGN_BYTES;
    currentWorkspace[0] = workspaceSize + systemWorkspacesSize + syncSize;
    return true;
}

bool TilingPolicy::TilingCore(gert::TilingContext* context)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum > MAX_AIV_NUM, OPS_LOG_E("", "vecCoreNum %d should be < %d\n", coreNum, MAX_AIV_NUM),
              return false);
    size_t aicCoreNum = ascendPlatform.GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);
    return true;
}

}  // namespace HstuForward
