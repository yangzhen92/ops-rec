/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#include <cstdint>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"
#include "tiling_policy_block_sparse_func.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    // 获取输入形状
    const gert::StorageShape* maskCntShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("maskCntShape", maskCntShape, return ge::GRAPH_FAILED);

    int64_t totalLength = maskCntShape->GetStorageShape().GetShapeSize();

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t maxAivCores = static_cast<int64_t>(ascendPlatform.GetCoreNumAiv());
    constexpr int64_t ELEMS_PER_CORE = 8192;
    int64_t coreNum = (totalLength + ELEMS_PER_CORE - 1) / ELEMS_PER_CORE;
    coreNum = coreNum > maxAivCores ? maxAivCores : coreNum;
    coreNum = coreNum < 1 ? 1 : coreNum;

    // 设置 tiling 数据
    DualExclusiveScanTilingData tiling;
    tiling.set_totalLength(totalLength);
    tiling.set_coreNum(coreNum);

    // 多核并发执行：各核先算本地总和，经 workspace 交换 + SyncAll 同步后叠加核间前缀
    context->SetBlockDim(coreNum);

    // workspace 布局（int32 计）：[0, coreNum+1) 存各核 mask 总和，[coreNum+1, 2*coreNum+2) 存 full 总和
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspaceSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = static_cast<size_t>(2 * (coreNum + 1)) * sizeof(int32_t) + systemWorkspaceSize;

    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* maskCntShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("maskCntShape", maskCntShape, return ge::GRAPH_FAILED);

    int64_t inputLength = maskCntShape->GetShapeSize();

    // 输出形状为 [inputLength + 1]
    gert::Shape* maskOffsetShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("maskOffsetShape", maskOffsetShape, return ge::GRAPH_FAILED);
    maskOffsetShape->SetDimNum(1);
    maskOffsetShape->SetDim(0, inputLength + 1);

    gert::Shape* fullOffsetShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL("fullOffsetShape", fullOffsetShape, return ge::GRAPH_FAILED);
    fullOffsetShape->SetDimNum(1);
    fullOffsetShape->SetDim(0, inputLength + 1);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    auto inputDataType = context->GetInputDataType(0);
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(0, inputDataType)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(1, inputDataType)) {
        return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class DualExclusiveScan : public OpDef {
public:
    explicit DualExclusiveScan(const char* name) : OpDef(name)
    {
        this->Input("mask_cnt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("full_cnt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("mask_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("full_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(DualExclusiveScan);

}  // namespace ops
