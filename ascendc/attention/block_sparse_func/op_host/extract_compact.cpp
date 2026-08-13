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
#include "tiling_policy_block_sparse_func.h"

#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"

namespace {
constexpr int32_t MAX_NUM_BLOCKS = 65535;
}
namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    // 获取输入blockId形状[B, H, numBlocks, maxBlocks]
    const gert::StorageShape* blockIdShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("blockIdShape", blockIdShape, return ge::GRAPH_FAILED);

    // 获取输入maskCnt形状[B, H, numBlocks]
    const gert::StorageShape* maskCntShape = context->GetInputShape(1);
    OPS_LOG_E_IF_NULL("maskCntShape", maskCntShape, return ge::GRAPH_FAILED);

    int32_t numBlocks = maskCntShape->GetStorageShape().GetShapeSize();
    OPS_LOG_E_IF(numBlocks <= 0, context, return ge::GRAPH_FAILED, "numBlocks must be greater than 0");

    int32_t maxBlocks = blockIdShape->GetStorageShape().GetDim(3);
    OPS_LOG_E_IF(maxBlocks <= 0, context, return ge::GRAPH_FAILED, "maxBlocks must be greater than 0");

    // 设置 tiling 数据
    ExtractCompactTilingData tiling;
    tiling.set_nBlocks(numBlocks);
    tiling.set_maxBlocks(maxBlocks);

    // 使用多核执行
    int grid = numBlocks >= MAX_NUM_BLOCKS ? MAX_NUM_BLOCKS : numBlocks;

    context->SetBlockDim(grid);

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

    const gert::Shape* blockIdShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("blockIdShape", blockIdShape, return ge::GRAPH_FAILED);

    int32_t nBlocks = blockIdShape->GetDim(0);
    int32_t maxBlocks = blockIdShape->GetDim(1);

    // 输出形状为 [nBlocks * maxBlocks]
    gert::Shape* maskCompactShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("maskCompactShape", maskCompactShape, return ge::GRAPH_FAILED);
    maskCompactShape->SetDimNum(1);
    maskCompactShape->SetDim(0, nBlocks * maxBlocks);

    gert::Shape* fullCompactShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL("fullCompactShape", fullCompactShape, return ge::GRAPH_FAILED);
    fullCompactShape->SetDimNum(1);
    fullCompactShape->SetDim(0, nBlocks * maxBlocks);

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
class ExtractCompact : public OpDef {
public:
    explicit ExtractCompact(const char* name) : OpDef(name)
    {
        this->Input("block_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
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
        this->Input("mask_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("full_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("mask_compact")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("full_compact")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(ExtractCompact);

}  // namespace ops
