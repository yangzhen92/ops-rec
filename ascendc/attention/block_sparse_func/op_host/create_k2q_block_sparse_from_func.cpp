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
#include "../op_kernel/block_sparse_common.h"

namespace {
#define DIVUP(x, y) (((x + y - 1) / y))
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    // 获取输入 func_tensor 形状 [B, H, func_q_len, n_func]
    const gert::StorageShape* funcShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("funcShape", funcShape, return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const int32_t* strideB = attrs->GetAttrPointer<int32_t>(0);
    const int32_t* strideH = attrs->GetAttrPointer<int32_t>(1);
    const int32_t* strideQ = attrs->GetAttrPointer<int32_t>(2);
    const int32_t* strideF = attrs->GetAttrPointer<int32_t>(3);
    const int32_t* qLen = attrs->GetAttrPointer<int32_t>(4);
    const int32_t* kvLen = attrs->GetAttrPointer<int32_t>(5);
    const int32_t* qBlockSize = attrs->GetAttrPointer<int32_t>(6);
    const int32_t* kvBlockSize = attrs->GetAttrPointer<int32_t>(7);
    OPS_LOG_E_IF_NULL("strideB", strideB, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("strideH", strideH, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("strideQ", strideQ, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("strideF", strideF, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("qLen", qLen, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("kvLen", kvLen, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("qBlockSize", qBlockSize, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("kvBlockSize", kvBlockSize, return ge::GRAPH_FAILED);

    int32_t bNum = funcShape->GetStorageShape().GetDim(0);
    int32_t hNum = funcShape->GetStorageShape().GetDim(1);
    int32_t funcQLen = funcShape->GetStorageShape().GetDim(2);
    int32_t nFunc = funcShape->GetStorageShape().GetDim(3);
    OPS_LOG_E_IF(*qLen <= 0, context, return ge::GRAPH_FAILED, "qLen must be greater than 0");
    OPS_LOG_E_IF(*kvLen <= 0, context, return ge::GRAPH_FAILED, "kvLen must be greater than 0");
    OPS_LOG_E_IF(*qBlockSize <= 0 || *qBlockSize % 32 != 0, context, return ge::GRAPH_FAILED, "qBlockSize must be greater than 0 and divisible by 32");
    OPS_LOG_E_IF(*kvBlockSize <= 0 || *kvBlockSize % 32 != 0, context, return ge::GRAPH_FAILED, "kvBlockSize must be greater than 0 and divisible by 32");
    OPS_LOG_E_IF(funcQLen < *qLen, context, return ge::GRAPH_FAILED, "func_q_len must be greater than or equal to qLen");
    OPS_LOG_E_IF(nFunc < 0 || nFunc % 2 != 0, context, return ge::GRAPH_FAILED, "nFunc must be greater than or equal to 0 and divisible by 2");

    int32_t numQBlocks = DIVUP(*qLen, *qBlockSize);
    int32_t numKvBlocks = DIVUP(*kvLen, *kvBlockSize);
    int32_t totalBlocks = numKvBlocks * hNum * bNum;

    // 设置 tiling 数据
    CreateK2qBlockSparseFromFuncTilingData tiling;
    tiling.set_strideB(*strideB);
    tiling.set_strideH(*strideH);
    tiling.set_strideQ(*strideQ);
    tiling.set_strideF(*strideF);
    tiling.set_qLen(*qLen);
    tiling.set_kvLen(*kvLen);
    tiling.set_nFunc(nFunc);
    tiling.set_qBlockSize(*qBlockSize);
    tiling.set_kvBlockSize(*kvBlockSize);
    tiling.set_numQBlocks(numQBlocks);
    tiling.set_numKvBlocks(numKvBlocks);
    tiling.set_hNum(hNum);
    tiling.set_bNum(bNum);
    tiling.set_totalBlocks(totalBlocks);

#ifdef ENABLE_WARP_LEVEL_OPT
    int32_t grid = DIVUP(totalBlocks, WARP_LEVEL_NUM);
#else
    int32_t grid = totalBlocks;
#endif
    grid = totalBlocks >= 65535 ? 65535 : totalBlocks;
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

    const gert::Shape* funcShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("funcShape", funcShape, return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const int32_t* qLen = attrs->GetAttrPointer<int32_t>(4);
    const int32_t* kvLen = attrs->GetAttrPointer<int32_t>(5);
    const int32_t* qBlockSize = attrs->GetAttrPointer<int32_t>(6);
    const int32_t* kvBlockSize = attrs->GetAttrPointer<int32_t>(7);
    OPS_LOG_E_IF_NULL("qLen", qLen, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("kvLen", kvLen, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("qBlockSize", qBlockSize, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("kvBlockSize", kvBlockSize, return ge::GRAPH_FAILED);

    int64_t bNum = funcShape->GetDim(0);
    int64_t hNum = funcShape->GetDim(1);
    int64_t numQBlocks = (*qLen + *qBlockSize - 1) / *qBlockSize;
    int64_t numKvBlocks = (*kvLen + *kvBlockSize - 1) / *kvBlockSize;

    // mask_block_cnt / full_block_cnt: [B, H, numKvBlocks]
    gert::Shape* maskBlockCntShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("maskBlockCntShape", maskBlockCntShape, return ge::GRAPH_FAILED);
    maskBlockCntShape->SetDimNum(3);
    maskBlockCntShape->SetDim(0, bNum);
    maskBlockCntShape->SetDim(1, hNum);
    maskBlockCntShape->SetDim(2, numKvBlocks);

    gert::Shape* fullBlockCntShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL("fullBlockCntShape", fullBlockCntShape, return ge::GRAPH_FAILED);
    fullBlockCntShape->SetDimNum(3);
    fullBlockCntShape->SetDim(0, bNum);
    fullBlockCntShape->SetDim(1, hNum);
    fullBlockCntShape->SetDim(2, numKvBlocks);

    // block_idx: [B, H, numKvBlocks, numQBlocks]
    gert::Shape* blockIdxShape = context->GetOutputShape(2);
    OPS_LOG_E_IF_NULL("blockIdxShape", blockIdxShape, return ge::GRAPH_FAILED);
    blockIdxShape->SetDimNum(4);
    blockIdxShape->SetDim(0, bNum);
    blockIdxShape->SetDim(1, hNum);
    blockIdxShape->SetDim(2, numKvBlocks);
    blockIdxShape->SetDim(3, numQBlocks);

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
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(2, inputDataType)) {
        return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class CreateK2qBlockSparseFromFunc : public OpDef {
public:
    explicit CreateK2qBlockSparseFromFunc(const char* name) : OpDef(name)
    {
        this->Input("func_tensor")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("q_block_kv_min")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("q_block_kv_max")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("mask_block_cnt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("full_block_cnt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("block_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("stride_b").AttrType(REQUIRED).Int(0);
        this->Attr("stride_h").AttrType(REQUIRED).Int(0);
        this->Attr("stride_q").AttrType(REQUIRED).Int(0);
        this->Attr("stride_f").AttrType(REQUIRED).Int(0);
        this->Attr("q_len").AttrType(REQUIRED).Int(0);
        this->Attr("kv_len").AttrType(REQUIRED).Int(0);
        this->Attr("q_block_size").AttrType(REQUIRED).Int(0);
        this->Attr("kv_block_size").AttrType(REQUIRED).Int(0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(CreateK2qBlockSparseFromFunc);

}  // namespace ops
