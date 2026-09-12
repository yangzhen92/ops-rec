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
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
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
    const bool* checkQBoundary = attrs->GetAttrPointer<bool>(8);
    OPS_LOG_E_IF_NULL("strideB", strideB, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("strideH", strideH, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("strideQ", strideQ, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("strideF", strideF, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("qLen", qLen, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("kvLen", kvLen, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("qBlockSize", qBlockSize, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("kvBlockSize", kvBlockSize, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("checkQBoundary", checkQBoundary, return ge::GRAPH_FAILED);

    int32_t bNum = funcShape->GetStorageShape().GetDim(0);
    int32_t hNum = funcShape->GetStorageShape().GetDim(1);
    int32_t funcQLen = funcShape->GetStorageShape().GetDim(2);
    int32_t nFunc = funcShape->GetStorageShape().GetDim(3);
    OPS_LOG_E_IF(*qLen <= 0, context, return ge::GRAPH_FAILED, "qLen must be greater than 0");
    OPS_LOG_E_IF(*kvLen <= 0, context, return ge::GRAPH_FAILED, "kvLen must be greater than 0");
    OPS_LOG_E_IF(*qBlockSize <= 0 || *qBlockSize % 32 != 0, context, return ge::GRAPH_FAILED,
                 "qBlockSize must be greater than 0 and divisible by 32");
    OPS_LOG_E_IF(*kvBlockSize <= 0 || *kvBlockSize % 32 != 0, context, return ge::GRAPH_FAILED,
                 "kvBlockSize must be greater than 0 and divisible by 32");
    OPS_LOG_E_IF(funcQLen < *qLen, context, return ge::GRAPH_FAILED,
                 "func_q_len must be greater than or equal to qLen");
    OPS_LOG_E_IF(nFunc % 2 != 0, context, return ge::GRAPH_FAILED, "nFunc must be divisible by 2");

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t aivCoreCount = static_cast<int32_t>(ascendPlatform.GetCoreNumAiv());
    int32_t numQBlocks = DIVUP(*qLen, *qBlockSize);
    int32_t numKvBlocks = DIVUP(*kvLen, *kvBlockSize);
    int32_t totalBlocks = numQBlocks * hNum * bNum;

    // 设置 tiling 数据
    CreateQ2kBlockSparseFromFuncTilingData tiling;
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
    tiling.set_checkQBoundary(*checkQBoundary ? 1 : 0);

#ifdef ENABLE_WARP_LEVEL_OPT
    // warp 级(qB<=256)：每个 256 线程 block=8 warps=8 个 q_block；block 级(qB>256 如 512)：每个 block 处理
    // 一个 q_block（256 线程各 1 token 或 split 512 token x2）。block 级若仍用 DIVUP(totalBlocks,8) 会把大批
    // q_block 压到极少数 block/核上，造成严重欠并行，故 block 级应 grid = totalBlocks（每核一个 q_block）。
    const bool use_warp_level = (*qBlockSize <= BLOCK_THREAD_NUM_256) && (*qBlockSize % 32 == 0);
    // 多 warp KV 拆分：warp 级路径下，若 q_block 总数不足以占满 AIV 核，则把每个 q_block 的 KV 区间
    // 拆给 block 内 8 个 warp 并行扫描。grid 改为 totalBlocks（每核一个 q_block），用满空余核，
    // 直击 q_block 少/单核串行扫描的欠并行根因。
    const int32_t split_waves = DIVUP(totalBlocks, aivCoreCount);
    const bool kv_split_mode = use_warp_level && (split_waves <= 2) &&
                               (numKvBlocks >= 16) && (numKvBlocks <= SPLIT_MAX_KV_BLOCKS);
    int32_t grid = kv_split_mode ? totalBlocks
                                 : (use_warp_level ? DIVUP(totalBlocks, WARP_LEVEL_NUM) : totalBlocks);
    tiling.set_kvSplitMode(kv_split_mode ? 1 : 0);
#else
    int32_t grid = totalBlocks;
    tiling.set_kvSplitMode(0);
#endif
    grid = grid >= 65535 ? 65535 : grid;
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

    // mask_block_cnt / full_block_cnt: [B, H, numQBlocks]
    gert::Shape* maskBlockCntShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("maskBlockCntShape", maskBlockCntShape, return ge::GRAPH_FAILED);
    maskBlockCntShape->SetDimNum(3);
    maskBlockCntShape->SetDim(0, bNum);
    maskBlockCntShape->SetDim(1, hNum);
    maskBlockCntShape->SetDim(2, numQBlocks);

    gert::Shape* fullBlockCntShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL("fullBlockCntShape", fullBlockCntShape, return ge::GRAPH_FAILED);
    fullBlockCntShape->SetDimNum(3);
    fullBlockCntShape->SetDim(0, bNum);
    fullBlockCntShape->SetDim(1, hNum);
    fullBlockCntShape->SetDim(2, numQBlocks);

    // block_idx: [B, H, numQBlocks, numKvBlocks]
    gert::Shape* blockIdxShape = context->GetOutputShape(2);
    OPS_LOG_E_IF_NULL("blockIdxShape", blockIdxShape, return ge::GRAPH_FAILED);
    blockIdxShape->SetDimNum(4);
    blockIdxShape->SetDim(0, bNum);
    blockIdxShape->SetDim(1, hNum);
    blockIdxShape->SetDim(2, numQBlocks);
    blockIdxShape->SetDim(3, numKvBlocks);

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

class CreateQ2kBlockSparseFromFunc : public OpDef {
public:
    explicit CreateQ2kBlockSparseFromFunc(const char* name) : OpDef(name)
    {
        this->Input("func_tensor")
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
        this->Attr("check_q_boundary").AttrType(REQUIRED).Bool(false);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(CreateQ2kBlockSparseFromFunc);

}  // namespace ops
