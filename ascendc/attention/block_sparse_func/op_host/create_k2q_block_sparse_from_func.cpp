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
    OPS_LOG_E_IF(*qBlockSize <= 0 || *qBlockSize % 32 != 0, context, return ge::GRAPH_FAILED,
                 "qBlockSize must be greater than 0 and divisible by 32");
    OPS_LOG_E_IF(*kvBlockSize <= 0 || *kvBlockSize % 32 != 0, context, return ge::GRAPH_FAILED,
                 "kvBlockSize must be greater than 0 and divisible by 32");
    OPS_LOG_E_IF(funcQLen < *qLen, context, return ge::GRAPH_FAILED,
                 "func_q_len must be greater than or equal to qLen");
    OPS_LOG_E_IF(nFunc % 2 != 0, context, return ge::GRAPH_FAILED, "nFunc must be divisible by 2");

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_LOG_E_IF(coreNum == 0, context, return ge::GRAPH_FAILED, "GetCoreNumAiv return 0");

    int32_t numQBlocks = DIVUP(*qLen, *qBlockSize);
    int32_t numKvBlocks = DIVUP(*kvLen, *kvBlockSize);
    int32_t totalBlocks = numKvBlocks * hNum * bNum;
    int32_t totalQTasks = numQBlocks * hNum * bNum;

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
    // 与 kernel 端 warp 分发保持一致：qBlockSize ∈ {32,64,128,256} 时走 warp 级路径
    // （每个 256 线程 block = 8 warps 处理 8 个 (b,h,q_block) 任务），否则 block 级每核一个任务。
    const bool use_warp_level =
        (*qBlockSize == 32) || (*qBlockSize == 64) || (*qBlockSize == 128) || (*qBlockSize == 256);
    const int32_t fillBlocks = use_warp_level ? DIVUP(totalQTasks, WARP_LEVEL_NUM) : totalQTasks;
#else
    const int32_t fillBlocks = totalQTasks;
#endif
    // 第二阶段（compact）：每个 warp 负责一个 (b*h, kv_group) 列组，每组 32 个 kv_block
    const int32_t compactGroups = DIVUP(numKvBlocks, WARP_SIZE_CONST) * hNum * bNum;
    const int32_t compactBlocks = DIVUP(compactGroups, WARP_LEVEL_NUM);

    // 小 shape 单阶段 warp-per-column 路径：行式 fill 并行度不足（fillBlocks*2 < AIV
    // 核数）且列式读放大量可控时，改走无 SyncAll、无状态矩阵的单阶段 kernel。
    // "多 block 小规模"场景（q_len 小、kv_block_size 小 → numKvBlocks 多）func 总量
    // 小、重复读几乎全命中 L2/DCache，故预算按 L2 流量放宽到 128MB；单阶段以
    // (b, h, 连续列组) 为任务，按 2 倍核数的目标任务数动态设置每组列数 kvColsPerTask
    //（8 的倍数，[8, 64]），既避免海量超小列任务的调度开销，又保证满核并行。
    const uint64_t funcBytesPerBH = static_cast<uint64_t>(*qLen) * nFunc * sizeof(int32_t);
    const uint64_t columnTraffic = static_cast<uint64_t>(bNum) * hNum * numKvBlocks * funcBytesPerBH;
    const bool useSinglePhase =
        (fillBlocks * 2 < static_cast<int32_t>(coreNum)) && (columnTraffic <= (128ULL << 20));
    int32_t kvColsPerTask = 8;  // 最小列组宽 = 每 block 的 warp 数
    if (useSinglePhase) {
        const int32_t targetTasks = 2 * static_cast<int32_t>(coreNum);
        if (totalBlocks > targetTasks) {
            kvColsPerTask = DIVUP(totalBlocks, targetTasks);
            kvColsPerTask = ((kvColsPerTask + 7) / 8) * 8;  // 向上取 8 的倍数（warp 数对齐）
            kvColsPerTask = kvColsPerTask > 64 ? 64 : kvColsPerTask;
        }
    }
    tiling.set_useSinglePhase(useSinglePhase ? 1 : 0);
    tiling.set_kvColsPerTask(kvColsPerTask);

    // 两阶段 fill 的 kv 列切分：大 q_len、小 B*H 的中大型用例行任务（nq*B*H）不足，
    // 无法吃满 AIV 核且每 warp 串行扫整行 kv 列。按"目标 warp 任务数 = 2×核数×8"
    // 把每行 kv 列切成多个 tile 倍增并行度；代价是每个 tile 任务重读一遍该行 func
    //（func 行远小于状态矩阵，且相邻 tile 的 warp 同 block 调度、DCache 命中）。
    int32_t kvColsPerFillTask = 0;  // 0 = 不切分
    if (!useSinglePhase) {
        const int32_t targetFillTasks = 2 * static_cast<int32_t>(coreNum) * WARP_LEVEL_NUM;
        if (totalQTasks < targetFillTasks && numKvBlocks > 1) {
            int32_t numTiles = DIVUP(targetFillTasks, totalQTasks);
            numTiles = numTiles > numKvBlocks ? numKvBlocks : numTiles;
            kvColsPerFillTask = DIVUP(numKvBlocks, numTiles);
        }
    }
    tiling.set_kvColsPerFillTask(kvColsPerFillTask);
    const int32_t fillTiles = (kvColsPerFillTask > 0) ? DIVUP(numKvBlocks, kvColsPerFillTask) : 1;
    const int32_t fillTasks = totalQTasks * fillTiles;
#ifdef ENABLE_WARP_LEVEL_OPT
    const int32_t fillBlocksTiled = use_warp_level ? DIVUP(fillTasks, WARP_LEVEL_NUM) : fillTasks;
#else
    const int32_t fillBlocksTiled = fillTasks;
#endif

    // SyncAll 硬同步要求 blockDim 不超过实际 AIV 核数；多余任务由 grid-stride 循环补齐。
    const int32_t singleBlocks = DIVUP(totalBlocks, kvColsPerTask);
    int32_t grid = useSinglePhase ? singleBlocks : (fillBlocksTiled > compactBlocks ? fillBlocksTiled : compactBlocks);
    grid = static_cast<int32_t>(coreNum) < grid ? static_cast<int32_t>(coreNum) : grid;
    grid = grid >= 65535 ? 65535 : grid;
    context->SetBlockDim(static_cast<uint32_t>(grid));
    if (!useSinglePhase) {
        // 使能核间同步（两阶段之间依赖 SyncAll 硬同步）
        context->SetScheduleMode(1);
    }

    // 用户 workspace：状态矩阵 [B, H, numQBlocks, numKvBlocks]（uint8 状态，仅两阶段路径使用）
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspaceSizes", workspaceSizes, return ge::GRAPH_FAILED);
    const uint64_t stateMatrixSize = static_cast<uint64_t>(bNum) * hNum * numQBlocks * numKvBlocks;
    workspaceSizes[0] = ascendPlatform.GetLibApiWorkSpaceSize() + (useSinglePhase ? 0 : stateMatrixSize);

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
