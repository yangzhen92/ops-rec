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
============================================================================== */

#include <cstdint>
#include <algorithm>
#include <array>

#include "register/op_def_registry.h"

#include "ops_log.h"
#include "hstu_forward_v2_define.h"
#include "hstu_forward_v2_tiling.h"
#include "../op_kernel/catlass_hstu/kernel/fwd/forward_kernel_tiling.hpp"
#include "../common/hstu_tile_size.h"

namespace optiling {
static ops_rec::hstu::CoreMemSize GetCoreMemSize(platform_ascendc::PlatformAscendC& platform)
{
    ops_rec::hstu::CoreMemSize mem;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, mem.l0aBytes);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, mem.l0bBytes);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, mem.l0cBytes);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, mem.l1Bytes);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, mem.ubBytes);
    return mem;
}

static bool ParseShape(gert::TilingContext* context, HstuForwardV2TilingData& tilingData)
{
    static constexpr std::array<int64_t, 2> supportHeadRange = {1, 16};

    const auto qShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::Q))->GetStorageShape();
    const auto vShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::V))->GetStorageShape();
    const auto seqOffsetShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::SEQ_OFFSET_Q))->GetStorageShape();

    auto totalSeqLenQ = qShape.GetDim(static_cast<size_t>(DIM_INDEX::ZERO));
    auto totalSeqLenK = vShape.GetDim(static_cast<size_t>(DIM_INDEX::ZERO));
    auto batch = seqOffsetShape.GetDim(static_cast<size_t>(DIM_INDEX::ZERO)) - 1;
    auto headQ = qShape.GetDim(static_cast<size_t>(DIM_INDEX::ONE));
    auto headK = vShape.GetDim(static_cast<size_t>(DIM_INDEX::ONE));
    auto dimQK = qShape.GetDim(static_cast<size_t>(DIM_INDEX::TWO));
    auto dimV = vShape.GetDim(static_cast<size_t>(DIM_INDEX::TWO));

    auto dimChecker = [](int64_t dim) -> bool {
        return ((dim % 16) == 0);
    };

    auto headChecker = [](int64_t head) -> bool {
        return head >= supportHeadRange[0] && head <= supportHeadRange[1];
    };

    OPS_CHECK(headQ != headK, OPS_LOG_E("", "tiling failed current only support MHA.\n"), return false);

    OPS_CHECK(!dimChecker(dimQK), OPS_LOG_E("", "tiling failed dimQK must mutiple of 16.\n"), return false);

    OPS_CHECK(!dimChecker(dimV), OPS_LOG_E("", "tiling failed dimGV must mutiple of 16.\n"), return false);
    OPS_CHECK(!headChecker(headQ), OPS_LOG_E("", "tiling failed headQ is not in support range[1~16].\n"), return false);

    tilingData.set_batch(batch);
    tilingData.set_heads(headQ);
    tilingData.set_dimQK(dimQK);
    tilingData.set_dimV(dimV);
    tilingData.set_totalSeqLenQ(totalSeqLenQ);
    tilingData.set_totalSeqLenK(totalSeqLenK);
    return true;
}

static bool ParseAttr(gert::TilingContext* context, HstuForwardV2TilingData& tilingData)
{
    const auto* attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, false);

    const auto* maxSeqQ = attrs->GetAttrPointer<uint32_t>(static_cast<size_t>(ATTR_INDEX::MAX_SEQLEN_Q));
    OPS_CHECK_PTR_NULL(maxSeqQ, false);

    const auto* maxSeqK = attrs->GetAttrPointer<uint32_t>(static_cast<size_t>(ATTR_INDEX::MAX_SEQLEN_K));
    OPS_CHECK_PTR_NULL(maxSeqK, false);

    const auto* targetGroupSize = attrs->GetAttrPointer<int32_t>(static_cast<size_t>(ATTR_INDEX::TARGET_GROUP_SIZE));
    OPS_CHECK_PTR_NULL(targetGroupSize, false);

    const auto* scale = attrs->GetAttrPointer<float>(static_cast<size_t>(ATTR_INDEX::SCALE));
    OPS_CHECK_PTR_NULL(scale, false);

    const auto* alpha = attrs->GetAttrPointer<float>(static_cast<size_t>(ATTR_INDEX::ALPHA));
    OPS_CHECK_PTR_NULL(alpha, false);
    OPS_CHECK(*alpha <= 0.0f, OPS_LOG_E("", "tiling failed alpha must be greater than 0.\n"), return false);
    const auto* winLeft = attrs->GetAttrPointer<int32_t>(static_cast<size_t>(ATTR_INDEX::WINDOW_SIZE_LEFT));
    const auto* winRight = attrs->GetAttrPointer<int32_t>(static_cast<size_t>(ATTR_INDEX::WINDOW_SIZE_RIGHT));
    const auto* deterministic = attrs->GetAttrPointer<bool>(static_cast<size_t>(ATTR_INDEX::DETERMINISTIC));
    OPS_CHECK_PTR_NULL(winLeft, false);
    OPS_CHECK_PTR_NULL(winRight, false);
    OPS_CHECK_PTR_NULL(deterministic, false);
    OPS_CHECK(*winLeft < -1 || *winRight < -1,
              OPS_LOG_E("", "tiling failed window sizes must be >= -1.\n"), return false);

    tilingData.set_maxSeqLenQ(*maxSeqQ);
    tilingData.set_maxSeqLenK(*maxSeqK);
    tilingData.set_targetGroupSize(*targetGroupSize);
    tilingData.set_scale(*scale);
    tilingData.set_alpha(*alpha);
    tilingData.set_windowSizeLeft(*winLeft);
    tilingData.set_windowSizeRight(*winRight);
    return true;
}

static bool TilingKeySet(gert::TilingContext* context, HstuForwardV2TilingData& tilingData)
{
    const auto* numContext = context->GetInputTensor(static_cast<size_t>(IN_INDEX::NUM_CONTEXT));
    const auto* numTarget = context->GetInputTensor(static_cast<size_t>(IN_INDEX::NUM_TARGET));
    const auto* rab = context->GetInputTensor(static_cast<size_t>(IN_INDEX::RAB));
    const auto* attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, false);
    const auto* maskType = attrs->GetAttrPointer<int32_t>(static_cast<size_t>(ATTR_INDEX::MASK_TYPE));
    const auto* deterministic = attrs->GetAttrPointer<bool>(static_cast<size_t>(ATTR_INDEX::DETERMINISTIC));
    OPS_CHECK_PTR_NULL(maskType, false);
    OPS_CHECK_PTR_NULL(deterministic, false);
    bool hasRab = (nullptr != rab);
    uint32_t rabHeads = 0;
    if (hasRab) {
        const auto& rabShape = rab->GetStorageShape();
        OPS_CHECK(rabShape.GetDimNum() != 4, OPS_LOG_E("", "forward RAB must be 4D.\n"), return false);
        rabHeads = static_cast<uint32_t>(rabShape.GetDim(static_cast<size_t>(DIM_INDEX::ONE)));
        OPS_CHECK(rabHeads != 1 && rabHeads != tilingData.get_heads(),
                  OPS_LOG_E("", "forward RAB heads must be 1 or equal to Q heads.\n"), return false);
        OPS_CHECK(rabShape.GetDim(static_cast<size_t>(DIM_INDEX::ZERO)) != tilingData.get_batch() ||
                      rabShape.GetDim(static_cast<size_t>(DIM_INDEX::TWO)) != tilingData.get_maxSeqLenQ() ||
                      rabShape.GetDim(static_cast<size_t>(DIM_INDEX::THREE)) != tilingData.get_maxSeqLenK(),
                  OPS_LOG_E("", "forward RAB geometry mismatch.\n"), return false);
    }
    tilingData.set_rabHeads(rabHeads);
    auto maskCnt = context->GetDynamicInputTensor(static_cast<size_t>(IN_INDEX::SPARSE_INFO), 0);
    auto rowBounds = context->GetDynamicInputTensor(static_cast<size_t>(IN_INDEX::SPARSE_INFO), 1);
    auto descriptor = context->GetDynamicInputTensor(static_cast<size_t>(IN_INDEX::SPARSE_INFO), 6);
    auto directTag = context->GetDynamicInputTensor(static_cast<size_t>(IN_INDEX::SPARSE_INFO), 7);
    auto bitmap = context->GetDynamicInputTensor(static_cast<size_t>(IN_INDEX::SPARSE_INFO), 8);
    const bool isArbitrary = maskCnt != nullptr && maskCnt->GetShapeSize() > 0;
    const auto winLeft = tilingData.get_windowSizeLeft();
    const auto winRight = tilingData.get_windowSizeRight();
    const bool isLocal = !isArbitrary && (winLeft >= 0 || winRight > 0);
    const bool isCausal = !isArbitrary && !isLocal && *maskType == 0;
    const bool isContext = !isArbitrary && numContext != nullptr;
    const bool isTarget = !isArbitrary && numTarget != nullptr;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAic());
    const auto runtimeTile = ops_rec::hstu::GetFwdRuntimeTile(
        tilingData.get_maxSeqLenQ(), tilingData.get_maxSeqLenK(), tilingData.get_batch(), tilingData.get_heads(),
        tilingData.get_dimQK(), tilingData.get_dimV(), coreNum, GetCoreMemSize(platform), true);
    tilingData.set_blockM(runtimeTile.l1.m);
    tilingData.set_blockN(runtimeTile.l1.n);
    uint32_t maskHeads = 0;
    if (isArbitrary) {
        for (int i = 0; i < 9; ++i) {
            OPS_CHECK(context->GetDynamicInputTensor(static_cast<size_t>(IN_INDEX::SPARSE_INFO), i) == nullptr,
                      OPS_LOG_E("", "forward sparse_info requires nine packed tensors.\n"), return false);
        }
        OPS_CHECK(directTag->GetStorageShape().GetDimNum() != 1,
                  OPS_LOG_E("", "forward sparse_info must use direct descriptors.\n"), return false);
        const auto& countShape = maskCnt->GetStorageShape();
        OPS_CHECK(countShape.GetDimNum() != 3 || countShape.GetDim(0) != tilingData.get_batch(),
                  OPS_LOG_E("", "forward packed count geometry mismatch.\n"), return false);
        maskHeads = static_cast<uint32_t>(countShape.GetDim(1));
        OPS_CHECK(maskHeads != 1 && maskHeads != tilingData.get_heads(),
                  OPS_LOG_E("", "forward packed heads must be 1 or equal to Q heads.\n"), return false);
        const auto tile = runtimeTile.l1;
        const uint64_t wordsPerBitmapRow = tile.n / 32U;
        const uint64_t nq = (static_cast<uint64_t>(tilingData.get_maxSeqLenQ()) + tile.m - 1) / tile.m;
        const uint64_t nk = (static_cast<uint64_t>(tilingData.get_maxSeqLenK()) + tile.n - 1) / tile.n;
        const uint64_t outer = static_cast<uint64_t>(tilingData.get_batch()) * maskHeads * nq;
        OPS_CHECK(static_cast<uint64_t>(descriptor->GetShapeSize()) < outer * nk,
                  OPS_LOG_E("", "forward descriptor is too small.\n"), return false);
        OPS_CHECK(static_cast<uint64_t>(bitmap->GetShapeSize()) < outer * nk * tile.m * wordsPerBitmapRow,
                  OPS_LOG_E("", "forward bitmap is too small.\n"), return false);
        OPS_CHECK(static_cast<uint64_t>(rowBounds->GetShapeSize()) < outer * 2,
                  OPS_LOG_E("", "forward row_bounds is too small.\n"), return false);
    }
    tilingData.set_maskHeads(maskHeads);

    auto dimQK = tilingData.get_dimQK();
    auto dimV = tilingData.get_dimV();

    const uint32_t tilingDim = ops_rec::hstu::GetFwdTileSize(
        std::max(tilingData.get_dimQK(), tilingData.get_dimV())).k;
    const uint64_t outerTasks = static_cast<uint64_t>(tilingData.get_batch()) * tilingData.get_heads() *
        ((tilingData.get_maxSeqLenQ() + tilingData.get_blockM() - 1U) / tilingData.get_blockM());
    const uint64_t innerBlocks =
        (tilingData.get_maxSeqLenK() + tilingData.get_blockN() - 1U) / tilingData.get_blockN();
    const bool enableStreamK = !*deterministic && outerTasks < coreNum && innerBlocks > 1U;
    ASCENDC_TPL_SEL_PARAM(context, hasRab, isLocal, isCausal, isContext, isTarget, isArbitrary,
                          enableStreamK, tilingDim);
    return true;
}

static void ShareMemorySet(gert::TilingContext* context, HstuForwardV2TilingData& tilingData)
{
    const uint32_t BLOCK_N = tilingData.get_blockN();

    auto totalKSeqLens = tilingData.get_batch() * tilingData.get_heads() * tilingData.get_maxSeqLenQ();
    auto totalKBlockCnt = (totalKSeqLens + BLOCK_N - 1) / BLOCK_N;
    int64_t shareMemorySizes = totalKBlockCnt * sizeof(uint32_t);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t systemWorkspaceSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = shareMemorySizes + systemWorkspaceSize;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    HstuForwardV2TilingData tilingData;

    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    OPS_CHECK(!ParseShape(context, tilingData), OPS_LOG_E("", "parse shape failed.\n"), return ge::GRAPH_FAILED);
    OPS_CHECK(!ParseAttr(context, tilingData), OPS_LOG_E("", "parse attr failed.\n"), return ge::GRAPH_FAILED);
    OPS_CHECK(!TilingKeySet(context, tilingData), OPS_LOG_E("", "tiling keyset failed.\n"), return ge::GRAPH_FAILED);
    ShareMemorySet(context, tilingData);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAic();
    const auto* metadata = context->GetInputTensor(static_cast<size_t>(IN_INDEX::METADATA));
    const auto* deterministic = context->GetAttrs()->GetAttrPointer<bool>(
        static_cast<size_t>(ATTR_INDEX::DETERMINISTIC));
    OPS_CHECK_PTR_NULL(deterministic, return ge::GRAPH_FAILED);
    const size_t outerTasks = static_cast<size_t>(tilingData.get_batch()) * tilingData.get_heads() *
        ((tilingData.get_maxSeqLenQ() + tilingData.get_blockM() - 1U) / tilingData.get_blockM());
    const size_t innerBlocks = (tilingData.get_maxSeqLenK() + tilingData.get_blockN() - 1U) /
                               tilingData.get_blockN();
    if (!*deterministic && outerTasks < coreNum && innerBlocks > 1U) {
        // Stream-K metadata is generated against the complete AIC topology.
        // Keep that launch width even when there are fewer pair tasks: the
        // evenly-spaced non-empty records may otherwise refer to AIC ids that
        // were removed by a host-side min(pairTasks) clamp.
    } else if (metadata == nullptr) {
        const auto tile = ops_rec::hstu::GetFwdTileSize(tilingData.get_dimQK());
        const size_t outerBlocks = static_cast<size_t>(tilingData.get_batch()) * tilingData.get_heads() *
                                   ((tilingData.get_maxSeqLenQ() + tilingData.get_blockM() - 1U) /
                                    tilingData.get_blockM());
        coreNum = std::max<size_t>(1U, std::min(coreNum, outerBlocks));
    }
    context->SetBlockDim(coreNum);

    OPS_LOG_E_IF_NULL("RawTilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {
static ge::graphStatus InferAttenOutputShape(gert::InferShapeContext* context)
{
    const gert::Shape* queryShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::Q));
    OPS_CHECK_PTR_NULL(queryShape, return ge::GRAPH_FAILED);

    gert::Shape* attenOutputShape = context->GetOutputShape(static_cast<size_t>(OUT_INDEX::ATTN_OUTPUT));
    OPS_CHECK_PTR_NULL(attenOutputShape, return ge::GRAPH_FAILED);
    *attenOutputShape = *queryShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    auto result = InferAttenOutputShape(context);
    OPS_CHECK(result == ge::GRAPH_FAILED, OPS_LOG_E("", "InferAttenOutputShape failed.\n"), return ge::GRAPH_FAILED);

    return result;
}
}  // namespace ge

namespace ops {
class HstuForwardV2 : public OpDef {
public:
    explicit HstuForwardV2(const char* name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).Follow("q", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).Follow("q", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("mask").ParamType(OPTIONAL).Follow("q", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("rab").ParamType(OPTIONAL).Follow("q", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("seq_offset_q")  // 规避optional类型无法正常生成json文件的问题
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("seq_offset_k")
            .ParamType(OPTIONAL)
            .Follow("seq_offset_q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("num_context")
            .ParamType(OPTIONAL)
            .Follow("seq_offset_q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("num_target")
            .ParamType(OPTIONAL)
            .Follow("seq_offset_q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        // 可选: flash_attn_metadata 分核输出(int32,HEAD+FA+FD 布局)。未传 → kernel 收到 nullptr →
        // 旧设备现算分核(零回归)
        this->Input("metadata").ParamType(OPTIONAL).DataType({ge::DT_INT32, ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("output_accum").ParamType(OPTIONAL).DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("sparse_info").ParamType(DYNAMIC).DataType({ge::DT_INT32, ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("attn_output").ParamType(REQUIRED).Follow("q", FollowType::DTYPE).FormatList({ge::FORMAT_ND});

        this->Attr("max_seqlen_q").Int();
        this->Attr("max_seqlen_k").Int();
        this->Attr("silu_scale").Float();
        this->Attr("target_group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("alpha").Float();
        this->Attr("mask_type").AttrType(OPTIONAL).Int(0);
        this->Attr("window_size_left").AttrType(OPTIONAL).Int(-1);
        this->Attr("window_size_right").AttrType(OPTIONAL).Int(-1);
        this->Attr("deterministic").AttrType(OPTIONAL).Bool(true);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(HstuForwardV2);
}  // namespace ops
