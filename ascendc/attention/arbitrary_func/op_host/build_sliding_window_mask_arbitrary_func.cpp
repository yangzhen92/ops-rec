/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#include <climits>
#include "ops_log.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr int32_t BUILDSLIDINGWINDOWMASKARBITRARYFUNC_N_FUNC = 2;
BEGIN_TILING_DATA_DEF(BuildSlidingWindowMaskArbitraryFuncTilingData)
TILING_DATA_FIELD_DEF(int32_t, totalSeqLen);
TILING_DATA_FIELD_DEF(int32_t, batchSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BuildSlidingWindowMaskArbitraryFunc, BuildSlidingWindowMaskArbitraryFuncTilingData)

static ge::graphStatus BuildSlidingWindowMaskArbitraryFuncTiling(gert::TilingContext* context)
{
    const auto* offsetsShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);
    const auto* windowSizesShape = context->GetInputShape(1);
    OPS_LOG_E_IF_NULL("windowSizesShape", windowSizesShape, return ge::GRAPH_FAILED);
    const auto& offsets = offsetsShape->GetStorageShape();
    OPS_LOG_E_IF(offsets.GetDimNum() != 1 || offsets.GetDim(0) < 1, context, return ge::GRAPH_FAILED,
                 "offsets must have shape [batch_size + 1]");
    const int32_t batchSize = static_cast<int32_t>(offsets.GetDim(0)) - 1;
    const auto& windowSizesStorage = windowSizesShape->GetStorageShape();
    OPS_LOG_E_IF(windowSizesStorage.GetDimNum() != 1 || windowSizesStorage.GetDim(0) != batchSize, context,
                 return ge::GRAPH_FAILED, "window_sizes must have shape [batch_size]");
    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    const auto* totalSeqLen = attrs->GetAttrPointer<int64_t>(0);
    OPS_LOG_E_IF_NULL("totalSeqLen", totalSeqLen, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF(*totalSeqLen < 0 || *totalSeqLen > INT32_MAX, context, return ge::GRAPH_FAILED,
                 "total_seqlen must fit non-negative int32");
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t blockDim = static_cast<uint32_t>(platform.GetCoreNumAiv());
    OPS_LOG_E_IF(blockDim == 0, context, return ge::GRAPH_FAILED, "AI Vector core count is zero");
    BuildSlidingWindowMaskArbitraryFuncTilingData tiling;
    tiling.set_totalSeqLen(static_cast<int32_t>(*totalSeqLen));
    tiling.set_batchSize(batchSize);
    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BuildSlidingWindowMaskArbitraryFuncInferShape(gert::InferShapeContext* context)
{
    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    const auto* totalSeqLen = attrs->GetAttrPointer<int64_t>(0);
    OPS_LOG_E_IF_NULL("totalSeqLen", totalSeqLen, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF(*totalSeqLen < 0, context, return ge::GRAPH_FAILED, "total_seqlen must be non-negative");
    auto* output = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("output", output, return ge::GRAPH_FAILED);
    output->SetDimNum(4);
    output->SetDim(0, 1);
    output->SetDim(1, 1);
    output->SetDim(2, *totalSeqLen);
    output->SetDim(3, BUILDSLIDINGWINDOWMASKARBITRARYFUNC_N_FUNC);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class BuildSlidingWindowMaskArbitraryFunc : public OpDef {
public:
    explicit BuildSlidingWindowMaskArbitraryFunc(const char* name) : OpDef(name)
    {
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("windowSizes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("totalSeqLen").AttrType(REQUIRED).Int(0);
        this->Output("arbitraryFunc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(optiling::BuildSlidingWindowMaskArbitraryFuncInferShape);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");
        this->AICore().SetTiling(optiling::BuildSlidingWindowMaskArbitraryFuncTiling);
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(BuildSlidingWindowMaskArbitraryFunc);
}  // namespace ops
