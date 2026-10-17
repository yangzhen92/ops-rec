/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "build_gist_mask_arbitrary_func_tiling.h"

#include <climits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

constexpr uint32_t ARBITRARY_FUNC_DIM = 4;
constexpr uint32_t N_FUNC = 4;

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* offsetsShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);
    const auto& offsets = offsetsShape->GetStorageShape();
    OPS_LOG_E_IF(offsets.GetDimNum() != 1 || offsets.GetDim(0) < 1, context, return ge::GRAPH_FAILED,
                 "offsets must have shape [batch_size + 1]");
    const int32_t batchSize = static_cast<int32_t>(offsets.GetDim(0)) - 1;
    for (int32_t index = 1; index <= 3; ++index) {
        const auto* cfgShape = context->GetInputShape(index);
        OPS_LOG_E_IF_NULL("cfgShape", cfgShape, return ge::GRAPH_FAILED);
        const auto& storage = cfgShape->GetStorageShape();
        OPS_LOG_E_IF(storage.GetDimNum() != 1 || storage.GetDim(0) != batchSize, context, return ge::GRAPH_FAILED,
                     "gist mask config tensors must have shape [batch_size]");
    }

    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    const auto* totalSeqLen = attrs->GetAttrPointer<int64_t>(0);
    OPS_LOG_E_IF_NULL("totalSeqLen", totalSeqLen, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF(*totalSeqLen < 0 || *totalSeqLen > INT32_MAX, context, return ge::GRAPH_FAILED,
                 "total_seqlen must fit non-negative int32");
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t blockDim = static_cast<uint32_t>(platform.GetCoreNumAiv());
    OPS_LOG_E_IF(blockDim == 0, context, return ge::GRAPH_FAILED, "AI Vector core count is zero");

    BuildGistMaskArbitraryFuncTilingData tiling;
    tiling.set_totalSeqLen(static_cast<int32_t>(*totalSeqLen));
    tiling.set_batchSize(batchSize);
    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    const auto* totalSeqLen = attrs->GetAttrPointer<int64_t>(0);
    OPS_LOG_E_IF_NULL("totalSeqLen", totalSeqLen, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF(*totalSeqLen < 0, context, return ge::GRAPH_FAILED, "total_seqlen must be non-negative");
    auto* output = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("output", output, return ge::GRAPH_FAILED);
    output->SetDimNum(ARBITRARY_FUNC_DIM);
    output->SetDim(0, 1);
    output->SetDim(1, 1);
    output->SetDim(2, *totalSeqLen);
    output->SetDim(3, N_FUNC);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class BuildGistMaskArbitraryFunc : public OpDef {
public:
    explicit BuildGistMaskArbitraryFunc(const char* name) : OpDef(name)
    {
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("preDocLens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gistStarts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gistEnds")
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
        this->SetInferShape(ge::InferShape);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(BuildGistMaskArbitraryFunc);
}  // namespace ops
