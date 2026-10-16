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

#include "build_causal_mask_arbitrary_func_tiling.h"

#include <climits>
#include "ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* offsetsShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);

    const gert::Shape& offsetsStorage = offsetsShape->GetStorageShape();
    if (offsetsStorage.GetDimNum() != 1 || offsetsStorage.GetDim(0) < 1) {
        OPS_LOG_E("[ERROR]", "offsets must have shape [batch_size + 1]");
        return ge::GRAPH_FAILED;
    }
    int32_t batchSize = static_cast<int32_t>(offsetsStorage.GetDim(0)) - 1;
    if (batchSize < 0) {
        OPS_LOG_E("[ERROR]", "offsets.size(0) must be >= 1, i.e. offsets shape must be [batch_size+1] with "
                             "batch_size >= 0");
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    const int64_t* totalSeqLenPtr = attrs->GetAttrPointer<int64_t>(0);
    OPS_LOG_E_IF_NULL("totalSeqLenPtr", totalSeqLenPtr, return ge::GRAPH_FAILED);
    if (*totalSeqLenPtr < 0 || *totalSeqLenPtr > INT32_MAX) {
        OPS_LOG_E("[ERROR]", "total_seqlen must fit non-negative int32");
        return ge::GRAPH_FAILED;
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim = static_cast<uint32_t>(ascendPlatform.GetCoreNumAiv());
    OPS_LOG_E_IF(blockDim == 0, context, return ge::GRAPH_FAILED, "AI Vector core count is zero");

    BuildCausalMaskArbitraryFuncTilingData tiling;
    tiling.set_totalSeqLen(static_cast<int32_t>(*totalSeqLenPtr));
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
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    const int64_t* totalSeqLenPtr = attrs->GetAttrPointer<int64_t>(0);
    OPS_LOG_E_IF_NULL("totalSeqLenPtr", totalSeqLenPtr, return ge::GRAPH_FAILED);
    int64_t totalSeqLen = *totalSeqLenPtr;
    OPS_LOG_E_IF(totalSeqLen < 0, context, return ge::GRAPH_FAILED, "total_seqlen must be non-negative");

    gert::Shape* arbitraryFuncShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("arbitraryFuncShape", arbitraryFuncShape, return ge::GRAPH_FAILED);
    arbitraryFuncShape->SetDimNum(4);
    arbitraryFuncShape->SetDim(0, 1);
    arbitraryFuncShape->SetDim(1, 1);
    arbitraryFuncShape->SetDim(2, totalSeqLen);
    arbitraryFuncShape->SetDim(3, 2);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class BuildCausalMaskArbitraryFunc : public OpDef {
public:
    explicit BuildCausalMaskArbitraryFunc(const char* name) : OpDef(name)
    {
        this->Input("offsets")
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

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(BuildCausalMaskArbitraryFunc);
}  // namespace ops
