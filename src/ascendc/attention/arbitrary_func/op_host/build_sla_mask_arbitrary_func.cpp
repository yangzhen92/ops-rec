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

#include "build_sla_mask_arbitrary_func_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

constexpr uint32_t ARBITRARY_FUNC_DIM = 4;
constexpr uint32_t N_FUNC = 4;

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* offsetsShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);
    const gert::StorageShape* numTargetsShape = context->GetInputShape(1);
    OPS_LOG_E_IF_NULL("numTargetsShape", numTargetsShape, return ge::GRAPH_FAILED);

    if (offsetsShape->GetStorageShape().GetDimNum() != 1) {
        OPS_LOG_E("[ERROR]", "offsets must be 1D");
        return ge::GRAPH_FAILED;
    }
    if (numTargetsShape->GetStorageShape().GetDimNum() != 1) {
        OPS_LOG_E("[ERROR]", "numTargets must be 1D");
        return ge::GRAPH_FAILED;
    }

    int32_t batchSize = static_cast<int32_t>(offsetsShape->GetStorageShape().GetDim(0)) - 1;
    if (batchSize < 0) {
        OPS_LOG_E("[ERROR]", "offsets.size(0) must be >= 1, i.e. offsets shape must be [batch_size+1] with "
                             "batch_size >= 0");
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    const int64_t* slaK1Ptr = attrs->GetAttrPointer<int64_t>(0);
    OPS_LOG_E_IF_NULL("slaK1Ptr", slaK1Ptr, return ge::GRAPH_FAILED);
    const int64_t* slaK2Ptr = attrs->GetAttrPointer<int64_t>(1);
    OPS_LOG_E_IF_NULL("slaK2Ptr", slaK2Ptr, return ge::GRAPH_FAILED);
    const int64_t* contextualSeqLenPtr = attrs->GetAttrPointer<int64_t>(2);
    OPS_LOG_E_IF_NULL("contextualSeqLenPtr", contextualSeqLenPtr, return ge::GRAPH_FAILED);
    const int64_t* totalSeqLenPtr = attrs->GetAttrPointer<int64_t>(3);
    OPS_LOG_E_IF_NULL("totalSeqLenPtr", totalSeqLenPtr, return ge::GRAPH_FAILED);
    if (*slaK1Ptr < 0 || *slaK1Ptr > INT32_MAX || *slaK2Ptr < 0 || *slaK2Ptr > INT32_MAX || *contextualSeqLenPtr < 0 ||
        *contextualSeqLenPtr > INT32_MAX || *totalSeqLenPtr < 0 || *totalSeqLenPtr > INT32_MAX) {
        OPS_LOG_E("[ERROR]",
                  "slaK1/slaK2/contextualSeqLen/totalSeqLen must be in [0, INT32_MAX] to fit in int32 tiling");
        return ge::GRAPH_FAILED;
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim = static_cast<uint32_t>(ascendPlatform.GetCoreNumAiv());
    if (blockDim == 0) {
        OPS_LOG_E("[ERROR]", "ai core num is zero.");
        return ge::GRAPH_FAILED;
    }

    BuildSlaMaskArbitraryFuncTilingData tiling;
    tiling.set_slaK1(static_cast<int32_t>(*slaK1Ptr));
    tiling.set_slaK2(static_cast<int32_t>(*slaK2Ptr));
    tiling.set_contextualSeqLen(static_cast<int32_t>(*contextualSeqLenPtr));
    tiling.set_totalSeqLen(static_cast<int32_t>(*totalSeqLenPtr));
    tiling.set_batchSize(batchSize);

    context->SetBlockDim(blockDim);
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
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
    const int64_t* totalSeqLenPtr = attrs->GetAttrPointer<int64_t>(3);
    OPS_LOG_E_IF_NULL("totalSeqLenPtr", totalSeqLenPtr, return ge::GRAPH_FAILED);
    int64_t totalSeqLen = *totalSeqLenPtr;
    if (totalSeqLen < 0) {
        OPS_LOG_E("[ERROR]", "total_seqlen must be non-negative");
        return ge::GRAPH_FAILED;
    }

    gert::Shape* arbitraryFuncShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("arbitraryFuncShape", arbitraryFuncShape, return ge::GRAPH_FAILED);
    arbitraryFuncShape->SetDimNum(ARBITRARY_FUNC_DIM);
    arbitraryFuncShape->SetDim(0, 1);
    arbitraryFuncShape->SetDim(1, 1);
    arbitraryFuncShape->SetDim(2, totalSeqLen);
    arbitraryFuncShape->SetDim(3, N_FUNC);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class BuildSlaMaskArbitraryFunc : public OpDef {
public:
    explicit BuildSlaMaskArbitraryFunc(const char* name) : OpDef(name)
    {
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("numTargets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("slaK1").AttrType(REQUIRED).Int(1);
        this->Attr("slaK2").AttrType(REQUIRED).Int(2);
        this->Attr("contextualSeqLen").AttrType(REQUIRED).Int(0);
        this->Attr("totalSeqLen").AttrType(REQUIRED).Int(0);
        this->Output("arbitraryFunc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(BuildSlaMaskArbitraryFunc);
}  // namespace ops
