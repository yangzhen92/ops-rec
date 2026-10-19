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

#include "build_hstu_mask_arbitrary_func_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* offsetsShape = context->GetInputShape(0);
    const gert::StorageShape* ctxLensShape = context->GetInputShape(1);
    const gert::StorageShape* tgtLensShape = context->GetInputShape(2);
    OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("ctxLensShape", ctxLensShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("tgtLensShape", tgtLensShape, return ge::GRAPH_FAILED);

    const gert::Shape& offsetsStorage = offsetsShape->GetStorageShape();
    const gert::Shape& ctxLensStorage = ctxLensShape->GetStorageShape();
    const gert::Shape& tgtLensStorage = tgtLensShape->GetStorageShape();
    if (offsetsStorage.GetDimNum() != 1 || ctxLensStorage.GetDimNum() != 1 || tgtLensStorage.GetDimNum() != 1) {
        OPS_LOG_E("[ERROR]", "offsets / ctx_lens / tgt_lens must all be 1D, actual ranks: ", offsetsStorage.GetDimNum(),
                  " / ", ctxLensStorage.GetDimNum(), " / ", tgtLensStorage.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    int32_t batchSize = static_cast<int32_t>(offsetsStorage.GetDim(0)) - 1;
    if (batchSize < 0) {
        OPS_LOG_E("[ERROR]", "offsets.size(0) must be >= 1, i.e. offsets shape must be [batch_size+1] with "
                             "batch_size >= 0");
        return ge::GRAPH_FAILED;
    }
    if (ctxLensStorage.GetDim(0) != batchSize) {
        OPS_LOG_E("[ERROR]", "ctx_lens must have shape [batch_size], actual dim0 = ", ctxLensStorage.GetDim(0),
                  ", batch_size = ", batchSize);
        return ge::GRAPH_FAILED;
    }
    if (tgtLensStorage.GetDim(0) != batchSize) {
        OPS_LOG_E("[ERROR]", "tgt_lens must have shape [batch_size], actual dim0 = ", tgtLensStorage.GetDim(0),
                  ", batch_size = ", batchSize);
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    const int64_t* targetGroupSizePtr = attrs->GetAttrPointer<int64_t>(0);
    OPS_LOG_E_IF_NULL("targetGroupSizePtr", targetGroupSizePtr, return ge::GRAPH_FAILED);
    const int64_t* totalSeqLenPtr = attrs->GetAttrPointer<int64_t>(1);
    OPS_LOG_E_IF_NULL("totalSeqLenPtr", totalSeqLenPtr, return ge::GRAPH_FAILED);
    if (*targetGroupSizePtr <= 0) {
        OPS_LOG_E("[ERROR]", "targetGroupSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim = static_cast<uint32_t>(ascendPlatform.GetCoreNumAiv());
    if (blockDim == 0) {
        OPS_LOG_E("[ERROR]", "ai core num is zero.");
        return ge::GRAPH_FAILED;
    }

    BuildHstuMaskArbitraryFuncTilingData tiling;
    tiling.set_totalSeqLen(static_cast<int32_t>(*totalSeqLenPtr));
    tiling.set_targetGroupSize(static_cast<int32_t>(*targetGroupSizePtr));
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
    const int64_t* totalSeqLenPtr = attrs->GetAttrPointer<int64_t>(1);
    OPS_LOG_E_IF_NULL("totalSeqLenPtr", totalSeqLenPtr, return ge::GRAPH_FAILED);
    int64_t totalSeqLen = *totalSeqLenPtr;
    if (totalSeqLen < 0) {
        OPS_LOG_E("[ERROR]", "total_seqlen must be non-negative");
        return ge::GRAPH_FAILED;
    }

    gert::Shape* arbitraryFuncShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("arbitraryFuncShape", arbitraryFuncShape, return ge::GRAPH_FAILED);
    arbitraryFuncShape->SetDimNum(4);
    arbitraryFuncShape->SetDim(0, 1);
    arbitraryFuncShape->SetDim(1, 1);
    arbitraryFuncShape->SetDim(2, totalSeqLen);
    arbitraryFuncShape->SetDim(3, 4);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class BuildHstuMaskArbitraryFunc : public OpDef {
public:
    explicit BuildHstuMaskArbitraryFunc(const char* name) : OpDef(name)
    {
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ctxLens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tgtLens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("targetGroupSize").AttrType(REQUIRED).Int(1);
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

#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950", aicore_config);
#endif
    }
};

OP_ADD(BuildHstuMaskArbitraryFunc);
}  // namespace ops
