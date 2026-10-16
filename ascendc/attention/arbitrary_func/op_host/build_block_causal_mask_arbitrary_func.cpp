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

#include "build_block_causal_mask_arbitrary_func_tiling.h"

#include <climits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

constexpr uint32_t ARBITRARY_FUNC_DIM = 4;
constexpr uint32_t N_FUNC = 2;

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* offsetsShape = context->GetInputShape(0);
    const gert::StorageShape* blockSizesShape = context->GetInputShape(1);
    OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("blockSizesShape", blockSizesShape, return ge::GRAPH_FAILED);

    const gert::Shape& offsetsStorage = offsetsShape->GetStorageShape();
    const gert::Shape& blockSizesStorage = blockSizesShape->GetStorageShape();
    if (offsetsStorage.GetDimNum() != 1 || offsetsStorage.GetDim(0) < 1) {
        OPS_LOG_E("[ERROR]", "offsets must have shape [batch_size + 1]");
        return ge::GRAPH_FAILED;
    }
    const int32_t batchSize = static_cast<int32_t>(offsetsStorage.GetDim(0)) - 1;
    if (batchSize < 0) {
        OPS_LOG_E("[ERROR]", "offsets.size(0) must be >= 1, i.e. offsets shape must be [batch_size+1] with "
                             "batch_size >= 0");
        return ge::GRAPH_FAILED;
    }
    if (blockSizesStorage.GetDimNum() != 1 || blockSizesStorage.GetDim(0) != batchSize) {
        OPS_LOG_E("[ERROR]", "block_sizes must have shape [batch_size], actual dim0 = ", blockSizesStorage.GetDim(0),
                  ", batch_size = ", batchSize);
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
    if (blockDim == 0) {
        OPS_LOG_E("[ERROR]", "ai core num is zero.");
        return ge::GRAPH_FAILED;
    }

    BuildBlockCausalMaskArbitraryFuncTilingData tiling;
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
class BuildBlockCausalMaskArbitraryFunc : public OpDef {
public:
    explicit BuildBlockCausalMaskArbitraryFunc(const char* name) : OpDef(name)
    {
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("blockSizes")
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

OP_ADD(BuildBlockCausalMaskArbitraryFunc);
}  // namespace ops
