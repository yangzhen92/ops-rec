/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "fa_infer_with_arbitrary_func_tiling.h"
#include "fa_tiling.h"

#include <cstdint>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"

namespace {
constexpr size_t INPUT_INDEX_Q = 0;
constexpr size_t INPUT_INDEX_K = 1;
constexpr size_t INPUT_INDEX_V = 2;
constexpr uint32_t TILE_N = 128;
constexpr int DCACHE_SIZE = 214 * 1024;
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    FaInferWithArbitraryFuncTilingData tiling;
    // 限定TND格式[seqlen, headNum, headDim]，batch为1
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("qShape", context->GetInputShape(INPUT_INDEX_Q), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("kShape", context->GetInputShape(INPUT_INDEX_K), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("vShape", context->GetInputShape(INPUT_INDEX_V), return ge::GRAPH_FAILED);
    int64_t qSeqlen = context->GetInputShape(INPUT_INDEX_Q)->GetStorageShape().GetDim(0);
    int64_t qHeads = context->GetInputShape(INPUT_INDEX_Q)->GetStorageShape().GetDim(1);
    int64_t embed = context->GetInputShape(INPUT_INDEX_Q)->GetStorageShape().GetDim(2);
    int64_t kvSeqlen = context->GetInputShape(INPUT_INDEX_K)->GetStorageShape().GetDim(0);
    int64_t kvHeads = context->GetInputShape(INPUT_INDEX_K)->GetStorageShape().GetDim(1);
    int64_t kEmbed = context->GetInputShape(INPUT_INDEX_K)->GetStorageShape().GetDim(2);
    int64_t vEmbed = context->GetInputShape(INPUT_INDEX_V)->GetStorageShape().GetDim(2);
    float scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * embed));

    OPS_LOG_E_IF(kvHeads <= 0, context, return ge::GRAPH_FAILED, "[ERROR]kvHeads must be greater than 0.");
    OPS_LOG_E_IF(qHeads % kvHeads != 0, context, return ge::GRAPH_FAILED,
                 "[ERROR]qHeads must be a multiple of kvHeads, qHeads: %lld, kvHeads: %lld.", qHeads, kvHeads);
    OPS_LOG_E_IF(embed != 128, context, return ge::GRAPH_FAILED, "[ERROR]embed must be 128.");
    OPS_LOG_E_IF(kEmbed != embed || vEmbed != embed, context, return ge::GRAPH_FAILED,
                 "[ERROR]embed of q, k, v must be the same, qEmbed: %lld, kEmbed: %lld, vEmbed: %lld.", embed, kEmbed,
                 vEmbed);

    FAInferTiling::FAInfo faInfo;
    faInfo.batchSize = 1;
    faInfo.numOfHeads = qHeads;
    faInfo.numOfKVHeads = kvHeads;
    faInfo.seqSize = qSeqlen;
    faInfo.seqInnerSize = kvSeqlen;
    faInfo.headSize = embed;
    faInfo.scaleValue = scaleValue;
    faInfo.blockSize = TILE_N;  // 默认值 tileshape 128*128
    faInfo.maxBlockNumPerBatch = (kvSeqlen + faInfo.blockSize - 1) / faInfo.blockSize;

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto aicCoreNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = aicCoreNum;

    FAInferTiling::GetFATilingParam(faInfo, blockDim, tiling);
    context->SetDynUBufSize(DCACHE_SIZE);
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
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const gert::Shape* q_shape = context->GetInputShape(INPUT_INDEX_Q);
    OPS_LOG_E_IF_NULL("q_shape", q_shape, return ge::GRAPH_FAILED);

    const auto out_shape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("out_shape", out_shape, return ge::GRAPH_FAILED);
    out_shape->SetDimNum(3);
    out_shape->SetDim(0, q_shape->GetDim(0));
    out_shape->SetDim(1, q_shape->GetDim(1));
    out_shape->SetDim(2, q_shape->GetDim(2));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    auto qDataType = context->GetInputDataType(INPUT_INDEX_Q);
    context->SetOutputDataType(0, qDataType);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class FaInferWithArbitraryFunc : public OpDef {
public:
    explicit FaInferWithArbitraryFunc(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("arbitrary_func")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mask_block_cnt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mask_block_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mask_block_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("full_block_cnt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("full_block_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("full_block_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(FaInferWithArbitraryFunc);
}  // namespace ops
