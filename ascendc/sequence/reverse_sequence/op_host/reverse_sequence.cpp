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
#include "reverse_sequence_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "ops_log.h"

namespace optiling {
// 常量定义
constexpr int32_t ALIGN_32 = 32;
constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM2 = 2;

constexpr int32_t PARAM_INPUT_INDEX = 0;
constexpr int32_t PARAM_SEQ_LENGTH_INDEX = 1;
constexpr int32_t OUTPUT_INDEX = 0;

constexpr int64_t DATA_BYTE_FP32 = 4;
constexpr int64_t DATA_BYTE_FP16 = 2;
constexpr int64_t DATA_BYTE_BF16 = 2;
constexpr int64_t DATA_BYTE_INT64 = 8;

constexpr uint32_t ROW_TILE_UB_BYTES = 64 * 1024;            // 行 tile 的 UB 预算
constexpr uint32_t DATA_COPY_PAD_MAX_BLOCK_COUNT = 4095;     // DataCopyPad blockCount 上限

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputShape", context->GetInputShape(PARAM_INPUT_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("seqLengthShape", context->GetInputShape(PARAM_SEQ_LENGTH_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("input", context->GetInputTensor(PARAM_INPUT_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("seqLength", context->GetInputTensor(PARAM_SEQ_LENGTH_INDEX), return ge::GRAPH_FAILED);

    // 获取输入形状和类型
    auto inputShape = context->GetInputShape(PARAM_INPUT_INDEX)->GetStorageShape();
    auto seqLengthShape = context->GetInputShape(PARAM_SEQ_LENGTH_INDEX)->GetStorageShape();

    OPS_CHECK(inputShape.GetDim(DIM0) != seqLengthShape.GetDim(DIM0),
              OPS_LOG_E("[ERROR]", "input shape[0] must equal to seq_lengths shape[0]"), return ge::GRAPH_FAILED);

    // Platform configuration
    size_t usrSize = 0;
    auto ascendCPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);

    // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
    size_t systemWorkspacesSize = ascendCPlatform.GetLibApiWorkSpaceSize();
    // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
    currentWorkspace[0] = usrSize + systemWorkspacesSize;
    size_t coreNum = ascendCPlatform.GetCoreNumAiv();

    // 一共要处理的dataDim维度的数据条数
    int64_t totalDataNum = inputShape.GetDim(DIM0) * inputShape.GetDim(DIM1);
    ge::DataType inputDataType = context->GetInputTensor(PARAM_INPUT_INDEX)->GetDataType();
    int64_t dataBytes = 0;
    if (inputDataType == ge::DataType::DT_FLOAT) {
        dataBytes = DATA_BYTE_FP32;
    } else if (inputDataType == ge::DataType::DT_FLOAT16) {
        dataBytes = DATA_BYTE_FP16;
    } else if (inputDataType == ge::DataType::DT_BF16) {
        dataBytes = DATA_BYTE_BF16;
    } else if (inputDataType == ge::DataType::DT_INT64) {
        dataBytes = DATA_BYTE_INT64;
    } else {
        OPS_LOG_E("", "invalid datatype, only support float/fp16/bf16/int64\n");
        return ge::GRAPH_FAILED;
    }

    OPS_CHECK(coreNum == 0, OPS_LOG_E("[ERROR]", "aiv core num == 0"), return ge::GRAPH_FAILED);

    auto dataDim = inputShape.GetDim(DIM2);

    // 空输入（batch/seq/data 任一维度为 0）时退化为单核空转，避免除零与零字节 UB 申请
    int64_t usedCoreNum = 1;
    int64_t handleTotalCount = 0;
    int64_t left = 0;
    uint32_t rowAlignCount = static_cast<uint32_t>(ALIGN_32 / dataBytes);  // 占位，保证 UB 申请非零
    uint32_t rowsPerTile = 1;
    if (totalDataNum > 0 && dataDim > 0) {
        // 小 shape 下裁剪 blockDim 到实际行数，避免启动大量空转核
        usedCoreNum = static_cast<int64_t>(coreNum);
        if (totalDataNum < usedCoreNum) {
            usedCoreNum = totalDataNum;
        }
        // 单个核总共要处理的数据条数，余数行由前 left 个核各多承担 1 行
        handleTotalCount = totalDataNum / usedCoreNum;
        left = totalDataNum % usedCoreNum;

        // UB 内行 stride：单行字节数向上取整到 32B 对齐后的【元素个数】
        int64_t oneDataBytes = dataBytes * dataDim;
        int64_t rowAlignBytes = (oneDataBytes + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32;
        rowAlignCount = static_cast<uint32_t>(rowAlignBytes / dataBytes);
        int64_t tileRows = ROW_TILE_UB_BYTES / rowAlignBytes;
        if (tileRows < 1) {
            tileRows = 1;
        }
        if (tileRows > DATA_COPY_PAD_MAX_BLOCK_COUNT) {
            tileRows = DATA_COPY_PAD_MAX_BLOCK_COUNT;
        }
        rowsPerTile = static_cast<uint32_t>(tileRows);
    }

    // 设置分片数据
    ReverseSequenceTilling tilingData;
    tilingData.set_batchSize(inputShape.GetDim(DIM0));
    tilingData.set_maxSeqLen(inputShape.GetDim(DIM1));
    tilingData.set_dataDim(dataDim);
    tilingData.set_handleTotalCount(handleTotalCount);
    tilingData.set_left(left);
    tilingData.set_rowsPerTile(rowsPerTile);
    tilingData.set_rowAlignCount(rowAlignCount);

    // 保存分片数据
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
using optiling::OUTPUT_INDEX;
using optiling::PARAM_INPUT_INDEX;

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* inputShape = context->GetInputShape(PARAM_INPUT_INDEX);
    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_INDEX);

    OPS_LOG_E_IF_NULL("inputShape", inputShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUTPUT_INDEX, context->GetInputDataType(PARAM_INPUT_INDEX));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
// 注意：GE 算子类型名取 ReverseSequenceCust 以避开 CANN 内置同名算子 ReverseSequence，
// 否则运行时 tiling 阶段可能解析到内置实现（属性不匹配导致进程崩溃）。
class ReverseSequenceCust : public OpDef {
public:
    explicit ReverseSequenceCust(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("seq_lengths")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).Follow("input", FollowType::DTYPE).FormatList({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(ReverseSequenceCust);
}  // namespace ops
