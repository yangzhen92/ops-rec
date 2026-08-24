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

#include "in_linear_silu_backward_tiling.h"
#include "../op_kernel/in_linear_silu_backward_tilingKey.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

using namespace std;

namespace {
static const uint64_t BLOCK_SIZE = 32;
static const uint64_t HALF_SIZE = 2;
static const uint64_t FLOAT_SIZE = 4;
static const uint64_t GM_ALIGN = 512;
static const uint64_t SPLIT_ARG_LIST_SIZE = 4;
static const uint64_t MAX_BLOCK_DIM = 512;
static const uint64_t BLOCK_HEIGHT = 256;
static const uint64_t MIN_X_DIM = 16;
static const uint64_t MAX_X_DIM = 8192;
static const uint64_t MIN_W_DIM = 64;
static const uint64_t MAX_W_DIM = 32768;
static const uint64_t COMPUTE_PIPE_NUM = 2;
static const uint64_t DIM_1 = 1;
static const uint64_t DIM_2 = 2;
}  // namespace

namespace INPUT_INDEX_T {
constexpr int X_INDEX = 0;
constexpr int WEIGHT_INDEX = 1;
constexpr int BIAS_INDEX = 2;
constexpr int USER_GRAD_INDEX = 3;
constexpr int VALUE_GRAD_INDEX = 4;
constexpr int QUERY_GRAD_INDEX = 5;
constexpr int KEY_GRAD_INDEX = 6;
constexpr int LINEAR_OUTPUT_INDEX = 7;
}  // namespace INPUT_INDEX_T

namespace optiling {
static bool TilingKeySetImpl(gert::TilingContext* context, InLinearSiluBackwardTilingData& tiling, bool isTrans,
                             bool isVardim)
{
    bool enableBias = tiling.get_enableBias();

    const uint64_t tilingKey = GET_TPL_TILING_KEY(enableBias, isTrans, isVardim);
    context->SetTilingKey(tilingKey);

    return true;
}

static uint32_t GetBlockK(gert::TilingContext* context, uint32_t hiddenSize)
{
    uint32_t embedDim = hiddenSize / SPLIT_ARG_LIST_SIZE;
    if (hiddenSize <= MAX_BLOCK_DIM) {
        return hiddenSize;
    } else {
        uint32_t blockK = MAX_BLOCK_DIM;
        while (blockK >= MIN_X_DIM) {
            if (embedDim % blockK == 0) {
                break;
            }
            blockK >>= 1;
        }
        return blockK;
    }
}

static uint32_t GetBlockM(gert::TilingContext* context, uint32_t seqLen)
{
    uint32_t blockM = BLOCK_HEIGHT;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    while (blockM * aivCoreNum >= seqLen && blockM > MIN_X_DIM) {
        blockM >>= 1;
    }
    return blockM;
}

static bool RangeCheck(gert::TilingContext* context, int64_t seqLen, int64_t dim, int64_t hiddenSize,
                       const int64_t* split_arg_list)
{
    int64_t uDim = split_arg_list[0];
    int64_t vDim = split_arg_list[1];
    int64_t qDim = split_arg_list[2];
    int64_t kDim = split_arg_list[3];
    int64_t totalDim = uDim + vDim + qDim + kDim;
    OPS_LOG_E_IF(dim < MIN_X_DIM || dim > MAX_X_DIM, context, return false,
                 "[ERROR]InLinearSilu normed_x dim should be in range[%llu, %llu]!", MIN_X_DIM, MAX_X_DIM);
    OPS_LOG_E_IF(hiddenSize < MIN_W_DIM || hiddenSize > MAX_W_DIM, context, return false,
                 "[ERROR]InLinearSilu weight hiddenSize should be in range[%llu, %llu]!", MIN_W_DIM, MAX_W_DIM);
    OPS_LOG_E_IF(hiddenSize != totalDim, context, return false,
                 "[ERROR]InLinearSilu hiddenSize should equal to sum(split_arg_list)!");
    OPS_LOG_E_IF(seqLen < 1, context, return false, "[ERROR]InLinearSilu seq_len should be positive!");
    OPS_LOG_E_IF(dim % MIN_X_DIM != 0, context, return false,
                 "[ERROR]InLinearSilu normed_x dim should be multiples of %llu!", MIN_X_DIM);
    OPS_LOG_E_IF(hiddenSize % MIN_X_DIM != 0, context, return false,
                 "[ERROR]InLinearSilu weight hiddenSize should be multiples of %llu!", MIN_X_DIM);
    OPS_LOG_E_IF(hiddenSize % (SPLIT_ARG_LIST_SIZE * dim) != 0, context, return false,
                 "[ERROR]InLinearSilu weight hiddenSize should be multiples of 4 * normed_x dim!");
    return true;
}

static bool SplitArgCheck(gert::TilingContext* context, const int64_t* split_arg_list)
{
    for (uint64_t i = 0; i < SPLIT_ARG_LIST_SIZE; i++) {
        OPS_LOG_E_IF(split_arg_list[i] < static_cast<int64_t>(MIN_X_DIM) ||
                         split_arg_list[i] > static_cast<int64_t>(MAX_X_DIM) ||
                         split_arg_list[i] % static_cast<int64_t>(MIN_X_DIM) != 0,
                     context, return false,
                     "[ERROR]InLinearSilu split_arg_list[%llu] should be in range[%llu, %llu] and multiples of %llu!",
                     i, MIN_X_DIM, MAX_X_DIM, MIN_X_DIM);
    }
    return true;
}

static bool GradShapeCheck(gert::TilingContext* context, int64_t seqLen, int64_t hiddenSize,
                           const int64_t* split_arg_list)
{
    const int gradIndexList[SPLIT_ARG_LIST_SIZE] = {INPUT_INDEX_T::USER_GRAD_INDEX, INPUT_INDEX_T::VALUE_GRAD_INDEX,
                                                    INPUT_INDEX_T::QUERY_GRAD_INDEX, INPUT_INDEX_T::KEY_GRAD_INDEX};
    for (uint64_t i = 0; i < SPLIT_ARG_LIST_SIZE; i++) {
        OPS_LOG_E_IF_NULL("gradShape", context->GetInputShape(gradIndexList[i]), return false);
        const gert::Shape gradShape = context->GetInputShape(gradIndexList[i])->GetStorageShape();
        OPS_LOG_E_IF(
            gradShape.GetDimNum() != DIM_2 || gradShape.GetDim(0) != seqLen || gradShape.GetDim(1) != split_arg_list[i],
            context, return false, "[ERROR]InLinearSilu grad tensor[%llu] shape should be [%lld, %lld]!", i, seqLen,
            split_arg_list[i]);
    }
    OPS_LOG_E_IF_NULL("linearOutputShape", context->GetInputShape(INPUT_INDEX_T::LINEAR_OUTPUT_INDEX), return false);
    const gert::Shape linearOutputShape = context->GetInputShape(INPUT_INDEX_T::LINEAR_OUTPUT_INDEX)->GetStorageShape();
    OPS_LOG_E_IF(linearOutputShape.GetDimNum() != DIM_2 || linearOutputShape.GetDim(0) != seqLen ||
                     linearOutputShape.GetDim(1) != hiddenSize,
                 context, return false, "[ERROR]InLinearSilu linear_output shape should be [%lld, %lld]!", seqLen,
                 hiddenSize);
    return true;
}

static ge::graphStatus TilingMatmulImpl(gert::TilingContext* context, uint32_t m, uint32_t n, uint32_t k, bool isTrans,
                                        TCubeTiling& tiling_mm)
{
    matmul_tiling::DataType dataType;
    ge::DataType xType = context->GetInputTensor(0)->GetDataType();
    if (xType == ge::DataType::DT_FLOAT) {
        dataType = matmul_tiling::DataType::DT_FLOAT;
    } else if (xType == ge::DataType::DT_FLOAT16) {
        dataType = matmul_tiling::DataType::DT_FLOAT16;
    } else if (xType == ge::DataType::DT_BF16) {
        dataType = matmul_tiling::DataType::DT_BFLOAT16;
    } else {
        OPS_LOG_E("", "invalid datatype, only support float/fp16/bf16\n");
        return ge::GRAPH_FAILED;
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling MM(ascendPlatform);
    MM.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType, isTrans);
    MM.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    MM.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    MM.SetOrgShape(m, n, k);
    MM.SetShape(m, n, k);
    MM.SetBias(false);
    MM.SetBufferSpace(-1, -1, -1);

    OPS_LOG_E_IF(MM.GetTiling(tiling_mm) == -1, context, return ge::GRAPH_FAILED, "matmul tiling failed");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("xShape", context->GetInputShape(INPUT_INDEX_T::X_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("wShape", context->GetInputShape(INPUT_INDEX_T::WEIGHT_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("xInputDesc", context->GetInputDesc(INPUT_INDEX_T::X_INDEX), return ge::GRAPH_FAILED);
    InLinearSiluBackwardTilingData tiling;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);

    const gert::Shape xShape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape wShape = context->GetInputShape(1)->GetStorageShape();
    OPS_LOG_E_IF(xShape.GetDimNum() != DIM_2 || wShape.GetDimNum() != DIM_2, context, return ge::GRAPH_FAILED,
                 "[ERROR]InLinearSilu x and weight should be 2D tensor!");
    uint32_t seqLen = xShape.GetDim(0);
    uint32_t embedDim = xShape.GetDim(1);

    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    auto list = context->GetAttrs()->GetListInt(0);
    OPS_LOG_E_IF_NULL("list", list, return ge::GRAPH_FAILED);
    size_t size = list->GetSize();
    OPS_LOG_E_IF(size != SPLIT_ARG_LIST_SIZE, context, return ge::GRAPH_FAILED,
                 "[ERROR]InLinearSilu attrs split_arg_list only support 4");
    const int64_t* data = list->GetData();
    OPS_LOG_E_IF_NULL("data", data, return ge::GRAPH_FAILED);

    uint32_t hiddenSize = wShape.GetDim(0);
    // 有效范围校验
    if (!RangeCheck(context, seqLen, embedDim, hiddenSize, data)) {
        return ge::GRAPH_FAILED;
    }
    if (!SplitArgCheck(context, data)) {
        return ge::GRAPH_FAILED;
    }
    if (!GradShapeCheck(context, seqLen, hiddenSize, data)) {
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_E_IF(wShape.GetDim(1) != embedDim, context, return ge::GRAPH_FAILED,
                 "[ERROR]InLinearSilu weight dim[1] should equal to x dim[1]!");

    auto biasTensor = context->GetOptionalInputTensor(INPUT_INDEX_T::BIAS_INDEX);
    if (biasTensor == nullptr) {
        tiling.set_enableBias(0);
    } else {
        const gert::Shape biasShape = context->GetInputShape(INPUT_INDEX_T::BIAS_INDEX)->GetStorageShape();
        OPS_LOG_E_IF(biasShape.GetDimNum() != DIM_1 || biasShape.GetDim(0) != hiddenSize, context,
                     return ge::GRAPH_FAILED, "[ERROR]InLinearSilu bias shape should be [%u]!", hiddenSize);
        tiling.set_enableBias(1);
    }
    uint32_t blockK = GetBlockK(context, hiddenSize);
    uint32_t blockM = GetBlockM(context, seqLen);
    tiling.set_embedDim(embedDim);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_seqLen(seqLen);
    tiling.set_uDim(data[0]);
    tiling.set_vDim(data[1]);
    tiling.set_qDim(data[2]);
    tiling.set_kDim(data[3]);
    tiling.set_blockM(blockM);
    tiling.set_blockK(blockK);
    bool isTrans = *context->GetAttrs()->GetBool(1);
    bool isVardim = *context->GetAttrs()->GetBool(2);

    TilingKeySetImpl(context, tiling, isTrans, isVardim);
    if (ge::GRAPH_FAILED == TilingMatmulImpl(context, BLOCK_HEIGHT, embedDim, blockK, false, tiling.LW_MM)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_FAILED == TilingMatmulImpl(context, blockK, embedDim, BLOCK_HEIGHT, true, tiling.LX_MM)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);

    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    size_t userSize = static_cast<size_t>(COMPUTE_PIPE_NUM * 2 * aicCoreNum * BLOCK_HEIGHT * blockK * sizeof(float));
    currentWorkspace[0] = userSize + sysWorkspaceSize;
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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("x1_shape", x1_shape, return ge::GRAPH_FAILED);
    gert::Shape* y_shape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("y_shape", y_shape, return ge::GRAPH_FAILED);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class InLinearSiluBackward : public OpDef {
public:
    explicit InLinearSiluBackward(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .Follow("x", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("user_grad")
            .ParamType(REQUIRED)
            .Follow("x", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value_grad")
            .ParamType(REQUIRED)
            .Follow("x", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("query_grad")
            .ParamType(REQUIRED)
            .Follow("x", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key_grad")
            .ParamType(REQUIRED)
            .Follow("x", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("linear_output")
            .ParamType(REQUIRED)
            .Follow("x", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("x_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("weights_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("bias_grad")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("split_arg_list").AttrType(REQUIRED).ListInt();
        this->Attr("isTrans").AttrType(OPTIONAL).Bool(true);
        this->Attr("isVardim").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(InLinearSiluBackward);
}  // namespace ops
