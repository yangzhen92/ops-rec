/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"
#include "tiling_policy_packed_block_sparse.h"

namespace {
constexpr int32_t THREADS_PER_BLOCK = 256;
constexpr int32_t WARPS_PER_BLOCK = 8;

struct Geometry {
    int32_t b;
    int32_t h;
    int32_t qLen;
    int32_t kvLen;
    int32_t qBlock;
    int32_t numQ;
    int32_t numKv;
};

bool ReadGeometry(gert::TilingContext* context, Geometry& g, optiling::PackedBlockSparseTilingData& tiling)
{
    const auto* shape = context->GetInputShape(0);
    const auto* qOffsetsShape = context->GetInputShape(1);
    const auto* kvOffsetsShape = context->GetInputShape(2);
    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("func shape", shape, return false);
    OPS_LOG_E_IF_NULL("q offsets shape", qOffsetsShape, return false);
    OPS_LOG_E_IF_NULL("kv offsets shape", kvOffsetsShape, return false);
    OPS_LOG_E_IF_NULL("attrs", attrs, return false);
    const auto* strideH = attrs->GetAttrPointer<int32_t>(0);
    const auto* strideQ = attrs->GetAttrPointer<int32_t>(1);
    const auto* strideF = attrs->GetAttrPointer<int32_t>(2);
    const auto* maxSeqLenQ = attrs->GetAttrPointer<int32_t>(3);
    const auto* maxSeqLenKv = attrs->GetAttrPointer<int32_t>(4);
    const auto* qBlock = attrs->GetAttrPointer<int32_t>(5);
    const auto* kvBlock = attrs->GetAttrPointer<int32_t>(6);
    OPS_LOG_E_IF_NULL("strideH", strideH, return false);
    OPS_LOG_E_IF_NULL("strideQ", strideQ, return false);
    OPS_LOG_E_IF_NULL("strideF", strideF, return false);
    OPS_LOG_E_IF_NULL("maxSeqLenQ", maxSeqLenQ, return false);
    OPS_LOG_E_IF_NULL("maxSeqLenKv", maxSeqLenKv, return false);
    OPS_LOG_E_IF_NULL("qBlock", qBlock, return false);
    OPS_LOG_E_IF_NULL("kvBlock", kvBlock, return false);
    const auto& funcShape = shape->GetStorageShape();
    const int32_t funcDimNum = funcShape.GetDimNum();
    OPS_LOG_E_IF(funcDimNum != 2 && funcDimNum != 4, context, return false,
                 "func_tensor must be [T,F] or [1,H,T,F]");
    OPS_LOG_E_IF(funcDimNum == 4 && funcShape.GetDim(0) != 1, context, return false,
                 "the leading dimension of varlen func_tensor must be 1");
    const int32_t batch = static_cast<int32_t>(qOffsetsShape->GetStorageShape().GetDim(0)) - 1;
    OPS_LOG_E_IF(kvOffsetsShape->GetStorageShape().GetDim(0) != batch + 1, context, return false,
                 "q_offsets and kv_offsets must have the same length");
    const int32_t heads = funcDimNum == 2 ? 1 : static_cast<int32_t>(funcShape.GetDim(1));
    g = {batch, heads, *maxSeqLenQ, *maxSeqLenKv, *qBlock,
         (*maxSeqLenQ + *qBlock - 1) / *qBlock, (*maxSeqLenKv + *kvBlock - 1) / *kvBlock};
    const int32_t nFunc = funcShape.GetDim(funcDimNum - 1);
    OPS_LOG_E_IF(g.b <= 0, context, return false, "seq_offsets must contain at least two elements");
    OPS_LOG_E_IF(g.qLen <= 0 || g.kvLen <= 0, context, return false, "sequence length must be positive");
    OPS_LOG_E_IF((*kvBlock != 64 && *kvBlock != 128) ||
                     (*qBlock != *kvBlock && *qBlock != 2 * *kvBlock),
                 context, return false,
                 "packed HSTU geometry requires q/kv=64/64,128/128,128/64 or 256/128");
    OPS_LOG_E_IF(nFunc <= 0 || (nFunc & 1) != 0, context, return false, "nFunc must be positive and even");
    tiling.set_strideH(*strideH);
    tiling.set_strideQ(*strideQ); tiling.set_strideF(*strideF);
    tiling.set_maxSeqLenQ(*maxSeqLenQ); tiling.set_maxSeqLenKv(*maxSeqLenKv); tiling.set_nFunc(nFunc);
    tiling.set_qBlockSize(g.qBlock); tiling.set_kvBlockSize(*kvBlock);
    tiling.set_numQBlocks(g.numQ); tiling.set_numKvBlocks(g.numKv);
    tiling.set_hNum(g.h); tiling.set_bNum(g.b);
    return true;
}

template <bool K2Q>
ge::graphStatus Tiling(gert::TilingContext* context)
{
    optiling::PackedBlockSparseTilingData tiling;
    Geometry g{};
    if (!ReadGeometry(context, g, tiling)) return ge::GRAPH_FAILED;
    const int32_t kvBlock = tiling.get_kvBlockSize();
    const bool directionMismatch = (K2Q && g.qBlock != 2 * kvBlock) || (!K2Q && g.qBlock != kvBlock);
    OPS_LOG_E_IF(directionMismatch, context, return ge::GRAPH_FAILED,
                 "direction and q_block_size mismatch");
    const uint32_t outer = static_cast<uint32_t>(g.b * g.h * (K2Q ? g.numKv : g.numQ));
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(std::min(outer, platform.GetCoreNumAiv()));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

template <bool K2Q>
ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* shape = context->GetInputShape(0);
    const auto* qOffsetsShape = context->GetInputShape(1);
    const auto* attrs = context->GetAttrs();
    if (shape == nullptr || qOffsetsShape == nullptr || attrs == nullptr) return ge::GRAPH_FAILED;
    const int64_t b = qOffsetsShape->GetDim(0) - 1;
    const int64_t h = shape->GetDimNum() == 2 ? 1 : shape->GetDim(1);
    const int64_t maxSeqLenQ = *attrs->GetAttrPointer<int32_t>(3);
    const int64_t maxSeqLenKv = *attrs->GetAttrPointer<int32_t>(4);
    const int64_t qBlock = *attrs->GetAttrPointer<int32_t>(5);
    const int64_t kvBlock = *attrs->GetAttrPointer<int32_t>(6);
    const int64_t nq = (maxSeqLenQ + qBlock - 1) / qBlock;
    const int64_t nk = (maxSeqLenKv + kvBlock - 1) / kvBlock;
    const int64_t outer = K2Q ? nk : nq, inner = K2Q ? nq : nk;
    auto* maskCnt = context->GetOutputShape(0);
    auto* fullCnt = context->GetOutputShape(1);
    auto* rowBounds = context->GetOutputShape(2);
    auto* descriptor = context->GetOutputShape(3);
    auto* bitmap = context->GetOutputShape(4);
    maskCnt->SetDimNum(3); maskCnt->SetDim(0,b); maskCnt->SetDim(1,h); maskCnt->SetDim(2,outer);
    *fullCnt = *maskCnt;
    rowBounds->SetDimNum(4); rowBounds->SetDim(0,b); rowBounds->SetDim(1,h);
    rowBounds->SetDim(2,outer); rowBounds->SetDim(3,2);
    descriptor->SetDimNum(4); descriptor->SetDim(0,b); descriptor->SetDim(1,h);
    descriptor->SetDim(2,outer); descriptor->SetDim(3,inner);
    bitmap->SetDimNum(4); bitmap->SetDim(0,b*h*outer); bitmap->SetDim(1,inner);
    bitmap->SetDim(2,qBlock); bitmap->SetDim(3,kvBlock/32);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferType(gert::InferDataTypeContext* context)
{
    for (size_t i = 0; i < 5; ++i) {
        if (context->SetOutputDataType(i, ge::DT_INT32) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {
template <bool K2Q> class PackedOp : public OpDef {
public:
    explicit PackedOp(const char* name) : OpDef(name)
    {
        this->Input("func_tensor").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("q_offsets").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("kv_offsets").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("mask_block_cnt").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("full_block_cnt").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("row_bounds").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("block_descriptor").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("partial_bitmap").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Attr("stride_h").Int(); this->Attr("stride_q").Int(); this->Attr("stride_f").Int();
        this->Attr("max_seq_len_q").Int(); this->Attr("max_seq_len_kv").Int();
        this->Attr("q_block_size").Int(); this->Attr("kv_block_size").Int();
        this->SetInferShape(InferShape<K2Q>).SetInferDataType(InferType);
        this->AICore().SetTiling(Tiling<K2Q>);
        this->AICore().AddConfig("ascend950");
    }
};
class CreateFwdPackedBlockSparse : public PackedOp<false> {
public: explicit CreateFwdPackedBlockSparse(const char* name) : PackedOp(name) {}
};
class CreateBwdPackedBlockSparse : public PackedOp<true> {
public: explicit CreateBwdPackedBlockSparse(const char* name) : PackedOp(name) {}
};
OP_ADD(CreateFwdPackedBlockSparse);
OP_ADD(CreateBwdPackedBlockSparse);
}
