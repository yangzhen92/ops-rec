/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#define OPS_REC_CUST_VENDORS "packed_block_sparse"

#include <torch/library.h>
#include "common_utils.h"
#include "pytorch_npu_helper.hpp"

using at::Tensor;
using tensor_list = std::vector<Tensor>;

namespace {
#define DIVUP(x, y) (((x) + (y) - 1) / (y))

void Validate(const Tensor& func, const Tensor& qOffsets, const Tensor& kvOffsets, int64_t maxSeqLenQ,
              int64_t maxSeqLenKv, int64_t qBlock)
{
    check_tensor_npu_device({func, qOffsets, kvOffsets}, {"func_tensor", "q_offsets", "kv_offsets"});
    TORCH_CHECK(func.dtype() == at::kInt, "func_tensor must be int32");
    TORCH_CHECK(qOffsets.dtype() == at::kInt && kvOffsets.dtype() == at::kInt, "offsets must be int32");
    TORCH_CHECK(
        qOffsets.dim() == 1 && kvOffsets.dim() == 1 && qOffsets.numel() >= 2 && qOffsets.numel() == kvOffsets.numel(),
        "q_offsets and kv_offsets must be one-dimensional tensors with the same length >= 2");
    TORCH_CHECK(func.dim() == 2 || (func.dim() == 4 && func.size(0) == 1),
                "func_tensor must be TND [T,2*groups] or [1,H,T,2*groups]");
    const int64_t seqDim = func.dim() == 2 ? 0 : 2;
    const int64_t funcDim = func.dim() == 2 ? 1 : 3;
    TORCH_CHECK(maxSeqLenQ > 0 && maxSeqLenKv > 0 && func.size(seqDim) > 0, "invalid sequence geometry");
    const Tensor qOffsetsCpu = qOffsets.to(at::kCPU);
    const Tensor kvOffsetsCpu = kvOffsets.to(at::kCPU);
    const int32_t* qData = qOffsetsCpu.const_data_ptr<int32_t>();
    const int32_t* kvData = kvOffsetsCpu.const_data_ptr<int32_t>();
    const int64_t batch = qOffsetsCpu.numel() - 1;
    TORCH_CHECK(qData[0] == 0 && kvData[0] == 0, "offsets[0] must be zero");
    TORCH_CHECK(kvData[batch] == func.size(seqDim), "kv_offsets[-1] must equal func_tensor's T dimension");
    for (int64_t i = 0; i < batch; ++i) {
        const int32_t qLen = qData[i + 1] - qData[i];
        const int32_t kvLen = kvData[i + 1] - kvData[i];
        TORCH_CHECK(qLen >= 0 && kvLen >= 0, "offsets must be nondecreasing");
        TORCH_CHECK(qLen <= kvLen, "each q sequence length must be <= its kv sequence length");
        TORCH_CHECK(qLen <= maxSeqLenQ && kvLen <= maxSeqLenKv,
                    "each sequence length must fit the corresponding max_seqlen");
    }
    TORCH_CHECK(func.size(funcDim) > 0 && (func.size(funcDim) & 1) == 0, "last dimension must be positive even");
    TORCH_CHECK(qBlock == 64 || qBlock == 128 || qBlock == 256, "q_block_size must be 64, 128 or 256");
}

template <bool K2Q>
tensor_list CreatePacked(const Tensor& func, const Tensor& qOffsets, const Tensor& kvOffsets, int64_t maxSeqLenQ,
                         int64_t maxSeqLenKv, int64_t qBlock, int64_t kvBlock)
{
    Validate(func, qOffsets, kvOffsets, maxSeqLenQ, maxSeqLenKv, qBlock);
    TORCH_CHECK(kvBlock == 64 || kvBlock == 128, "kv_block_size must be 64 or 128");
    TORCH_CHECK((K2Q && qBlock == 2 * kvBlock) || (!K2Q && qBlock == kvBlock), "direction and block geometry mismatch");
    const int64_t b = qOffsets.numel() - 1;
    const int64_t h = func.dim() == 2 ? 1 : func.size(1);
    const int64_t nq = DIVUP(maxSeqLenQ, qBlock), nk = DIVUP(maxSeqLenKv, kvBlock);
    const int64_t outer = K2Q ? nk : nq, inner = K2Q ? nq : nk;
    const Tensor funcContiguous = func.contiguous();
    auto maskCnt = at::empty({b, h, outer}, func.options());
    auto fullCnt = at::empty_like(maskCnt);
    auto rowBounds = at::empty({b, h, outer, 2}, func.options());
    auto directDescriptor = at::empty({b, h, outer, inner}, func.options());
    auto directBitmap = at::empty({b * h * outer, inner, qBlock, kvBlock / 32}, func.options());
    const Tensor qOffsetsContiguous = qOffsets.contiguous();
    const Tensor kvOffsetsContiguous = kvOffsets.contiguous();
    const int32_t sh = funcContiguous.dim() == 2 ? 0 : funcContiguous.stride(1);
    const int32_t sq = funcContiguous.stride(funcContiguous.dim() == 2 ? 0 : 2);
    const int32_t sf = funcContiguous.stride(funcContiguous.dim() == 2 ? 1 : 3);
    const int32_t mql = maxSeqLenQ, mkl = maxSeqLenKv, qb = qBlock, kb = kvBlock;
    if constexpr (K2Q) {
        EXEC_NPU_CMD(aclnnCreateBwdPackedBlockSparse, funcContiguous, qOffsetsContiguous, kvOffsetsContiguous, sh, sq,
                     sf, mql, mkl, qb, kb, maskCnt, fullCnt, rowBounds, directDescriptor, directBitmap);
    } else {
        EXEC_NPU_CMD(aclnnCreateFwdPackedBlockSparse, funcContiguous, qOffsetsContiguous, kvOffsetsContiguous, sh, sq,
                     sf, mql, mkl, qb, kb, maskCnt, fullCnt, rowBounds, directDescriptor, directBitmap);
    }
    // Return the direct descriptor and bitmap emitted by Create*. HSTU V2
    // recognizes a one-dimensional prefix tensor as direct descriptor mode, so
    // no separate finalize/compact pass is needed for the current packed path.
    auto directPrefix = at::empty({1}, func.options());
    auto placeholder = at::empty({1}, func.options());
    return {maskCnt,     rowBounds,        placeholder,  fullCnt,     placeholder,
            placeholder, directDescriptor, directPrefix, directBitmap};
}

template <bool K2Q>
tensor_list CreatePackedMeta(const Tensor& func, const Tensor& qOffsets, const Tensor& kvOffsets, int64_t maxSeqLenQ,
                             int64_t maxSeqLenKv, int64_t qBlock, int64_t kvBlock)
{
    const int64_t b = qOffsets.size(0) - 1, h = func.dim() == 2 ? 1 : func.size(1);
    const int64_t nq = DIVUP(maxSeqLenQ, qBlock), nk = DIVUP(maxSeqLenKv, kvBlock);
    const int64_t outer = K2Q ? nk : nq, inner = K2Q ? nq : nk;
    auto cnt = at::empty({b, h, outer}, func.options()), placeholder = at::empty({1}, func.options());
    return {cnt,
            at::empty({b, h, outer, 2}, func.options()),
            placeholder,
            at::empty_like(cnt),
            placeholder,
            placeholder,
            at::empty({b, h, outer, inner}, func.options()),
            placeholder,
            at::empty({b * h * outer, inner, qBlock, kvBlock / 32}, func.options())};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("create_fwd_packed_block_sparse(Tensor func_tensor, Tensor q_offsets, Tensor kv_offsets, "
          "int max_seqlen_q, int max_seqlen_kv, "
          "int q_block_size=128, int kv_block_size=128) -> Tensor[]");
    m.def("create_bwd_packed_block_sparse(Tensor func_tensor, Tensor q_offsets, Tensor kv_offsets, "
          "int max_seqlen_q, int max_seqlen_kv, "
          "int q_block_size=256, int kv_block_size=128) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("create_fwd_packed_block_sparse", &CreatePacked<false>);
    m.impl("create_bwd_packed_block_sparse", &CreatePacked<true>);
}
TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("create_fwd_packed_block_sparse", &CreatePackedMeta<false>);
    m.impl("create_bwd_packed_block_sparse", &CreatePackedMeta<true>);
}
