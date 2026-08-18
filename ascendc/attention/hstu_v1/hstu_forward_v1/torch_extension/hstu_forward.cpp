/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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

// hstu_forward_v1 算子 PyTorch 适配层（由 RecSDK torch_library/hstu 迁移合并，forward-only）
// 注册：hstu_dense / hstu_jagged.equal|delta / hstu_paged（backward 算子不在本次迁移范围，未注册）
#define OPS_REC_CUST_VENDORS "hstu_forward_v1"

#include <cstdint>
#include "c10/core/ScalarType.h"
#include "torch/types.h"
#include "common_utils.h"
#include "hstu_common.h"
#include "pytorch_npu_helper.hpp"

namespace hstu {
at::Tensor hstu_dense_forward_impl_npu(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                       const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& attnBias,
                                       const int64_t maskType, const int64_t maxSeqLen, const double siluScale)
{
    TORCH_CHECK(q.dim() == CONST_4, "The q should be 4D in dense layout");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = attnBias.value_or(at::Tensor());
    auto maskNpu = mask.value_or(at::Tensor());

    TORCH_CHECK(MaxSeqLenCheck(maxSeqLen), "maxSeqLen check failed");
    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    auto attnOutput = at::empty_like(denseQ);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / static_cast<double>(maxSeqLen) : siluScale;

    EXEC_NPU_CMD(aclnnHstuDenseForward, denseQ, denseK, denseV, maskNpu, denseBias, maskType, maxSeqLen, realSiluScale,
                 attnOutput);
    return attnOutput;
}

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("hstu_dense(Tensor q, "
          "           Tensor k, "
          "           Tensor v, "
          "           Tensor? mask=None, "
          "           Tensor? attn_bias=None, "
          "           int mask_type=0, "
          "           int max_seq_len=0, "
          "           float silu_scale=0.0) -> Tensor");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("hstu_dense", TORCH_FN(hstu_dense_forward_impl_npu));
}

}  // namespace hstu
namespace hstu {
at::Tensor hstu_deltaq_forward_impl_npu(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& attnBias, const int64_t maskType, const int64_t maxSeqLenQ,
    const int64_t maxSeqLenK, const double siluScale, const at::Tensor& seqOffset, const at::Tensor& seqOffsetK,
    const c10::optional<at::Tensor>& numContext, const c10::optional<at::Tensor>& numTarget,
    const c10::optional<int64_t>& targetGroupSize, const c10::optional<double>& alpha, const bool deterministic = false)
{
    TORCH_CHECK(q.dim() == CONST_3, "The q should be 3D in jagged layout");

    auto acSeqOffset = seqOffset;
    auto acSeqOffsetK = seqOffsetK.to(acSeqOffset.scalar_type());
    TORCH_CHECK(acSeqOffset.size(0) >= CONST_2, "acSeqOffset params error should have at least two element.");
    TORCH_CHECK(acSeqOffsetK.size(0) >= CONST_2, "acSeqOffsetK params error should have at least two element.");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = attnBias.value_or(at::Tensor());
    auto maskNpu = mask.value_or(at::Tensor());

    auto batchsize = acSeqOffset.size(0) - 1;
    auto _zeros = at::zeros({batchsize}, acSeqOffset.options());
    auto acNumContext = numContext.has_value() ? numContext.value().to(acSeqOffset.scalar_type()) : _zeros;
    auto acNumTarget = numTarget.has_value() ? numTarget.value().to(acSeqOffset.scalar_type()) : _zeros;
    auto acTargetGroupSize = targetGroupSize.value_or(0);
    double realAlpha = alpha.value_or(1.0);

    TORCH_CHECK(MaxSeqLenCheck(maxSeqLenQ), "maxSeqLen check failed");
    TORCH_CHECK(MaxSeqLenCheck(maxSeqLenK), "maxSeqLenK check failed");
    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    bool use_fp8 = (q.scalar_type() == at::kFloat8_e4m3fn);
    at::ScalarType output_dtype = use_fp8 ? at::kHalf : denseQ.scalar_type();

    auto attnOutput =
        deterministic
            ? at::empty({denseQ.size(0), denseQ.size(1), denseV.size(2)}, denseQ.options().dtype(output_dtype))
            : at::zeros({denseQ.size(0), denseQ.size(1), denseV.size(2)}, denseQ.options().dtype(output_dtype));

    double realSiluScale = (siluScale == 0.0) ? 1.0f / static_cast<double>(maxSeqLenQ) : siluScale;

    EXEC_NPU_CMD(aclnnHstuJaggedForward, denseQ, denseK, denseV, maskNpu, denseBias, acSeqOffset, acSeqOffsetK,
                 acNumContext, acNumTarget, maskType, maxSeqLenQ, maxSeqLenK, realSiluScale, acTargetGroupSize,
                 realAlpha, deterministic, attnOutput);
    return attnOutput;
}


at::Tensor hstu_jagged_equal(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                             const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& attnBias,
                             const int64_t maskType, const int64_t maxSeqLen, const double siluScale,
                             const at::Tensor& seqOffset, const c10::optional<at::Tensor>& numContext,
                             const c10::optional<at::Tensor>& numTarget, const c10::optional<int64_t>& targetGroupSize,
                             const c10::optional<double>& alpha, const bool deterministic = false)
{
    return hstu_deltaq_forward_impl_npu(q, k, v, mask, attnBias, maskType, maxSeqLen, maxSeqLen, siluScale, seqOffset,
                                        seqOffset, numContext, numTarget, targetGroupSize, alpha);
}

at::Tensor hstu_jagged_deltaq(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                              const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& attnBias,
                              const int64_t maskType, const int64_t maxSeqLenQ, const int64_t maxSeqLenK,
                              const double siluScale, const at::Tensor& seqOffset, const at::Tensor& seqOffsetK,
                              const c10::optional<at::Tensor>& numContext, const c10::optional<at::Tensor>& numTarget,
                              const c10::optional<int64_t>& targetGroupSize, const c10::optional<double>& alpha,
                              const bool deterministic = false)
{
    return hstu_deltaq_forward_impl_npu(q, k, v, mask, attnBias, maskType, maxSeqLenQ, maxSeqLenK, siluScale, seqOffset,
                                        seqOffsetK, numContext, numTarget, targetGroupSize, alpha);
}

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("hstu_jagged.equal(Tensor q, "
          "                  Tensor k, "
          "                  Tensor v, "
          "                  Tensor? mask=None, "
          "                  Tensor? attn_bias=None, "
          "                  int mask_type=0, "
          "                  int max_seq_len=0, "
          "                  float silu_scale=0.0, "
          "                  Tensor seq_offset=None, "
          "                  Tensor? num_context=None, "
          "                  Tensor? num_target=None, "
          "                  int? target_group_size=0, "
          "                  float? alpha=1.0, "
          "                  bool deterministic=False) -> Tensor");
    
    m.def("hstu_jagged.delta(Tensor q, "
          "                  Tensor k, "
          "                  Tensor v, "
          "                  Tensor? mask=None, "
          "                  Tensor? attn_bias=None, "
          "                  int mask_type=0, "
          "                  int max_seq_len=0, "
          "                  int max_seq_len_k=0, "
          "                  float silu_scale=0.0, "
          "                  Tensor seq_offset=None, "
          "                  Tensor seq_offset_k=None, "
          "                  Tensor? num_context=None, "
          "                  Tensor? num_target=None, "
          "                  int? target_group_size=0, "
          "                  float? alpha=1.0, "
          "                  bool deterministic=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("hstu_jagged.equal", TORCH_FN(hstu_jagged_equal));
    m.impl("hstu_jagged.delta", TORCH_FN(hstu_jagged_deltaq));
}

}  // namespace hstu
namespace hstu {
at::Tensor hstu_paged_forward_impl_npu(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                       const at::Tensor& kvCache, const c10::optional<at::Tensor>& mask,
                                       const c10::optional<at::Tensor>& attnBias, const int64_t maskType,
                                       const int64_t maxSeqLen, const int64_t maxSeqLenK, const double siluScale,
                                       const at::Tensor& seqOffset, const at::Tensor& seqOffsetK,
                                       const at::Tensor& seqOffsetT, const at::Tensor& pageOffsets,
                                       const at::Tensor& pageIds, const at::Tensor& lastPageLen,
                                       const at::Tensor& numTarget, const int64_t targetGroupSize,
                                       const c10::optional<double>& alpha, const bool deterministic = false)
{
    check_tensor_non_empty(kvCache, "kv_cache");
    check_tensor_non_empty(pageIds, "page_ids");
    check_tensor_non_empty(lastPageLen, "last_page_len");
    auto acPageOffsets = pageOffsets.to(seqOffset.scalar_type());
    auto acPageIds = pageIds.to(seqOffset.scalar_type());
    auto acLastPageLen = lastPageLen.to(seqOffset.scalar_type());

    TORCH_CHECK(q.dim() == CONST_3, "The q should be 3D in jagged layout");

    auto acSeqOffset = seqOffset;
    auto acSeqOffsetK = seqOffsetK.to(seqOffset.scalar_type());
    auto acSeqOffsetT = seqOffsetT.to(seqOffset.scalar_type());
    TORCH_CHECK(acSeqOffset.size(0) >= CONST_2, "acSeqOffsetq params error should have at least two element.");
    TORCH_CHECK(acSeqOffsetK.size(0) >= CONST_2, "acSeqOffsetK params error should have at least two element.");
    TORCH_CHECK(acSeqOffsetT.size(0) >= CONST_2, "acSeqOffsetT params error should have at least two element.");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseKvCache = kvCache.contiguous();
    auto denseBias = attnBias.value_or(at::Tensor());
    auto maskNpu = mask.value_or(at::Tensor());

    TORCH_CHECK(MaxSeqLenCheck(maxSeqLen), "maxSeqLen check failed");
    TORCH_CHECK(MaxSeqLenCheck(maxSeqLenK), "maxSeqLenK check failed");
    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");
    TORCH_CHECK(!numTarget.defined() || targetGroupSize > 0,
                "targetGroupSize must be greater than 0 when numTarget is defined");

    auto attnOutput = deterministic ? at::empty_like(denseQ) : at::zeros_like(denseQ);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / static_cast<double>(maxSeqLen) : siluScale;
    double realAlpha = alpha.value_or(1.0);

    const auto _numContext = at::zeros_like(numTarget);
    const auto acNumTarget = numTarget.to(seqOffset.scalar_type());

    EXEC_NPU_CMD(aclnnHstuPagedForward, denseQ, denseK, denseV, maskNpu, denseBias, acSeqOffset, acSeqOffsetK,
                 acSeqOffsetT, denseKvCache, acPageOffsets, acPageIds, acLastPageLen, _numContext, acNumTarget,
                 maskType, maxSeqLen, maxSeqLenK, realSiluScale, targetGroupSize, realAlpha, deterministic, attnOutput);
    return attnOutput;
}

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("hstu_paged(Tensor q, "
          "           Tensor k, "
          "           Tensor v, "
          "           Tensor kv_cache=None, "
          "           Tensor? mask=None, "
          "           Tensor? attn_bias=None,"
          "           int mask_type=0, "
          "           int max_seq_len=0, "
          "           int max_seq_len_k=0, "
          "           float silu_scale=0.0, "
          "           Tensor seq_offset=None, "
          "           Tensor seq_offset_k=None, "
          "           Tensor seq_offset_t=None, "
          "           Tensor page_offsets=None, "
          "           Tensor page_ids=None, "
          "           Tensor last_page_len=None, "
          "           Tensor num_target=None, "
          "           int target_group_size=0, "
          "           float? alpha=1.0, "
          "           bool deterministic=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)

{
    m.impl("hstu_paged", TORCH_FN(hstu_paged_forward_impl_npu));
}
}  // namespace hstu

namespace {

// Meta 注册：inductor 构图阶段只推导 shape，不真正计算（增量能力，不影响 eager 路径）
at::Tensor hstu_dense_meta(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                           const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& attnBias,
                           const int64_t maskType, const int64_t maxSeqLen, const double siluScale)
{
    TORCH_CHECK(q.dim() == 4, "The q should be 4D in dense layout");
    return at::empty_like(q, q.options());
}

at::Tensor hstu_jagged_equal_meta(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                  const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& attnBias,
                                  const int64_t maskType, const int64_t maxSeqLen, const double siluScale,
                                  const at::Tensor& seqOffset, const c10::optional<at::Tensor>& numContext,
                                  const c10::optional<at::Tensor>& numTarget,
                                  const c10::optional<int64_t>& targetGroupSize,
                                  const c10::optional<double>& alpha, const bool deterministic)
{
    TORCH_CHECK(q.dim() == 3, "The q should be 3D in jagged layout");
    at::ScalarType outputDtype = (q.scalar_type() == at::kFloat8_e4m3fn) ? at::kHalf : q.scalar_type();
    return at::empty({q.size(0), q.size(1), v.size(2)}, q.options().dtype(outputDtype));
}

at::Tensor hstu_jagged_delta_meta(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                  const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& attnBias,
                                  const int64_t maskType, const int64_t maxSeqLenQ, const int64_t maxSeqLenK,
                                  const double siluScale, const at::Tensor& seqOffset, const at::Tensor& seqOffsetK,
                                  const c10::optional<at::Tensor>& numContext,
                                  const c10::optional<at::Tensor>& numTarget,
                                  const c10::optional<int64_t>& targetGroupSize,
                                  const c10::optional<double>& alpha, const bool deterministic)
{
    TORCH_CHECK(q.dim() == 3, "The q should be 3D in jagged layout");
    at::ScalarType outputDtype = (q.scalar_type() == at::kFloat8_e4m3fn) ? at::kHalf : q.scalar_type();
    return at::empty({q.size(0), q.size(1), v.size(2)}, q.options().dtype(outputDtype));
}

at::Tensor hstu_paged_meta(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                           const at::Tensor& kvCache, const c10::optional<at::Tensor>& mask,
                           const c10::optional<at::Tensor>& attnBias, const int64_t maskType,
                           const int64_t maxSeqLen, const int64_t maxSeqLenK, const double siluScale,
                           const at::Tensor& seqOffset, const at::Tensor& seqOffsetK,
                           const at::Tensor& seqOffsetT, const at::Tensor& pageOffsets,
                           const at::Tensor& pageIds, const at::Tensor& lastPageLen,
                           const at::Tensor& numTarget, const int64_t targetGroupSize,
                           const c10::optional<double>& alpha, const bool deterministic)
{
    TORCH_CHECK(q.dim() == 3, "The q should be 3D in paged layout");
    return at::empty_like(q, q.options());
}

// ---- 前向算子 Meta 注册 ----
// 当前为 forward-only 版本：仅注册 4 个前向算子的 Meta 实现，供 inductor 构图阶段
// 做 shape/dtype 推导（不真正计算）。反向算子（hstu_*_backward）不在本次迁移范围内，
// 本文件未注册任何 backward 算子。

}  // anonymous namespace

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("hstu_dense", &hstu_dense_meta);
    m.impl("hstu_jagged.equal", &hstu_jagged_equal_meta);
    m.impl("hstu_jagged.delta", &hstu_jagged_delta_meta);
    m.impl("hstu_paged", &hstu_paged_meta);
}
