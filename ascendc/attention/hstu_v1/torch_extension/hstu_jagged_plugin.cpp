/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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
#define OPS_REC_CUST_VENDORS "hstu_forward_v1,hstu_backward_v1"
#include <cstdint>
#include "c10/core/ScalarType.h"
#include "hstu_common.h"
#include "torch/types.h"
#include "common_utils.h"
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_deltaq_backward_impl_npu(
    const at::Tensor& grad, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor> mask, const c10::optional<at::Tensor> attnBias, const int64_t maskType,
    const int64_t maxSeqLenQ, const int64_t maxSeqLenK, const double siluScale, const at::Tensor& seqOffsetQ,
    const at::Tensor& seqOffsetK, const c10::optional<at::Tensor>& numContext,
    const c10::optional<at::Tensor>& numTarget, const c10::optional<int64_t>& targetGroupSize,
    const c10::optional<double>& alpha)
{
    TORCH_CHECK(seqOffsetQ.size(0) >= CONST_2, "seqOffsetQ params error should have at least two element.");
    TORCH_CHECK(seqOffsetK.size(0) >= CONST_2, "seqOffsetK params error should have at least two element.");
    TORCH_CHECK(MaxSeqLenCheck(maxSeqLenQ), "maxSeqLenQ check failed");
    TORCH_CHECK(MaxSeqLenCheck(maxSeqLenK), "maxSeqLenK check failed");
    if (CheckOptionalTensorIsNotNone(numContext) || CheckOptionalTensorIsNotNone(numTarget)) {
        uint32_t batchSize = seqOffsetQ.size(0) - 1;
        TORCH_CHECK(numContext.has_value(), "numContext is required when numTarget or targetGroupSize is not None");
        TORCH_CHECK(numTarget.has_value(), "numTarget is required when numContext or targetGroupSize is not None");
        TORCH_CHECK(numContext.value().dim() == CONST_1, "The numContext should be 1D in normal layout");
        TORCH_CHECK(numTarget.value().dim() == CONST_1, "The numTarget should be 1D in normal layout");
        TORCH_CHECK(numContext.value().size(0) == batchSize,
                    "The numContext batch size should be equal to the grad batch size");
        TORCH_CHECK(numTarget.value().size(0) == batchSize,
                    "The numTarget batch size should be equal to the grad batch size");
        TORCH_CHECK(CheckInList(targetGroupSize.value_or(0), {1, 3}), "The targetGroupSize should be in [1, 3]");
    }

    auto acSeqOffsetQ = seqOffsetQ;
    auto acSeqOffsetK = seqOffsetK;

    auto acAttnBias = attnBias.value_or(at::Tensor());
    auto acMask = mask.value_or(at::Tensor());
    auto acNumContext = numContext.value_or(at::Tensor());
    auto acNumTarget = numTarget.value_or(at::Tensor());

    auto acTargetGroupSize = targetGroupSize.value_or(0);
    double realAlpha = alpha.value_or(1.0);

    auto denseGrad = grad.contiguous();
    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseAttnBias = acAttnBias.contiguous();
    auto denseMask = acMask.contiguous();
    auto denseNumContext = acNumContext.contiguous();
    auto denseNumTarget = acNumTarget.contiguous();

    double realSiluScale = (siluScale == 0.0) ? 1.0f / static_cast<double>(maxSeqLenQ) : siluScale;

    auto qGradOutput = at::empty_like(denseQ);
    auto kGradOutput = at::empty_like(denseK);
    auto vGradOutput = at::empty_like(denseV);

    at::Tensor attnBiasGradOutput;
    if (denseAttnBias.defined()) {
        attnBiasGradOutput = at::zeros_like(denseAttnBias);
    } else {
        attnBiasGradOutput = at::Tensor();
    }

    const char* layout = "jagged";

    EXEC_NPU_CMD(aclnnHstuJaggedBackward, denseGrad, denseQ, denseK, denseV, denseMask, denseAttnBias, acSeqOffsetQ,
                 acSeqOffsetK, denseNumContext, denseNumTarget, layout, maskType, maxSeqLenQ, maxSeqLenK, realSiluScale,
                 acTargetGroupSize, realAlpha, qGradOutput, kGradOutput, vGradOutput, attnBiasGradOutput);

    if (denseAttnBias.defined()) {
        return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, attnBiasGradOutput);
    } else {
        return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, at::Tensor());
    }
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_jagged_backward(
    const at::Tensor& grad, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor> mask, const c10::optional<at::Tensor> attnBias, const int64_t maskType,
    const int64_t maxSeqLen, const double siluScale, const at::Tensor& seqOffset,
    const c10::optional<at::Tensor>& numContext, const c10::optional<at::Tensor>& numTarget,
    const c10::optional<int64_t>& targetGroupSize, const c10::optional<double>& alpha)
{
    return hstu_deltaq_backward_impl_npu(grad, q, k, v, mask, attnBias, maskType, maxSeqLen, maxSeqLen, siluScale,
                                         seqOffset, seqOffset, numContext, numTarget, targetGroupSize, alpha);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_deltaq_backward(
    const at::Tensor& grad, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor> mask, const c10::optional<at::Tensor> attnBias, const int64_t maskType,
    const int64_t maxSeqLenQ, const int64_t maxSeqLenK, const double siluScale, const at::Tensor& seqOffsetQ,
    const at::Tensor& seqOffsetK, const c10::optional<at::Tensor>& numContext,
    const c10::optional<at::Tensor>& numTarget, const c10::optional<int64_t>& targetGroupSize,
    const c10::optional<double>& alpha)
{
    return hstu_deltaq_backward_impl_npu(grad, q, k, v, mask, attnBias, maskType, maxSeqLenQ, maxSeqLenK, siluScale,
                                         seqOffsetQ, seqOffsetK, numContext, numTarget, targetGroupSize, alpha);
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
    m.def("hstu_jagged_backward.equal(Tensor grad, "
          "                           Tensor q, "
          "                           Tensor k, "
          "                           Tensor v, "
          "                           Tensor? mask=None, "
          "                           Tensor? attn_bias=None, "
          "                           int mask_type=0, "
          "                           int max_seq_len=0, "
          "                           float silu_scale=0.0, "
          "                           Tensor seq_offset=None, "
          "                           Tensor? num_context=None, "
          "                           Tensor? num_target=None, "
          "                           int? target_group_size=0,"
          "                           float? alpha=1.0) -> (Tensor, Tensor, Tensor, Tensor)");
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
    m.def("hstu_jagged_backward.delta(Tensor grad, "
          "                           Tensor q, "
          "                           Tensor k, "
          "                           Tensor v, "
          "                           Tensor? mask=None, "
          "                           Tensor? attn_bias=None, "
          "                           int mask_type=0, "
          "                           int max_seq_len=0, "
          "                           int max_seq_len_k=0, "
          "                           float silu_scale=0.0, "
          "                           Tensor seq_offset=None, "
          "                           Tensor seq_offset_k=None, "
          "                           Tensor? num_context=None, "
          "                           Tensor? num_target=None, "
          "                           int? target_group_size=0,"
          "                           float? alpha=1.0) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("hstu_jagged.equal", TORCH_FN(hstu_jagged_equal));
    m.impl("hstu_jagged_backward.equal", TORCH_FN(hstu_jagged_backward));
    m.impl("hstu_jagged.delta", TORCH_FN(hstu_jagged_deltaq));
    m.impl("hstu_jagged_backward.delta", TORCH_FN(hstu_deltaq_backward));
}

// 注意：forward 不注册 AutogradPrivateUse1（C++ autograd Function 黑盒会导致 inductor
// 图模式数值错误--graph 模式仅走 PrivateUse1 + Meta 干净路径，与 paged/ln_mul/
// disentangle_attention 已验证模式一致）。训练自动微分由 Python 侧
// torch.library.register_autograd 将 hstu_jagged_backward 绑定为 hstu_jagged
// （equal/delta 两个 overload）的 autograd 公式（见 torch_extension/hstu_v1_autograd.py，
// import ops_rec 后自动生效）；hstu_jagged_backward.equal/delta 亦可直接调用。

namespace {

// Meta 输出 shape 与真实实现对齐：真实输出为 {q.size(0), q.size(1), v.size(2)}
// （jagged: total, N, D_v，D_v 可不同于 q 的 head_dim；fp8 输入时输出为 half）。
at::Tensor hstu_jagged_equal_meta(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                  const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& attnBias,
                                  const int64_t maskType, const int64_t maxSeqLen, const double siluScale,
                                  const at::Tensor& seqOffset, const c10::optional<at::Tensor>& numContext,
                                  const c10::optional<at::Tensor>& numTarget,
                                  const c10::optional<int64_t>& targetGroupSize, const c10::optional<double>& alpha,
                                  const bool deterministic)
{
    TORCH_CHECK(q.dim() == 3, "The q should be 3D in jagged layout");
    const bool use_fp8 = (q.scalar_type() == at::kFloat8_e4m3fn);
    at::ScalarType output_dtype = use_fp8 ? at::kHalf : q.scalar_type();
    std::vector<c10::SymInt> shape = {q.sym_size(0), q.sym_size(1), v.sym_size(2)};
    return at::empty_symint(shape, q.options().dtype(output_dtype).device(c10::kMeta));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_jagged_backward_equal_meta(
    const at::Tensor& grad, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor> mask, const c10::optional<at::Tensor> attnBias, const int64_t maskType,
    const int64_t maxSeqLen, const double siluScale, const at::Tensor& seqOffset,
    const c10::optional<at::Tensor>& numContext, const c10::optional<at::Tensor>& numTarget,
    const c10::optional<int64_t>& targetGroupSize, const c10::optional<double>& alpha)
{
    TORCH_CHECK(grad.dim() == 3, "The grad should be 3D in jagged layout");
    auto qGradOutput = at::empty_like(q);
    auto kGradOutput = at::empty_like(k);
    auto vGradOutput = at::empty_like(v);
    at::Tensor attnBiasGradOutput;
    if (attnBias.has_value() && attnBias.value().defined()) {
        attnBiasGradOutput = at::empty_like(attnBias.value());
    } else {
        attnBiasGradOutput = at::Tensor();
    }
    return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, attnBiasGradOutput);
}

// Meta 输出 shape 与真实实现对齐：真实输出为 {q.size(0), q.size(1), v.size(2)}
// （jagged: total, N, D_v，D_v 可不同于 q 的 head_dim；fp8 输入时输出为 half）。
at::Tensor hstu_jagged_delta_meta(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                  const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& attnBias,
                                  const int64_t maskType, const int64_t maxSeqLen, const int64_t maxSeqLenK,
                                  const double siluScale, const at::Tensor& seqOffset, const at::Tensor& seqOffsetK,
                                  const c10::optional<at::Tensor>& numContext,
                                  const c10::optional<at::Tensor>& numTarget,
                                  const c10::optional<int64_t>& targetGroupSize, const c10::optional<double>& alpha,
                                  const bool deterministic)
{
    TORCH_CHECK(q.dim() == 3, "The q should be 3D in jagged layout");
    const bool use_fp8 = (q.scalar_type() == at::kFloat8_e4m3fn);
    at::ScalarType output_dtype = use_fp8 ? at::kHalf : q.scalar_type();
    std::vector<c10::SymInt> shape = {q.sym_size(0), q.sym_size(1), v.sym_size(2)};
    return at::empty_symint(shape, q.options().dtype(output_dtype).device(c10::kMeta));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_jagged_backward_delta_meta(
    const at::Tensor& grad, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor> mask, const c10::optional<at::Tensor> attnBias, const int64_t maskType,
    const int64_t maxSeqLen, const int64_t maxSeqLenK, const double siluScale, const at::Tensor& seqOffsetQ,
    const at::Tensor& seqOffsetK, const c10::optional<at::Tensor>& numContext,
    const c10::optional<at::Tensor>& numTarget, const c10::optional<int64_t>& targetGroupSize,
    const c10::optional<double>& alpha)
{
    TORCH_CHECK(grad.dim() == 3, "The grad should be 3D in jagged layout");
    auto qGradOutput = at::empty_like(q);
    auto kGradOutput = at::empty_like(k);
    auto vGradOutput = at::empty_like(v);
    at::Tensor attnBiasGradOutput;
    if (attnBias.has_value() && attnBias.value().defined()) {
        attnBiasGradOutput = at::empty_like(attnBias.value());
    } else {
        attnBiasGradOutput = at::Tensor();
    }
    return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, attnBiasGradOutput);
}

}  // anonymous namespace

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("hstu_jagged.equal", &hstu_jagged_equal_meta);
    m.impl("hstu_jagged_backward.equal", &hstu_jagged_backward_equal_meta);
    m.impl("hstu_jagged.delta", &hstu_jagged_delta_meta);
    m.impl("hstu_jagged_backward.delta", &hstu_jagged_backward_delta_meta);
}

}  // namespace hstu
