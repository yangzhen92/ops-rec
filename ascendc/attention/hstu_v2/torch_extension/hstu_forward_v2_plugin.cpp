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
============================================================================== */
#define OPS_REC_CUST_VENDORS "hstu_v2"
#include <cstdint>
#include "c10/core/ScalarType.h"
#include "torch/types.h"
#include "hstu_common.h"
#include "hstu_v2_param_check.h"
#include "common_utils.h"

using tensor_list = std::vector<at::Tensor>;

at::Tensor hstu_forward_v2_impl_npu(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& rab, const int64_t maskType, const int64_t maxSeqLenQ,
    const c10::optional<int64_t> maxSeqLenK, const c10::optional<double> siluScale_, const at::Tensor& seqOffset,
    const c10::optional<at::Tensor>& seqOffsetK, const c10::optional<at::Tensor>& numContext,
    const c10::optional<at::Tensor>& numTarget, const c10::optional<int64_t>& targetGroupSize,
    const c10::optional<double>& alpha, const c10::optional<at::Tensor>& metadata,
    const c10::optional<tensor_list>& sparseInfo, const int64_t windowSizeLeft, const int64_t windowSizeRight,
    const c10::optional<at::Tensor>& planAf, const bool deterministic)
{
    const int64_t realmaxSeqLenK = maxSeqLenK.value_or(maxSeqLenQ);
    auto acSeqOffset = seqOffset;
    auto acSeqOffsetK = seqOffsetK.has_value() ? seqOffsetK.value() : seqOffset;
    hstu_v2::HstuForwardV2ParamChecker(q, k, v, acSeqOffset, acSeqOffsetK, maxSeqLenQ, realmaxSeqLenK, rab);
    hstu::MaskCheck(maskType, mask.has_value() && mask->defined());
    TORCH_CHECK(windowSizeLeft >= -1 && windowSizeRight >= -1, "window_size_left and window_size_right must be >= -1");
    TORCH_CHECK(windowSizeLeft < realmaxSeqLenK && windowSizeRight < realmaxSeqLenK,
                "window_size must be < max_seqlen_k");
    const bool isLocal = windowSizeLeft >= 0 || windowSizeRight > 0;
    TORCH_CHECK(isLocal || maskType == MASK_TYPE_TRIL || maskType == MASK_TYPE_NONE,
                "hstu_forward_v2 currently supports mask_type 0 (causal) or 2 (none)");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = rab.value_or(at::Tensor());
    auto maskNpu = mask.value_or(at::Tensor());

    auto acNumContext =
        CheckOptionalTensorIsNotNone(numContext) ? numContext.value().to(acSeqOffset.scalar_type()) : at::Tensor();
    auto acNumTarget =
        CheckOptionalTensorIsNotNone(numTarget) ? numTarget.value().to(acSeqOffset.scalar_type()) : at::Tensor();
    auto acTargetGroupSize = targetGroupSize.value_or(0);
    double realAlpha = alpha.value_or(1.0);
    TORCH_CHECK(realAlpha > 0.0, "alpha must be greater than 0");
    // metadata 为可选: 未传/None → 空 tensor → aclnn 视作 null → kernel nullptr → 旧设备现算分核(零回归)
    at::Tensor acMetadata = CheckOptionalTensorIsNotNone(metadata) ? metadata.value().to(at::kInt) : at::Tensor();
    bool use_fp8 = (q.scalar_type() == at::kFloat8_e4m3fn);
    at::ScalarType output_dtype = use_fp8 ? at::kHalf : denseQ.scalar_type();

    auto attnOutput = at::zeros({denseQ.size(0), denseQ.size(1), denseV.size(2)}, denseQ.options().dtype(output_dtype));
    auto outputAccum =
        deterministic ? at::Tensor()
                      : at::zeros({denseQ.size(1), denseQ.size(0), denseV.size(2)}, denseQ.options().dtype(at::kFloat));

    TORCH_CHECK(attnOutput.defined(), "attnOutput is not defined!");
    TORCH_CHECK(attnOutput.storage().data_ptr() != nullptr, "attnOutput storage is null!");
    TORCH_CHECK(attnOutput.is_contiguous(), "attnOutput must be contiguous!");

    auto siluScale = siluScale_.value_or(0.0);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / static_cast<double>(maxSeqLenQ) : siluScale;
    auto inMetadata = acMetadata.defined() ? acMetadata.contiguous() : at::Tensor();

    // op exec —— metadata 紧跟 num_target(与 OpDef 输入顺序一致: ...num_target, metadata),置于所有 attr 之前
    tensor_list emptySparseInfo{at::empty({0}, denseQ.options().dtype(at::kInt))};
    tensor_list sparseVec = sparseInfo.value_or(emptySparseInfo);
    TORCH_CHECK(!sparseInfo.has_value() || sparseVec.size() == 9,
                "sparse_info must contain exactly nine packed tensors");
    at::TensorList inSparseInfo(sparseVec);
    EXEC_NPU_CMD(aclnnHstuForwardV2, denseQ, denseK, denseV, maskNpu, denseBias, acSeqOffset, acSeqOffsetK,
                 acNumContext, acNumTarget, inMetadata, outputAccum, inSparseInfo, maxSeqLenQ, realmaxSeqLenK,
                 realSiluScale, acTargetGroupSize, realAlpha, maskType, windowSizeLeft, windowSizeRight, deterministic,
                 attnOutput);

    return attnOutput;
}

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("hstu_forward_v2(Tensor q, "
          "            Tensor k, "
          "            Tensor v, "
          "            Tensor? mask, "
          "            Tensor? rab, "
          "            int mask_type, "
          "            int max_seq_len, "
          "            int? max_seq_len_k, "
          "            float? silu_scale, "
          "            Tensor seq_offset, "
          "            Tensor? seq_offset_k, "
          "            Tensor? num_context, "
          "            Tensor? num_target, "
          "            int? target_group_size, "
          "            float? alpha, "
          "            Tensor? metadata=None, "
          "            Tensor[]? sparse_info=None, "
          "            int window_size_left=-1, "
          "            int window_size_right=-1, "
          "            Tensor? plan_af=None, "
          "            bool deterministic=True) -> Tensor");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("hstu_forward_v2", TORCH_FN(hstu_forward_v2_impl_npu));
}

namespace {
at::Tensor hstu_forward_v2_meta(const at::Tensor& q, const at::Tensor&, const at::Tensor& v,
                                const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&, const int64_t,
                                const int64_t, const c10::optional<int64_t>, const c10::optional<double>,
                                const at::Tensor&, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
                                const c10::optional<at::Tensor>&, const c10::optional<int64_t>&,
                                const c10::optional<double>&, const c10::optional<at::Tensor>&,
                                const c10::optional<tensor_list>&, const int64_t, const int64_t,
                                const c10::optional<at::Tensor>&, const bool)
{
    auto sizes = q.sym_sizes().vec();
    sizes[2] = v.sym_size(2);
    return at::empty_symint(sizes, q.options().device(c10::kMeta));
}
}  // namespace

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("hstu_forward_v2", &hstu_forward_v2_meta);
}
