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
#define OPS_REC_CUST_VENDORS "arbitrary_func"

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "common_utils.h"
#include "pytorch_npu_helper.hpp"

using namespace at;

constexpr uint32_t N_FUNC = 4;
constexpr uint32_t SINGLE_INTERVAL_N_FUNC = 2;
constexpr uint32_t DOUBLE_INTERVAL_N_FUNC = 4;

at::Tensor build_causal_mask_arbitrary_func_impl_npu(const at::Tensor& offsets, int64_t total_seqlen)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_npu_device({offsets}, {"offsets"});
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1,
                "offsets must be a one-dimensional int32 tensor");
    TORCH_CHECK(total_seqlen >= 0 && total_seqlen <= INT32_MAX, "total_seqlen must fit non-negative int32");
    auto offsets_conti = offsets.contiguous();
    TORCH_CHECK(offsets_conti[-1].item<int32_t>() == total_seqlen, "total_seqlen must equal offsets[-1]");
    auto output = at::empty({1, 1, total_seqlen, SINGLE_INTERVAL_N_FUNC}, offsets.options());
    if (total_seqlen != 0) EXEC_NPU_CMD(aclnnBuildCausalMaskArbitraryFunc, offsets_conti, total_seqlen, output);
    return output;
}

at::Tensor build_sliding_window_mask_arbitrary_func_impl_npu(const at::Tensor& offsets,
                                                              const at::Tensor& window_sizes,
                                                              int64_t total_seqlen)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_non_empty(window_sizes, "window_sizes");
    check_tensor_npu_device({offsets, window_sizes}, {"offsets", "window_sizes"});
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1, "offsets must be one-dimensional int32");
    TORCH_CHECK(window_sizes.scalar_type() == at::kInt && window_sizes.dim() == 1,
                "window_sizes must be one-dimensional int32");
    TORCH_CHECK(window_sizes.numel() == offsets.numel() - 1, "window_sizes must have shape [batch_size]");
    TORCH_CHECK(total_seqlen >= 0 && total_seqlen <= INT32_MAX, "total_seqlen must fit non-negative int32");
    auto offsets_conti = offsets.contiguous();
    auto windows_conti = window_sizes.contiguous();
    TORCH_CHECK(offsets_conti[-1].item<int32_t>() == total_seqlen, "total_seqlen must equal offsets[-1]");
    auto output = at::empty({1, 1, total_seqlen, SINGLE_INTERVAL_N_FUNC}, offsets.options());
    if (total_seqlen != 0) EXEC_NPU_CMD(aclnnBuildSlidingWindowMaskArbitraryFunc, offsets_conti, windows_conti,
                                        total_seqlen, output);
    return output;
}

at::Tensor build_prefix_lm_mask_arbitrary_func_impl_npu(const at::Tensor& offsets,
                                                        const at::Tensor& prefix_lens,
                                                        int64_t total_seqlen)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_non_empty(prefix_lens, "prefix_lens");
    check_tensor_npu_device({offsets, prefix_lens}, {"offsets", "prefix_lens"});
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1, "offsets must be one-dimensional int32");
    TORCH_CHECK(prefix_lens.scalar_type() == at::kInt && prefix_lens.dim() == 1,
                "prefix_lens must be one-dimensional int32");
    TORCH_CHECK(prefix_lens.numel() == offsets.numel() - 1, "prefix_lens must have shape [batch_size]");
    TORCH_CHECK(total_seqlen >= 0 && total_seqlen <= INT32_MAX, "total_seqlen must fit non-negative int32");
    auto offsets_conti = offsets.contiguous();
    auto prefix_conti = prefix_lens.contiguous();
    TORCH_CHECK(offsets_conti[-1].item<int32_t>() == total_seqlen, "total_seqlen must equal offsets[-1]");
    auto output = at::empty({1, 1, total_seqlen, SINGLE_INTERVAL_N_FUNC}, offsets.options());
    if (total_seqlen != 0) EXEC_NPU_CMD(aclnnBuildPrefixLmMaskArbitraryFunc, offsets_conti, prefix_conti,
                                        total_seqlen, output);
    return output;
}

at::Tensor build_group_wise_mask_arbitrary_func_impl_npu(const at::Tensor& offsets,
                                                         const at::Tensor& group_sizes,
                                                         int64_t total_seqlen)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_non_empty(group_sizes, "group_sizes");
    check_tensor_npu_device({offsets, group_sizes}, {"offsets", "group_sizes"});
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1, "offsets must be one-dimensional int32");
    TORCH_CHECK(group_sizes.scalar_type() == at::kInt && group_sizes.dim() == 1,
                "group_sizes must be one-dimensional int32");
    TORCH_CHECK(group_sizes.numel() == offsets.numel() - 1, "group_sizes must have shape [batch_size]");
    TORCH_CHECK(total_seqlen >= 0 && total_seqlen <= INT32_MAX, "total_seqlen must fit non-negative int32");
    auto offsets_conti = offsets.contiguous();
    auto groups_conti = group_sizes.contiguous();
    TORCH_CHECK(offsets_conti[-1].item<int32_t>() == total_seqlen, "total_seqlen must equal offsets[-1]");
    auto output = at::empty({1, 1, total_seqlen, SINGLE_INTERVAL_N_FUNC}, offsets.options());
    if (total_seqlen != 0) EXEC_NPU_CMD(aclnnBuildGroupWiseMaskArbitraryFunc, offsets_conti, groups_conti,
                                        total_seqlen, output);
    return output;
}

at::Tensor build_block_causal_mask_arbitrary_func_impl_npu(const at::Tensor& offsets,
                                                           const at::Tensor& block_sizes,
                                                           int64_t total_seqlen)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_non_empty(block_sizes, "block_sizes");
    check_tensor_npu_device({offsets, block_sizes}, {"offsets", "block_sizes"});
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1, "offsets must be one-dimensional int32");
    TORCH_CHECK(block_sizes.scalar_type() == at::kInt && block_sizes.dim() == 1,
                "block_sizes must be one-dimensional int32");
    TORCH_CHECK(block_sizes.numel() == offsets.numel() - 1, "block_sizes must have shape [batch_size]");
    TORCH_CHECK(total_seqlen >= 0 && total_seqlen <= INT32_MAX, "total_seqlen must fit non-negative int32");
    auto offsets_conti = offsets.contiguous();
    auto blocks_conti = block_sizes.contiguous();
    TORCH_CHECK(offsets_conti[-1].item<int32_t>() == total_seqlen, "total_seqlen must equal offsets[-1]");
    auto output = at::empty({1, 1, total_seqlen, SINGLE_INTERVAL_N_FUNC}, offsets.options());
    if (total_seqlen != 0) EXEC_NPU_CMD(aclnnBuildBlockCausalMaskArbitraryFunc, offsets_conti, blocks_conti,
                                        total_seqlen, output);
    return output;
}

at::Tensor build_grouped_causal_mask_arbitrary_func_impl_npu(const at::Tensor& offsets,
                                                             const at::Tensor& group_sizes,
                                                             int64_t total_seqlen)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_non_empty(group_sizes, "group_sizes");
    check_tensor_npu_device({offsets, group_sizes}, {"offsets", "group_sizes"});
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1, "offsets must be one-dimensional int32");
    TORCH_CHECK(group_sizes.scalar_type() == at::kInt && group_sizes.dim() == 1,
                "group_sizes must be one-dimensional int32");
    TORCH_CHECK(group_sizes.numel() == offsets.numel() - 1, "group_sizes must have shape [batch_size]");
    TORCH_CHECK(total_seqlen >= 0 && total_seqlen <= INT32_MAX, "total_seqlen must fit non-negative int32");
    auto offsets_conti = offsets.contiguous();
    auto groups_conti = group_sizes.contiguous();
    TORCH_CHECK(offsets_conti[-1].item<int32_t>() == total_seqlen, "total_seqlen must equal offsets[-1]");
    auto output = at::empty({1, 1, total_seqlen, SINGLE_INTERVAL_N_FUNC}, offsets.options());
    if (total_seqlen != 0) EXEC_NPU_CMD(aclnnBuildGroupedCausalMaskArbitraryFunc, offsets_conti, groups_conti,
                                        total_seqlen, output);
    return output;
}

at::Tensor build_block_diagonal_mask_arbitrary_func_impl_npu(const at::Tensor& offsets, int64_t total_seqlen)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_npu_device({offsets}, {"offsets"});
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1,
                "offsets must be a one-dimensional int32 tensor");
    TORCH_CHECK(total_seqlen >= 0 && total_seqlen <= INT32_MAX, "total_seqlen must fit non-negative int32");
    auto offsets_conti = offsets.contiguous();
    TORCH_CHECK(offsets_conti[-1].item<int32_t>() == total_seqlen, "total_seqlen must equal offsets[-1]");
    auto output = at::empty({1, 1, total_seqlen, SINGLE_INTERVAL_N_FUNC}, offsets.options());
    if (total_seqlen != 0) EXEC_NPU_CMD(aclnnBuildBlockDiagonalMaskArbitraryFunc, offsets_conti, total_seqlen, output);
    return output;
}

at::Tensor build_upper_triangular_mask_arbitrary_func_impl_npu(const at::Tensor& offsets, int64_t total_seqlen)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_npu_device({offsets}, {"offsets"});
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1,
                "offsets must be a one-dimensional int32 tensor");
    TORCH_CHECK(total_seqlen >= 0 && total_seqlen <= INT32_MAX, "total_seqlen must fit non-negative int32");
    auto offsets_conti = offsets.contiguous();
    TORCH_CHECK(offsets_conti[-1].item<int32_t>() == total_seqlen, "total_seqlen must equal offsets[-1]");
    auto output = at::empty({1, 1, total_seqlen, SINGLE_INTERVAL_N_FUNC}, offsets.options());
    if (total_seqlen != 0) EXEC_NPU_CMD(aclnnBuildUpperTriangularMaskArbitraryFunc, offsets_conti,
                                        total_seqlen, output);
    return output;
}

at::Tensor build_padding_row_mask_arbitrary_func_impl_npu(const at::Tensor& offsets,
                                                          const at::Tensor& valid_lens,
                                                          int64_t total_seqlen)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_non_empty(valid_lens, "valid_lens");
    check_tensor_npu_device({offsets, valid_lens}, {"offsets", "valid_lens"});
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1, "offsets must be one-dimensional int32");
    TORCH_CHECK(valid_lens.scalar_type() == at::kInt && valid_lens.dim() == 1,
                "valid_lens must be one-dimensional int32");
    TORCH_CHECK(valid_lens.numel() == offsets.numel() - 1, "valid_lens must have shape [batch_size]");
    TORCH_CHECK(total_seqlen >= 0 && total_seqlen <= INT32_MAX, "total_seqlen must fit non-negative int32");
    auto offsets_conti = offsets.contiguous();
    auto valid_conti = valid_lens.contiguous();
    TORCH_CHECK(offsets_conti[-1].item<int32_t>() == total_seqlen, "total_seqlen must equal offsets[-1]");
    auto output = at::empty({1, 1, total_seqlen, SINGLE_INTERVAL_N_FUNC}, offsets.options());
    if (total_seqlen != 0) EXEC_NPU_CMD(aclnnBuildPaddingRowMaskArbitraryFunc, offsets_conti, valid_conti,
                                        total_seqlen, output);
    return output;
}

at::Tensor build_target_group_block_mask_arbitrary_func_impl_npu(const at::Tensor& offsets,
                                                                 const at::Tensor& target_lens,
                                                                 const at::Tensor& group_sizes,
                                                                 int64_t total_seqlen)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_non_empty(target_lens, "target_lens");
    check_tensor_non_empty(group_sizes, "group_sizes");
    check_tensor_npu_device({offsets, target_lens, group_sizes}, {"offsets", "target_lens", "group_sizes"});
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1, "offsets must be one-dimensional int32");
    TORCH_CHECK(target_lens.scalar_type() == at::kInt && target_lens.dim() == 1,
                "target_lens must be one-dimensional int32");
    TORCH_CHECK(group_sizes.scalar_type() == at::kInt && group_sizes.dim() == 1,
                "group_sizes must be one-dimensional int32");
    TORCH_CHECK(target_lens.numel() == offsets.numel() - 1, "target_lens must have shape [batch_size]");
    TORCH_CHECK(group_sizes.numel() == offsets.numel() - 1, "group_sizes must have shape [batch_size]");
    TORCH_CHECK(total_seqlen >= 0 && total_seqlen <= INT32_MAX, "total_seqlen must fit non-negative int32");
    auto offsets_conti = offsets.contiguous();
    auto targets_conti = target_lens.contiguous();
    auto groups_conti = group_sizes.contiguous();
    TORCH_CHECK(offsets_conti[-1].item<int32_t>() == total_seqlen, "total_seqlen must equal offsets[-1]");
    auto output = at::empty({1, 1, total_seqlen, DOUBLE_INTERVAL_N_FUNC}, offsets.options());
    if (total_seqlen != 0) EXEC_NPU_CMD(aclnnBuildTargetGroupBlockMaskArbitraryFunc, offsets_conti, targets_conti,
                                        groups_conti, total_seqlen, output);
    return output;
}

at::Tensor build_gist_mask_arbitrary_func_impl_npu(const at::Tensor& offsets,
                                                   const at::Tensor& pre_doc_lens,
                                                   const at::Tensor& gist_starts,
                                                   const at::Tensor& gist_ends,
                                                   int64_t total_seqlen)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_non_empty(pre_doc_lens, "pre_doc_lens");
    check_tensor_non_empty(gist_starts, "gist_starts");
    check_tensor_non_empty(gist_ends, "gist_ends");
    check_tensor_npu_device({offsets, pre_doc_lens, gist_starts, gist_ends},
                            {"offsets", "pre_doc_lens", "gist_starts", "gist_ends"});
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1, "offsets must be one-dimensional int32");
    TORCH_CHECK(pre_doc_lens.scalar_type() == at::kInt && pre_doc_lens.dim() == 1,
                "pre_doc_lens must be one-dimensional int32");
    TORCH_CHECK(gist_starts.scalar_type() == at::kInt && gist_starts.dim() == 1,
                "gist_starts must be one-dimensional int32");
    TORCH_CHECK(gist_ends.scalar_type() == at::kInt && gist_ends.dim() == 1,
                "gist_ends must be one-dimensional int32");
    const int64_t batch_size = offsets.numel() - 1;
    TORCH_CHECK(pre_doc_lens.numel() == batch_size && gist_starts.numel() == batch_size &&
                gist_ends.numel() == batch_size,
                "pre_doc_lens/gist_starts/gist_ends must have shape [batch_size]");
    TORCH_CHECK(total_seqlen >= 0 && total_seqlen <= INT32_MAX, "total_seqlen must fit non-negative int32");
    auto offsets_conti = offsets.contiguous();
    auto pre_doc_conti = pre_doc_lens.contiguous();
    auto gist_start_conti = gist_starts.contiguous();
    auto gist_end_conti = gist_ends.contiguous();
    TORCH_CHECK(offsets_conti[-1].item<int32_t>() == total_seqlen, "total_seqlen must equal offsets[-1]");
    auto output = at::empty({1, 1, total_seqlen, DOUBLE_INTERVAL_N_FUNC}, offsets.options());
    if (total_seqlen != 0) {
        EXEC_NPU_CMD(aclnnBuildGistMaskArbitraryFunc, offsets_conti, pre_doc_conti, gist_start_conti,
                     gist_end_conti, total_seqlen, output);
    }
    return output;
}

at::Tensor build_hstu_mask_arbitrary_func_impl_npu(const at::Tensor& offsets, const at::Tensor& ctx_lens,
                                                   const at::Tensor& tgt_lens, int64_t target_group_size,
                                                   int64_t total_seqlen)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_non_empty(ctx_lens, "ctx_lens");
    check_tensor_non_empty(tgt_lens, "tgt_lens");

    std::vector<at::Tensor> tensors = {offsets, ctx_lens, tgt_lens};
    std::vector<std::string> names = {"offsets", "ctx_lens", "tgt_lens"};
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(offsets.scalar_type() == at::kInt, "offsets must be int32");
    TORCH_CHECK(ctx_lens.scalar_type() == at::kInt, "ctx_lens must be int32");
    TORCH_CHECK(tgt_lens.scalar_type() == at::kInt, "tgt_lens must be int32");
    TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D");
    TORCH_CHECK(ctx_lens.dim() == 1, "ctx_lens must be 1D");
    TORCH_CHECK(tgt_lens.dim() == 1, "tgt_lens must be 1D");
    TORCH_CHECK(target_group_size > 0, "target_group_size must > 0");
    TORCH_CHECK(total_seqlen >= 0, "total_seqlen must be non-negative");

    const int64_t batch_size = offsets.size(0) - 1;
    TORCH_CHECK(ctx_lens.size(0) == batch_size, "ctx_lens.size(0) (", ctx_lens.size(0),
                ") must equal offsets.size(0)-1 (", batch_size, "), i.e. ctx_lens shape must be [batch_size]");
    TORCH_CHECK(tgt_lens.size(0) == batch_size, "tgt_lens.size(0) (", tgt_lens.size(0),
                ") must equal offsets.size(0)-1 (", batch_size, "), i.e. tgt_lens shape must be [batch_size]");

    const int32_t offsets_last = offsets[-1].item<int32_t>();
    TORCH_CHECK(total_seqlen == offsets_last, "total_seqlen (", total_seqlen, ") must equal offsets[-1] (",
                offsets_last, ")");

    auto offsets_conti = offsets.contiguous();
    auto ctx_lens_conti = ctx_lens.contiguous();
    auto tgt_lens_conti = tgt_lens.contiguous();

    // Kernel explicitly writes all 4 columns, so empty (not zeros) is safe.
    at::Tensor arbitrary_func = at::empty({1, 1, total_seqlen, N_FUNC}, offsets_conti.options());
    if (total_seqlen == 0)
        return arbitrary_func;

    EXEC_NPU_CMD(aclnnBuildHstuMaskArbitraryFunc, offsets_conti, ctx_lens_conti, tgt_lens_conti, target_group_size,
                 total_seqlen, arbitrary_func);
    return arbitrary_func;
}

at::Tensor build_sla_mask_arbitrary_func_impl_npu(const at::Tensor& offsets, int64_t total_seqlen, int64_t sla_k1,
                                                  int64_t sla_k2, int64_t contextual_seq_len,
                                                  const c10::optional<at::Tensor>& num_targets)
{
    check_tensor_non_empty(offsets, "offsets");
    check_tensor_npu_device({offsets}, {"offsets"});

    TORCH_CHECK(offsets.scalar_type() == at::kInt, "offsets must be int32");
    TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D");
    TORCH_CHECK(total_seqlen >= 0 && total_seqlen <= INT32_MAX, "total_seqlen must be in [0, INT32_MAX]");
    TORCH_CHECK(sla_k1 >= 0 && sla_k1 <= INT32_MAX, "sla_k1 must be in [0, INT32_MAX]");
    TORCH_CHECK(sla_k2 >= 0 && sla_k2 <= INT32_MAX, "sla_k2 must be in [0, INT32_MAX]");
    TORCH_CHECK(contextual_seq_len >= 0 && contextual_seq_len <= INT32_MAX,
                "contextual_seq_len must be in [0, INT32_MAX]");

    auto offsets_conti = offsets.contiguous();

    const int32_t offsetsTotal = offsets[-1].item<int32_t>();
    TORCH_CHECK(total_seqlen == offsetsTotal, "total_seqlen must equal offsets[-1] (total tokens), got ", total_seqlen,
                " vs ", offsetsTotal);

    // num_targets 可选，缺省按全 0 处理（等价于全部为 history token）
    at::Tensor num_targets_conti;
    if (num_targets.has_value() && num_targets->defined() && num_targets->numel() > 0) {
        check_tensor_npu_device({*num_targets}, {"num_targets"});
        TORCH_CHECK(num_targets->scalar_type() == at::kInt, "num_targets must be int32");
        TORCH_CHECK(num_targets->dim() == 1, "num_targets must be 1D");
        TORCH_CHECK(num_targets->size(0) == offsets_conti.size(0) - 1,
                    "num_targets must have length offsets.size(0)-1 (batch_size), got ", num_targets->size(0), " vs ",
                    offsets_conti.size(0) - 1);
        num_targets_conti = num_targets->contiguous();
    } else {
        int64_t batch_size = offsets_conti.size(0) - 1;
        num_targets_conti = at::zeros({batch_size}, offsets_conti.options());
    }

    at::Tensor arbitrary_func = at::empty({1, 1, total_seqlen, N_FUNC}, offsets_conti.options());
    if (total_seqlen == 0)
        return arbitrary_func;

    EXEC_NPU_CMD(aclnnBuildSlaMaskArbitraryFunc, offsets_conti, num_targets_conti, sla_k1, sla_k2, contextual_seq_len,
                 total_seqlen, arbitrary_func);
    return arbitrary_func;
}

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("build_hstu_mask_arbitrary_func(Tensor offsets, Tensor ctx_lens, Tensor tgt_lens, "
          "int target_group_size, int total_seqlen) -> Tensor");
    m.def("build_sla_mask_arbitrary_func(Tensor offsets, int total_seqlen, int sla_k1, int sla_k2, "
          "int contextual_seq_len, Tensor? num_targets=None) -> Tensor");
    m.def("build_causal_mask_arbitrary_func(Tensor offsets, int total_seqlen) -> Tensor");
    m.def("build_sliding_window_mask_arbitrary_func(Tensor offsets, Tensor window_sizes, int total_seqlen) -> Tensor");
    m.def("build_prefix_lm_mask_arbitrary_func(Tensor offsets, Tensor prefix_lens, int total_seqlen) -> Tensor");
    m.def("build_group_wise_mask_arbitrary_func(Tensor offsets, Tensor group_sizes, int total_seqlen) -> Tensor");
    m.def("build_block_causal_mask_arbitrary_func(Tensor offsets, Tensor block_sizes, int total_seqlen) -> Tensor");
    m.def("build_grouped_causal_mask_arbitrary_func(Tensor offsets, Tensor group_sizes, int total_seqlen) -> Tensor");
    m.def("build_block_diagonal_mask_arbitrary_func(Tensor offsets, int total_seqlen) -> Tensor");
    m.def("build_upper_triangular_mask_arbitrary_func(Tensor offsets, int total_seqlen) -> Tensor");
    m.def("build_padding_row_mask_arbitrary_func(Tensor offsets, Tensor valid_lens, int total_seqlen) -> Tensor");
    m.def("build_target_group_block_mask_arbitrary_func(Tensor offsets, Tensor target_lens, Tensor group_sizes, "
          "int total_seqlen) -> Tensor");
    m.def("build_gist_mask_arbitrary_func(Tensor offsets, Tensor pre_doc_lens, Tensor gist_starts, "
          "Tensor gist_ends, int total_seqlen) -> Tensor");
}

// NPU 设备实现
TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("build_hstu_mask_arbitrary_func", &build_hstu_mask_arbitrary_func_impl_npu);
    m.impl("build_sla_mask_arbitrary_func", &build_sla_mask_arbitrary_func_impl_npu);
    m.impl("build_causal_mask_arbitrary_func", &build_causal_mask_arbitrary_func_impl_npu);
    m.impl("build_sliding_window_mask_arbitrary_func", &build_sliding_window_mask_arbitrary_func_impl_npu);
    m.impl("build_prefix_lm_mask_arbitrary_func", &build_prefix_lm_mask_arbitrary_func_impl_npu);
    m.impl("build_group_wise_mask_arbitrary_func", &build_group_wise_mask_arbitrary_func_impl_npu);
    m.impl("build_block_causal_mask_arbitrary_func", &build_block_causal_mask_arbitrary_func_impl_npu);
    m.impl("build_grouped_causal_mask_arbitrary_func", &build_grouped_causal_mask_arbitrary_func_impl_npu);
    m.impl("build_block_diagonal_mask_arbitrary_func", &build_block_diagonal_mask_arbitrary_func_impl_npu);
    m.impl("build_upper_triangular_mask_arbitrary_func", &build_upper_triangular_mask_arbitrary_func_impl_npu);
    m.impl("build_padding_row_mask_arbitrary_func", &build_padding_row_mask_arbitrary_func_impl_npu);
    m.impl("build_target_group_block_mask_arbitrary_func", &build_target_group_block_mask_arbitrary_func_impl_npu);
    m.impl("build_gist_mask_arbitrary_func", &build_gist_mask_arbitrary_func_impl_npu);
}

namespace {

at::Tensor build_named_mask_meta(const at::Tensor& offsets, int64_t total_seqlen, int64_t n_func)
{
    TORCH_CHECK(total_seqlen >= 0, "total_seqlen must be non-negative");
    return at::empty_symint({1, 1, total_seqlen, n_func}, offsets.options());
}

at::Tensor build_causal_mask_arbitrary_func_meta(const at::Tensor& offsets, int64_t total_seqlen)
{ return build_named_mask_meta(offsets, total_seqlen, SINGLE_INTERVAL_N_FUNC); }
at::Tensor build_sliding_window_mask_arbitrary_func_meta(const at::Tensor& offsets, const at::Tensor&,
                                                         int64_t total_seqlen)
{ return build_named_mask_meta(offsets, total_seqlen, SINGLE_INTERVAL_N_FUNC); }
at::Tensor build_prefix_lm_mask_arbitrary_func_meta(const at::Tensor& offsets, const at::Tensor&,
                                                    int64_t total_seqlen)
{ return build_named_mask_meta(offsets, total_seqlen, SINGLE_INTERVAL_N_FUNC); }
at::Tensor build_group_wise_mask_arbitrary_func_meta(const at::Tensor& offsets, const at::Tensor&,
                                                     int64_t total_seqlen)
{ return build_named_mask_meta(offsets, total_seqlen, SINGLE_INTERVAL_N_FUNC); }
at::Tensor build_block_causal_mask_arbitrary_func_meta(const at::Tensor& offsets, const at::Tensor&,
                                                       int64_t total_seqlen)
{ return build_named_mask_meta(offsets, total_seqlen, SINGLE_INTERVAL_N_FUNC); }
at::Tensor build_grouped_causal_mask_arbitrary_func_meta(const at::Tensor& offsets, const at::Tensor&,
                                                         int64_t total_seqlen)
{ return build_named_mask_meta(offsets, total_seqlen, SINGLE_INTERVAL_N_FUNC); }
at::Tensor build_block_diagonal_mask_arbitrary_func_meta(const at::Tensor& offsets, int64_t total_seqlen)
{ return build_named_mask_meta(offsets, total_seqlen, SINGLE_INTERVAL_N_FUNC); }
at::Tensor build_upper_triangular_mask_arbitrary_func_meta(const at::Tensor& offsets, int64_t total_seqlen)
{ return build_named_mask_meta(offsets, total_seqlen, SINGLE_INTERVAL_N_FUNC); }
at::Tensor build_padding_row_mask_arbitrary_func_meta(const at::Tensor& offsets, const at::Tensor&,
                                                      int64_t total_seqlen)
{ return build_named_mask_meta(offsets, total_seqlen, SINGLE_INTERVAL_N_FUNC); }
at::Tensor build_target_group_block_mask_arbitrary_func_meta(const at::Tensor& offsets, const at::Tensor&,
                                                             const at::Tensor&, int64_t total_seqlen)
{ return build_named_mask_meta(offsets, total_seqlen, DOUBLE_INTERVAL_N_FUNC); }
at::Tensor build_gist_mask_arbitrary_func_meta(const at::Tensor& offsets, const at::Tensor&, const at::Tensor&,
                                               const at::Tensor&, int64_t total_seqlen)
{ return build_named_mask_meta(offsets, total_seqlen, DOUBLE_INTERVAL_N_FUNC); }

at::Tensor build_hstu_mask_arbitrary_func_meta(const at::Tensor& offsets, const at::Tensor& ctx_lens,
                                               const at::Tensor& tgt_lens, int64_t target_group_size,
                                               int64_t total_seqlen)
{
    TORCH_CHECK(total_seqlen >= 0, "total_seqlen must be non-negative");
    return at::empty_symint({1, 1, total_seqlen, N_FUNC}, offsets.options());
}

at::Tensor build_sla_mask_arbitrary_func_meta(const at::Tensor& offsets, int64_t total_seqlen, int64_t sla_k1,
                                              int64_t sla_k2, int64_t contextual_seq_len,
                                              const c10::optional<at::Tensor>& num_targets)
{
    TORCH_CHECK(total_seqlen >= 0, "total_seqlen must be non-negative");
    return at::empty_symint({1, 1, total_seqlen, N_FUNC}, offsets.options());
}
}  // anonymous namespace

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("build_hstu_mask_arbitrary_func", &build_hstu_mask_arbitrary_func_meta);
    m.impl("build_sla_mask_arbitrary_func", &build_sla_mask_arbitrary_func_meta);
    m.impl("build_causal_mask_arbitrary_func", &build_causal_mask_arbitrary_func_meta);
    m.impl("build_sliding_window_mask_arbitrary_func", &build_sliding_window_mask_arbitrary_func_meta);
    m.impl("build_prefix_lm_mask_arbitrary_func", &build_prefix_lm_mask_arbitrary_func_meta);
    m.impl("build_group_wise_mask_arbitrary_func", &build_group_wise_mask_arbitrary_func_meta);
    m.impl("build_block_causal_mask_arbitrary_func", &build_block_causal_mask_arbitrary_func_meta);
    m.impl("build_grouped_causal_mask_arbitrary_func", &build_grouped_causal_mask_arbitrary_func_meta);
    m.impl("build_block_diagonal_mask_arbitrary_func", &build_block_diagonal_mask_arbitrary_func_meta);
    m.impl("build_upper_triangular_mask_arbitrary_func", &build_upper_triangular_mask_arbitrary_func_meta);
    m.impl("build_padding_row_mask_arbitrary_func", &build_padding_row_mask_arbitrary_func_meta);
    m.impl("build_target_group_block_mask_arbitrary_func", &build_target_group_block_mask_arbitrary_func_meta);
    m.impl("build_gist_mask_arbitrary_func", &build_gist_mask_arbitrary_func_meta);
}
