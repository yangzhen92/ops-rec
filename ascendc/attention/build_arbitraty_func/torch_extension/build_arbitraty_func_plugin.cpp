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
#define OPS_REC_CUST_VENDORS "build_arbitrary_func"

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "common_utils.h"
#include "pytorch_npu_helper.hpp"

using namespace at;

at::Tensor build_arbitrary_func_impl_npu(const at::Tensor& offsets, const at::Tensor& ctx_lens,
                                         const at::Tensor& tgt_lens, int64_t target_group_size, int64_t total_seqlen)
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

    auto offsets_conti = offsets.contiguous();
    auto ctx_lens_conti = ctx_lens.contiguous();
    auto tgt_lens_conti = tgt_lens.contiguous();

    // Kernel explicitly writes all 4 columns, so empty (not zeros) is safe.
    at::Tensor arbitrary_func = at::empty({1, 1, total_seqlen, 4}, offsets_conti.options());
    if (total_seqlen == 0) return arbitrary_func;
    
    EXEC_NPU_CMD(aclnnBuildArbitraryFunc, offsets_conti, ctx_lens_conti, tgt_lens_conti, target_group_size, total_seqlen,
                 arbitrary_func);
    return arbitrary_func;
}

// 注册 build_arbitrary_func 的 schema
TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def(
        "build_arbitrary_func(Tensor offsets, Tensor ctx_lens, Tensor tgt_lens, "
        "int target_group_size, int total_seqlen) -> Tensor");
}

// NPU 设备实现
TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("build_arbitrary_func", &build_arbitrary_func_impl_npu);
}

namespace {

at::Tensor build_arbitrary_func_meta(const at::Tensor& offsets, const at::Tensor& ctx_lens, const at::Tensor& tgt_lens,
                                     int64_t target_group_size, int64_t total_seqlen)
{
    TORCH_CHECK(total_seqlen >= 0, "total_seqlen must be non-negative");
    return at::empty_symint({1, 1, total_seqlen, 4}, offsets.options());
}

}  // anonymous namespace

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("build_arbitrary_func", &build_arbitrary_func_meta);
}
