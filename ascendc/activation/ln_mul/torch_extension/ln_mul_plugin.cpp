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

#include <torch/library.h>
#include "aclnn_common.h"


namespace {

at::Tensor ln_mul_impl_npu(const at::Tensor &x, const at::Tensor &u,
                           const at::Tensor &gamma, const at::Tensor &beta)
{
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(u.scalar_type() == at::kFloat, "u must be float32");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(u.dim() == 2, "u must be 2D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1D");
    TORCH_CHECK(x.sizes() == u.sizes(), "x and u must share the same shape");

    auto x_conti = x.contiguous();
    auto u_conti = u.contiguous();
    auto gamma_conti = gamma.contiguous();
    auto beta_conti = beta.contiguous();

    at::Tensor y = at::empty_like(x_conti);

    ACLNN_CMD(aclnnLnMul, x_conti, u_conti, gamma_conti, beta_conti, y);
    return y;
}

} // anonymous namespace

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("ln_mul(Tensor x, Tensor u, Tensor gamma, Tensor beta) -> Tensor");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("ln_mul", &ln_mul_impl_npu);
}

namespace {

at::Tensor ln_mul_meta(const at::Tensor &x, const at::Tensor &u,
                       const at::Tensor &gamma, const at::Tensor &beta)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(u.dim() == 2, "u must be 2D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1D");
    TORCH_CHECK(x.sizes() == u.sizes(), "x and u must share the same shape");
    return at::empty_like(x, x.options());
}

} // anonymous namespace

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("ln_mul", &ln_mul_meta);
}
