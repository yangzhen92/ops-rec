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

#define OPS_REC_CUST_VENDORS "token_mixing"
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "common_utils.h"
#include "pytorch_npu_helper.hpp"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

constexpr int EXPECTED_DIM_3D = 3;
constexpr int EXPECTED_DIM_1D = 1;

at::Tensor token_mixing_impl_npu(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& beta,
                                 const double epsilon)
{
    check_tensor_non_empty(x, "x");
    check_tensor_dim(x, EXPECTED_DIM_3D, "x");
    check_tensor_non_empty(gamma, "gamma");
    check_tensor_dim(gamma, EXPECTED_DIM_1D, "gamma");
    check_tensor_non_empty(beta, "beta");
    check_tensor_dim(beta, EXPECTED_DIM_1D, "beta");

    std::vector<at::Tensor> tensors = {x, gamma, beta};
    std::vector<std::string> names = {"x", "gamma", "beta"};
    check_tensor_npu_device(tensors, names);

    // x: [B, S, H] -> x_t: [B, H, S]；kernel 侧对 (x + x_t) 做 LayerNorm
    at::Tensor x_t = x.permute({0, 2, 1});
    auto x_conti = x.contiguous();
    auto x_t_conti = x_t.contiguous();
    auto gamma_conti = gamma.contiguous();
    auto beta_conti = beta.contiguous();
    // 输出 shape [B, H, S]
    at::Tensor y = at::empty({x.size(0), x.size(2), x.size(1)}, x.options());

    EXEC_NPU_CMD(aclnnTokenMixing, x_conti, x_t_conti, gamma_conti, beta_conti, epsilon, y);

    return y;
}

class TokenMixing : public torch::autograd::Function<TokenMixing> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor x, at::Tensor gamma, at::Tensor beta,
                              const double epsilon)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->saved_data["epsilon"] = epsilon;
        auto y = token_mixing_impl_npu(x, gamma, beta, epsilon);
        ctx->save_for_backward({x, gamma, beta});
        return y;
    }
};

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("token_mixing(Tensor x, Tensor gamma, Tensor beta, float epsilon=1e-7) -> Tensor");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("token_mixing", &token_mixing_impl_npu);
}

namespace {

// 只推导 shape，不真正计算；inductor 构图走 Meta，执行仍派发 PrivateUse1
at::Tensor token_mixing_meta(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& beta,
                             const double /*epsilon*/)
{
    TORCH_CHECK(x.dim() == EXPECTED_DIM_3D, "x must be 3D");
    TORCH_CHECK(gamma.dim() == EXPECTED_DIM_1D, "gamma must be 1D");
    TORCH_CHECK(beta.dim() == EXPECTED_DIM_1D, "beta must be 1D");
    // 与 host InferShape 一致：y = [B, H, S]
    auto xSize = x.sym_sizes();
    return at::empty_symint({xSize[0], xSize[2], xSize[1]}, x.options());
}

}  // anonymous namespace

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("token_mixing", &token_mixing_meta);
}
