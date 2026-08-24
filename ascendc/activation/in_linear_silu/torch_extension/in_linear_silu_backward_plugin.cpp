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

#define OPS_REC_CUST_VENDORS "in_linear_silu_backward"
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"
#include "common_utils.h"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

const int MIN_DIM = 16;
const int MAX_DIM = 512 * 16; // max_dim * max_num_head
const int TENSOR_NUM = 4;
const int DIM_2 = 2;
const int DIM_1 = 1;

namespace {
void IsValidShape(const at::Tensor& x, const at::Tensor& weight,
                  const at::Tensor& bias, at::IntArrayRef split_list)
{
    check_tensor_non_empty(x, "x");
    check_tensor_non_empty(weight, "weight");

    check_tensor_dim(x, DIM_2, "x");
    check_tensor_dim(weight, DIM_2, "weight");

    TORCH_CHECK(split_list.size() == TENSOR_NUM, "split_list must have 4 elements.");
    int32_t totalDim = 0;
    for (int i = 0; i < TENSOR_NUM; i++) {
        TORCH_CHECK(split_list[i] >= MIN_DIM && split_list[i] <= MAX_DIM && split_list[i] % MIN_DIM == 0,
            "uvqk dim must in range[16, 8192] and multiples of 16.");
        totalDim += split_list[i];
    }

    auto k = x.size(1);
    auto n = weight.size(0);
    TORCH_CHECK(k >= MIN_DIM && k <= MAX_DIM && k % MIN_DIM == 0,
        "x dim[1] must in range[16, 8192] and multiples of 16.");
    TORCH_CHECK(n >= MIN_DIM * TENSOR_NUM && n <= MAX_DIM * TENSOR_NUM && n % MIN_DIM == 0,
        "weight dim[0] must in range[64, 32768] and multiples of 16.");
    TORCH_CHECK(totalDim == n, "weight_dim[0] must equal to sum(split_list).");
    TORCH_CHECK(weight.size(1) == k, "weight dim[1] must equal to x dim[1].");
    TORCH_CHECK((n % (4 * k) == 0), "weight dim[0] must be a multiple of 4 x dim[1]");

    if (bias.defined()) {
        check_tensor_non_empty(bias, "bias");
        check_tensor_dim(bias, DIM_1, "bias");
        TORCH_CHECK(bias.size(0) == n, "bias dim[0] must equal to weight dim[0].");
    }
}

void IsValidShapeBackward(const at::Tensor& x, const at::Tensor& weight,
                          const at::Tensor& user_grad, const at::Tensor& value_grad,
                          const at::Tensor& query_grad, const at::Tensor& key_grad,
                          const at::Tensor& linear_output, at::IntArrayRef split_list)
{
    check_tensor_non_empty(user_grad, "user_grad");
    check_tensor_non_empty(value_grad, "value_grad");
    check_tensor_non_empty(query_grad, "query_grad");
    check_tensor_non_empty(key_grad, "key_grad");
    check_tensor_non_empty(linear_output, "linear_output");

    check_tensor_dim(user_grad, DIM_2, "user_grad");
    check_tensor_dim(value_grad, DIM_2, "value_grad");
    check_tensor_dim(query_grad, DIM_2, "query_grad");
    check_tensor_dim(key_grad, DIM_2, "key_grad");
    check_tensor_dim(linear_output, DIM_2, "linear_output");

    auto n = weight.size(0);
    TORCH_CHECK(user_grad.size(1) == split_list[0] && value_grad.size(1) == split_list[1] &&
                query_grad.size(1) == split_list[2] && key_grad.size(1) == split_list[3] &&
                "user_grad, value_grad, query_grad, key_grad dim[1] must be equal to split_list.");
    TORCH_CHECK(user_grad.size(0) == x.size(0) && value_grad.size(0) == x.size(0) &&
                query_grad.size(0) == x.size(0) && key_grad.size(0) == x.size(0) &&
                linear_output.size(0) == x.size(0),
                "user_grad, value_grad, query_grad, key_grad linear_output dim[0] must equal to x dim[0].");
    TORCH_CHECK(linear_output.size(1) == n, "linear_output dim[1] must equal to weight dim[0].");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> InLinearSiluBackwardMeta(
    const at::Tensor& x,
    const at::Tensor& weight, const at::optional<Tensor>& bias, const at::Tensor& user_grad,
    const at::Tensor& value_grad, const at::Tensor& query_grad,
    const at::Tensor& key_grad, const at::Tensor& linear_output, at::IntArrayRef attr_dict)
{
    auto x_grad = at::empty_like(x, x.options());
    auto weight_grad = at::empty_like(weight, weight.options());
    auto bias_grad = at::Tensor();
    if (bias.has_value()) {
        bias_grad = at::empty_like(bias.value(), bias.value().options());
    }
    return std::tuple(x_grad, weight_grad, bias_grad, at::Tensor());
}
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> RunInLinearSiluBackward(
    const at::Tensor& x,
    const at::Tensor& weight, const at::optional<Tensor>& bias, const at::Tensor& user_grad,
    const at::Tensor& value_grad, const at::Tensor& query_grad,
    const at::Tensor& key_grad, const at::Tensor& linear_output, at::IntArrayRef attr_dict)
{
    auto xConti = x.contiguous();
    auto weightConti = weight.contiguous();
    auto biasConti = bias.has_value() ? bias.value() : at::Tensor();
    biasConti = biasConti.contiguous();
    IsValidShape(xConti, weightConti, biasConti, attr_dict);

    auto user_gradConti = user_grad.contiguous();
    auto value_gradConti = value_grad.contiguous();
    auto query_gradConti = query_grad.contiguous();
    auto key_gradConti = key_grad.contiguous();
    auto linear_outputConti = linear_output.contiguous();
    IsValidShapeBackward(xConti, weightConti, user_gradConti, value_gradConti,
                         query_gradConti, key_gradConti, linear_outputConti, attr_dict);

    auto x_grad = at::empty_like(xConti, at::kFloat);
    auto weight_grad = at::zeros_like(weightConti, at::kFloat);
    auto bias_grad = at::Tensor();
    bool isVardim = false;
    
    if (attr_dict[0] != attr_dict[1] || attr_dict[0] != attr_dict[2] || attr_dict[0] != attr_dict[3]) {
        isVardim = true;
        user_gradConti = at::cat({user_gradConti, value_gradConti, query_gradConti, key_gradConti}, -1);
    }

    if (bias.has_value()) {
        bias_grad = at::zeros_like(biasConti, biasConti.options());
    }
    bool isTrans = false;
    EXEC_NPU_CMD(aclnnInLinearSiluBackward, xConti, weightConti, biasConti,
                 user_gradConti, value_gradConti, query_gradConti, key_gradConti,
                 linear_outputConti, attr_dict, isTrans, isVardim, x_grad, weight_grad, bias_grad);
    return std::tuple(x_grad.to(x.options()), weight_grad.to(weight.options()), bias_grad, at::Tensor());
}


TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("in_linear_silu_backward("
        "Tensor x, Tensor weight, Tensor ? bias,"
        "Tensor user_grad, Tensor value_grad, Tensor query_grad, Tensor key_grad,"
        "Tensor linear_output,"
        "int[] attr_dict) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("in_linear_silu_backward", TORCH_FN(RunInLinearSiluBackward));
}

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("in_linear_silu_backward", TORCH_FN(InLinearSiluBackwardMeta));
}
