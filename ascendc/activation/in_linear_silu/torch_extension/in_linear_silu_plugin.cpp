/**
 * @file in_linear_silu_plugin.cpp
 *
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#define OPS_REC_CUST_VENDORS "in_linear_silu_forward"
#include <torch/torch.h>

#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <stdexcept>
#include <algorithm>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"
#include "common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using namespace at;

const int MIN_DIM = 16;
const int MAX_DIM = 512 * 16;  // max_dim * max_num_head
const int TENSOR_NUM = 4;
const int DIM_2 = 2;
const int DIM_1 = 1;

void IsValidShape(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias, at::IntArrayRef splitList)
{
    check_tensor_non_empty(x, "x");
    check_tensor_non_empty(weight, "weight");

    check_tensor_dim(x, DIM_2, "x");
    check_tensor_dim(weight, DIM_2, "weight");

    TORCH_CHECK(splitList.size() == TENSOR_NUM, "splitList must have 4 elements.");
    int32_t totalDim = 0;
    for (int i = 0; i < TENSOR_NUM; i++) {
        TORCH_CHECK(splitList[i] >= MIN_DIM && splitList[i] <= MAX_DIM && splitList[i] % MIN_DIM == 0,
                    "uvqk dim must in range[16, 8192] and multiples of 16.");
        totalDim += splitList[i];
    }

    auto k = x.size(1);
    auto n = weight.size(0);
    TORCH_CHECK(k >= MIN_DIM && k <= MAX_DIM && k % MIN_DIM == 0,
                "x dim[1] must in range[16, 8192] and multiples of 16.");
    TORCH_CHECK(n >= MIN_DIM * TENSOR_NUM && n <= MAX_DIM * TENSOR_NUM && n % MIN_DIM == 0,
                "weight dim[0] must in range[64, 32768] and multiples of 16.");
    TORCH_CHECK(totalDim == n, "weight_dim[0] must equal to sum(splitList).");
    TORCH_CHECK(weight.size(1) == k, "weight dim[1] must equal to x dim[1].");
    TORCH_CHECK((n % (4 * k) == 0), "weight dim[0] must be a multiple of 4 x dim[1]");

    if (bias.defined()) {
        check_tensor_non_empty(bias, "bias");
        check_tensor_dim(bias, DIM_1, "bias");
        TORCH_CHECK(bias.size(0) == n, "bias dim[0] must equal to weight dim[0].");
    }
}

void IsValidShapeBackward(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& user_grad,
                          const at::Tensor& value_grad, const at::Tensor& query_grad, const at::Tensor& key_grad,
                          const at::Tensor& linear_output, at::IntArrayRef splitList)
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
    TORCH_CHECK(user_grad.size(1) == splitList[0] && value_grad.size(1) == splitList[1] &&
                query_grad.size(1) == splitList[2] && key_grad.size(1) == splitList[3] &&
                "user_grad, value_grad, query_grad, key_grad dim[1] must be equal to splitList.");
    TORCH_CHECK(user_grad.size(0) == x.size(0) && value_grad.size(0) == x.size(0) && query_grad.size(0) == x.size(0) &&
                    key_grad.size(0) == x.size(0) && linear_output.size(0) == x.size(0),
                "user_grad, value_grad, query_grad, key_grad linear_output dim[0] must equal to x dim[0].");
    TORCH_CHECK(linear_output.size(1) == n, "linear_output dim[1] must equal to weight dim[0].");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> RunInLinearSiluForwardInter(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias, at::IntArrayRef splitList)
{
    IsValidShape(x, weight, bias, splitList);

    at::TensorOptions options = x.options();
    auto m = x.size(0);
    auto n = weight.size(0);
    auto userOut = at::empty({m, splitList[0]}, options);
    auto valueOut = at::empty({m, splitList[1]}, options);
    auto queryOut = at::empty({m, splitList[2]}, options);
    auto keyOut = at::empty({m, splitList[3]}, options);
    auto linearOutputOut = at::empty({m, n}, options);

    auto xCont = x.contiguous();
    auto weightCont = weight.contiguous();
    auto biasCont = bias.contiguous();

    EXEC_NPU_CMD(aclnnInLinearSilu, xCont, weightCont, biasCont, splitList, userOut, valueOut, queryOut, keyOut,
                 linearOutputOut);
    return {userOut, valueOut, queryOut, keyOut, linearOutputOut};
}

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> RunInLinearSiluForwardInterMeta(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias, at::IntArrayRef splitList)
{
    at::TensorOptions options = x.options();
    auto m = x.size(0);
    auto n = weight.size(0);
    auto userOut = at::empty({m, splitList[0]}, options);
    auto valueOut = at::empty({m, splitList[1]}, options);
    auto queryOut = at::empty({m, splitList[2]}, options);
    auto keyOut = at::empty({m, splitList[3]}, options);
    auto linearOutputOut = at::empty({m, n}, options);
    return {userOut, valueOut, queryOut, keyOut, linearOutputOut};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("distance_in_linear_silu(Tensor x, Tensor weight, Tensor bias, int[] attr_dict) -> "
          "(Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("distance_in_linear_silu", TORCH_FN(RunInLinearSiluForwardInter));
}

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("distance_in_linear_silu", &RunInLinearSiluForwardInterMeta);
}