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
#define OPS_REC_CUST_VENDORS "norm_multiply_dropout_backward"
#include <torch/torch.h>

#include <stdexcept>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"
#include "common_utils.h"
using namespace at;

void CheckNormMultiplyDropoutBackwardInput(const at::Tensor& dOut, const at::Tensor& x, const at::Tensor& u,
                                           const at::Tensor& weight, const at::Tensor& bias,
                                           const at::Tensor& mean, const at::Tensor& var,
                                           const at::Tensor& mask)
{
    check_tensor_non_empty(dOut, "d_out");
    check_tensor_non_empty(x, "x");
    check_tensor_non_empty(u, "u");
    check_tensor_non_empty(weight, "weight");
    check_tensor_non_empty(bias, "bias");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {dOut, x, u, weight, bias, mean, var, mask};
    std::vector<std::string> names = {"d_out", "x", "u", "weight", "bias", "mean", "var", "mask"};
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(dOut.dim() == 2, "The d_out should be 2D");
    TORCH_CHECK(x.dim() == 2, "The x should be 2D");
    TORCH_CHECK(u.dim() == 2, "The u should be 2D");
    TORCH_CHECK(weight.dim() == 1, "The weight should be 1D");
    TORCH_CHECK(bias.dim() == 1, "The bias should be 1D");
    TORCH_CHECK(mean.dim() == 1, "The mean should be 1D");
    TORCH_CHECK(var.dim() == 1, "The var should be 1D");
    TORCH_CHECK(x.scalar_type() == dOut.scalar_type(), "The dtype of x should be same as d_out");
    TORCH_CHECK(u.scalar_type() == dOut.scalar_type(), "The dtype of u should be same as d_out");
    TORCH_CHECK(weight.scalar_type() == dOut.scalar_type(), "The dtype of weight should be same as d_out");
    TORCH_CHECK(bias.scalar_type() == dOut.scalar_type(), "The dtype of bias should be same as d_out");
    TORCH_CHECK(mean.scalar_type() == at::kFloat, "The dtype of mean should be float32");
    TORCH_CHECK(var.scalar_type() == at::kFloat, "The dtype of var should be float32");
    TORCH_CHECK(mask.scalar_type() == at::kChar, "The dtype of mask should be uint8");
    TORCH_CHECK(dOut.sizes() == x.sizes(), "d_out and x must have identical shape");
    TORCH_CHECK(dOut.sizes() == u.sizes(), "d_out and u must have identical shape");
    TORCH_CHECK(mean.size(0) == x.size(0), "mean dim[0] must equal to x dim[0]");
    TORCH_CHECK(var.size(0) == x.size(0), "var dim[0] must equal to x dim[0]");
    TORCH_CHECK(weight.size(0) == x.size(1), "weight dim[0] must equal to x dim[1]");
    TORCH_CHECK(bias.size(0) == x.size(1), "bias dim[0] must equal to x dim[1]");
}

// 反向 NPU 实现，返回 (d_u, d_x, d_weight, d_bias)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> NormMultiplyDropoutBackwardInter(
    const at::Tensor& dOut, const at::Tensor& x, const at::Tensor& u, const at::Tensor& weight,
    const at::Tensor& bias, const at::Tensor& mean, const at::Tensor& var, const at::Tensor& mask, double eps,
    double dropoutRatio)
{
    CheckNormMultiplyDropoutBackwardInput(dOut, x, u, weight, bias, mean, var, mask);
    TORCH_CHECK(eps > 0, "eps must be greater than 0");
    TORCH_CHECK(dropoutRatio >= 0 && dropoutRatio <= 1, "dropout_ratio must be in range [0.0, 1.0]");

    auto dOutCont = dOut.contiguous();
    auto xCont = x.contiguous();
    auto uCont = u.contiguous();
    auto weightCont = weight.contiguous();
    auto biasCont = bias.contiguous();
    auto meanCont = mean.contiguous();
    auto varCont = var.contiguous();
    auto maskCont = mask.contiguous();

    at::TensorOptions options = xCont.options();
    auto dU = at::empty(xCont.sizes(), options);
    auto dX = at::empty(xCont.sizes(), options);
    auto dWeight = at::empty({xCont.size(1)}, options.dtype(at::kFloat));
    auto dBias = at::empty({xCont.size(1)}, options.dtype(at::kFloat));

    // aclnn 生成的接口 Float 属性为 double，直接透传
    EXEC_NPU_CMD(aclnnNormMultiplyDropoutBackward, dOutCont, xCont, uCont, weightCont, biasCont, meanCont, varCont,
                 maskCont, eps, dropoutRatio, dU, dX, dWeight, dBias);
    return {dU, dX, dWeight, dBias};
}

namespace {
// Meta 实现：供 torch.compile (inductor) trace 阶段做 shape/dtype 推断
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> NormMultiplyDropoutBackwardInterMeta(
    const at::Tensor& dOut, const at::Tensor& x, const at::Tensor&, const at::Tensor& weight,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, double, double)
{
    at::TensorOptions options = x.options();
    auto dU = at::empty(dOut.sizes(), options);
    auto dX = at::empty(x.sizes(), options);
    auto dWeight = at::empty({x.size(1)}, options.dtype(at::kFloat));
    auto dBias = at::empty({x.size(1)}, options.dtype(at::kFloat));
    return {dU, dX, dWeight, dBias};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("norm_multiply_dropout_backward(Tensor d_out, Tensor x, Tensor u, Tensor weight, Tensor bias, "
          "Tensor mean, Tensor var, Tensor mask, float eps=1e-05, float dropout_ratio=0.0) "
          "-> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("norm_multiply_dropout_backward", TORCH_FN(NormMultiplyDropoutBackwardInter));
}

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("norm_multiply_dropout_backward", &NormMultiplyDropoutBackwardInterMeta);
}
