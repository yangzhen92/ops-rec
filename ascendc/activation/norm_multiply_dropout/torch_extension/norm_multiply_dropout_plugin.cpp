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
#define OPS_REC_CUST_VENDORS "norm_multiply_dropout_forward"
#include <torch/torch.h>

#include <stdexcept>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"
#include "common_utils.h"
using namespace at;

// dropout 概率小于该阈值时不执行 dropout，mask 无效
constexpr float NO_DROP_THRESHOLD = 1e-10;

void CheckNormMultiplyDropoutInput(const at::Tensor& x, const at::Tensor& u, const at::Tensor& weight,
                                   const at::Tensor& bias, const at::Tensor& mask)
{
    check_tensor_non_empty(x, "x");
    check_tensor_non_empty(u, "u");
    check_tensor_non_empty(weight, "weight");
    check_tensor_non_empty(bias, "bias");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {x, u, weight, bias, mask};
    std::vector<std::string> names = {"x", "u", "weight", "bias", "mask"};
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(x.dim() == 2, "The x should be 2D");
    TORCH_CHECK(u.dim() == 2, "The u should be 2D");
    TORCH_CHECK(weight.dim() == 1, "The weight should be 1D");
    TORCH_CHECK(bias.dim() == 1, "The bias should be 1D");
    TORCH_CHECK(u.scalar_type() == x.scalar_type(), "The dtype of u should be same as x");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "The dtype of weight should be same as x");
    TORCH_CHECK(bias.scalar_type() == x.scalar_type(), "The dtype of bias should be same as x");
    TORCH_CHECK(mask.scalar_type() == at::kChar, "The dtype of mask should be uint8");
    TORCH_CHECK(x.sizes() == u.sizes(), "x and u must have identical shape");
    TORCH_CHECK(weight.size(0) == x.size(1), "weight dim[0] must equal to x dim[1]");
    TORCH_CHECK(bias.size(0) == x.size(1), "bias dim[0] must equal to x dim[1]");
}

// 前向 NPU 实现，返回 (output, mean, var)
std::tuple<at::Tensor, at::Tensor, at::Tensor> NormMultiplyDropoutForwardInter(const at::Tensor& x,
                                                                               const at::Tensor& u,
                                                                               const at::Tensor& weight,
                                                                               const at::Tensor& bias,
                                                                               const at::Tensor& mask, double eps,
                                                                               double dropoutRatio)
{
    CheckNormMultiplyDropoutInput(x, u, weight, bias, mask);
    TORCH_CHECK(eps > 0, "eps must be greater than 0");
    TORCH_CHECK(dropoutRatio >= 0 && dropoutRatio <= 1, "dropout_ratio must be in range [0.0, 1.0]");

    auto xCont = x.contiguous();
    auto uCont = u.contiguous();
    auto weightCont = weight.contiguous();
    auto biasCont = bias.contiguous();
    auto maskCont = mask.contiguous();

    at::TensorOptions options = xCont.options();
    auto output = at::empty(xCont.sizes(), options);
    auto mean = at::empty({xCont.size(0)}, options.dtype(at::kFloat));
    auto var = at::empty({xCont.size(0)}, options.dtype(at::kFloat));

    // dropout_ratio 小于阈值时 mask 可传任意 uint8 空量，直接透传算子处理
    float epsVal = static_cast<float>(eps);
    float ratioVal = static_cast<float>(dropoutRatio);
    EXEC_NPU_CMD(aclnnNormMultiplyDropout, xCont, uCont, weightCont, biasCont, maskCont, epsVal, ratioVal,
                 output, mean, var);
    return {output, mean, var};
}

namespace {
// Meta 实现：供 torch.compile (inductor) trace 阶段做 shape/dtype 推断
std::tuple<at::Tensor, at::Tensor, at::Tensor> NormMultiplyDropoutForwardInterMeta(const at::Tensor& x,
                                                                                   const at::Tensor&,
                                                                                   const at::Tensor& weight,
                                                                                   const at::Tensor& bias,
                                                                                   const at::Tensor&, double, double)
{
    at::TensorOptions options = x.options();
    auto output = at::empty(x.sizes(), options);
    auto mean = at::empty({x.size(0)}, options.dtype(at::kFloat));
    auto var = at::empty({x.size(0)}, options.dtype(at::kFloat));
    return {output, mean, var};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("norm_multiply_dropout(Tensor x, Tensor u, Tensor weight, Tensor bias, Tensor mask, "
          "float eps=1e-05, float dropout_ratio=0.0) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("norm_multiply_dropout", TORCH_FN(NormMultiplyDropoutForwardInter));
}

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("norm_multiply_dropout", &NormMultiplyDropoutForwardInterMeta);
}
