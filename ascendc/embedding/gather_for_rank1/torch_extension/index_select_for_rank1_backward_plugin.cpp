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

// 声明本插件依赖的 vendor（aclnn 接口所在 op_api 库），必须在 include "pytorch_npu_helper.hpp" 之前定义
#define OPS_REC_CUST_VENDORS "index_select_for_rank1_backward"

#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "common_utils.h"
#include "pytorch_npu_helper.hpp"

using tensor_list = std::vector<at::Tensor>;
using namespace at;

namespace npu_index_select_for_rank1_backward {

constexpr int64_t RANK1_DIM = 1;

std::tuple<at::Tensor, at::Tensor> index_select_for_rank1_backward_impl(const at::Tensor& grad_y,
                                                                        const at::Tensor& x,
                                                                        const at::Tensor& index)
{
    check_tensor_non_empty(grad_y, "grad_y");
    check_tensor_non_empty(x, "x");
    check_tensor_non_empty(index, "index");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {grad_y, x, index};
    std::vector<std::string> names = {"grad_y", "x", "index"};
    check_tensor_npu_device(tensors, names);

    // rank-1 语义：权重表/梯度/索引均为 1D
    check_tensor_dim(grad_y, RANK1_DIM, "grad_y");
    check_tensor_dim(x, RANK1_DIM, "x");
    check_tensor_dim(index, RANK1_DIM, "index");
    TORCH_CHECK(grad_y.numel() == index.numel(), "grad_y numel (", grad_y.numel(),
                ") must be equal to index numel (", index.numel(), ").");

    auto grad_y_conti = grad_y.contiguous();
    auto x_conti = x.contiguous();
    auto index_conti = index.contiguous();

    // kernel 通过 AtomicAdd 向 grad_x 逐核累加，grad_x 必须以全零初始化
    at::Tensor grad_x = at::zeros_like(x_conti);
    EXEC_NPU_CMD(aclnnIndexSelectForRank1Backward, grad_y_conti, x_conti, index_conti, grad_x);

    // 第二个返回值为 index 梯度的占位（index 不可导，恒为 undefined）
    return std::make_tuple(grad_x, at::Tensor());
}

}  // namespace npu_index_select_for_rank1_backward

// 注册 index_select_for_rank1_backward 的 schema。
// 返回 (grad_x, grad_index)：grad_index 恒为 undefined，保持与 RecSDK
// torch.ops.mxrec.index_select_for_rank1_backward 相同的解包语义，
// 前向算子 gather_for_rank1 的 autograd 绑定以此返回值建立反向。
TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("index_select_for_rank1_backward(Tensor grad_y, Tensor x, Tensor index) -> (Tensor, Tensor)");
}

// NPU 设备实现（PrivateUse1 = Ascend NPU 后端标识）
TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("index_select_for_rank1_backward",
            TORCH_FN(npu_index_select_for_rank1_backward::index_select_for_rank1_backward_impl));
}

namespace {

std::tuple<at::Tensor, at::Tensor> index_select_for_rank1_backward_meta(const at::Tensor& grad_y,
                                                                        const at::Tensor& x,
                                                                        const at::Tensor& index)
{
    check_tensor_dim(grad_y, npu_index_select_for_rank1_backward::RANK1_DIM, "grad_y");
    check_tensor_dim(x, npu_index_select_for_rank1_backward::RANK1_DIM, "x");
    check_tensor_dim(index, npu_index_select_for_rank1_backward::RANK1_DIM, "index");
    TORCH_CHECK(grad_y.numel() == index.numel(), "grad_y numel (", grad_y.numel(),
                ") must be equal to index numel (", index.numel(), ").");
    return std::make_tuple(at::empty_like(x, x.options()), at::Tensor());
}

}  // anonymous namespace

// Meta 实现，用于 torch.compile 的 shape 推断
TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("index_select_for_rank1_backward", TORCH_FN(index_select_for_rank1_backward_meta));
}
