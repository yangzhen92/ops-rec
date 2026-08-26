#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""in_linear_silu 算子的 autograd 注册（inductor 兼容）。

前向算子 distance_in_linear_silu 返回 (user, value, query, key, linear_output)。
反向算子 in_linear_silu_backward 接收 user_grad/value_grad/query_grad/key_grad
以及前向保存的 linear_output，内部已完成 SiLU 反向 + 线性层反向，输出
(x_grad, weight_grad, bias_grad, _)。

通过 torch.library.register_autograd 注册，使前向算子支持 loss.backward()
并在 torch.compile (inductor) 图模式下正确工作。
"""

import torch

# 前向算子全名：ops_rec::distance_in_linear_silu
# 反向算子全名：ops_rec::in_linear_silu_backward
_OP_NAMESPACE = "ops_rec"
_FORWARD_OP = f"{_OP_NAMESPACE}::distance_in_linear_silu"
_BACKWARD_OP = f"{_OP_NAMESPACE}::in_linear_silu_backward"


def _setup_context(ctx, inputs, output):
    """前向后保存反向所需张量。

    Args:
        ctx: autograd context
        inputs: 前向输入 (x, weight, bias, attr_dict)
        output: 前向输出 (user, value, query, key, linear_output)
    """
    x, weight, bias, attr_dict = inputs
    # output[-1] 是 linear_output，反向算子需要它来计算 SiLU 的梯度
    linear_output = output[-1]
    # bias 可能为 None（无偏置场景），save_for_backward 只能保存张量
    ctx.has_bias = bias is not None
    if ctx.has_bias:
        ctx.save_for_backward(x, weight, bias, linear_output)
    else:
        ctx.save_for_backward(x, weight, linear_output)
    ctx.attr_dict = list(attr_dict)


def _backward(ctx, user_grad, value_grad, query_grad, key_grad, linear_output_grad):
    """反向传播。

    Args:
        ctx: autograd context
        user_grad, value_grad, query_grad, key_grad: 前向四个输出的梯度
        linear_output_grad: linear_output 的梯度（忽略，反向算子内部已处理 SiLU 反向）

    Returns:
        与前向输入 (x, weight, bias, attr_dict) 一一对应的梯度
    """
    if ctx.has_bias:
        x, weight, bias, linear_output = ctx.saved_tensors
    else:
        x, weight, linear_output = ctx.saved_tensors
        bias = None
    # 反向算子内部已完成 SiLU 反向 + 线性层反向
    x_grad, weight_grad, bias_grad, _ = torch.ops.ops_rec.in_linear_silu_backward(
        x,
        weight,
        bias,
        user_grad,
        value_grad,
        query_grad,
        key_grad,
        linear_output,
        ctx.attr_dict,
    )
    # 返回顺序对应 forward 输入：x, weight, bias, attr_dict
    # attr_dict 是非张量输入，梯度为 None
    return x_grad, weight_grad, bias_grad, None


def register_autograd():
    """注册 distance_in_linear_silu 的 autograd 公式。

    在 ops_rec.autograd 插件包自动加载时被调用，也可手动调用。
    重复注册会抛异常，因此做一次幂等保护。
    """
    try:
        torch.library.register_autograd(_FORWARD_OP, _backward, setup_context=_setup_context)
    except RuntimeError as exc:
        if "already" in str(exc).lower() or "registered" in str(exc).lower():
            return
        raise


if __name__ == "__main__":
    register_autograd()
