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
"""norm_multiply_dropout 算子的 autograd 注册（inductor 兼容）。

前向算子 ops_rec::norm_multiply_dropout 返回 (output, mean, var)。
反向算子 ops_rec::norm_multiply_dropout_backward 接收上游梯度与前向保存的
中间结果，输出 (d_u, d_x, d_weight, d_bias)。

通过 torch.library.register_autograd 注册，使前向算子支持 loss.backward()
并在 torch.compile (inductor) 图模式下正确工作。
"""

import torch

# 前向算子全名：ops_rec::norm_multiply_dropout
# 反向算子全名：ops_rec::norm_multiply_dropout_backward
_FORWARD_OP = "ops_rec::norm_multiply_dropout"

# dropout 概率小于该阈值时不执行 dropout
_NO_DROP_THRESHOLD = 1e-10

# mask 为 uint8 位图，1 字节对应 8 个元素
_ELEMS_PER_BYTE = 8


def _gen_mask(x: torch.Tensor, dropout_ratio: float) -> torch.Tensor:
    """生成 dropout 掩码（uint8 位图，bit 为随机 0/1）。

    dropout_ratio <= 1e-10 时不 dropout，返回 1 行占位 mask（kernel 侧不消费）。
    """
    if dropout_ratio <= _NO_DROP_THRESHOLD:
        return torch.zeros(1, dtype=torch.uint8, device=x.device)
    col = x.shape[-1]
    mask = torch.randint(0, 256, (x.shape[0], col // _ELEMS_PER_BYTE), dtype=torch.uint8, device=x.device)
    return mask


def norm_multiply_dropout_forward(x, u, weight, bias, eps=1e-5, dropout_ratio=0.0):
    """norm_multiply_dropout 前向接口（自动生成 mask 并支持自动求导）。

    Args:
        x: 输入张量 [B, C]
        u: 乘法因子 [B, C]
        weight: LayerNorm 权重 [C]
        bias: LayerNorm 偏置 [C]
        eps: 防除 0 极小值，取值范围 [1e-10, 1e-4]
        dropout_ratio: dropout 概率，取值范围 [0.0, 1.0]

    Returns:
        (output, mean, var)
    """
    mask = _gen_mask(x, dropout_ratio)
    return torch.ops.ops_rec.norm_multiply_dropout(x, u, weight, bias, mask, eps, dropout_ratio)


def _setup_context(ctx, inputs, output):
    """前向后保存反向所需张量。

    Args:
        ctx: autograd context
        inputs: 前向输入 (x, u, weight, bias, mask, eps, dropout_ratio)
        output: 前向输出 (output, mean, var)
    """
    x, u, weight, bias, mask, eps, dropout_ratio = inputs
    # output[1]/output[2] 为 mean/var，反向计算必需
    ctx.save_for_backward(x, u, weight, bias, mask, output[1], output[2])
    ctx.eps = eps
    ctx.dropout_ratio = dropout_ratio


def _backward(ctx, out_grad, mean_grad, var_grad):
    """反向传播。

    Args:
        ctx: autograd context
        out_grad: 前向 output 的上游梯度
        mean_grad/var_grad: mean/var 仅为反向中间结果，不参与梯度回传（忽略）

    Returns:
        与前向输入 (x, u, weight, bias, mask, eps, dropout_ratio) 一一对应的梯度
    """
    x, u, weight, bias, mask, mean, var = ctx.saved_tensors
    # 反向算子输出顺序：(d_u, d_x, d_weight, d_bias)
    d_u, d_x, d_weight, d_bias = torch.ops.ops_rec.norm_multiply_dropout_backward(
        out_grad, x, u, weight, bias, mean, var, mask, ctx.eps, ctx.dropout_ratio,
    )
    # 返回顺序对应前向输入：x, u, weight, bias, mask, eps, dropout_ratio
    return d_x, d_u, d_weight, d_bias, None, None, None


def register_autograd():
    """注册 norm_multiply_dropout 的 autograd 公式。

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
