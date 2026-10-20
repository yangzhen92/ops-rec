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
"""norm_multiply_dropout 反向 torch 调用示例（autograd 触发）。"""

import torch

import ops_rec


def main():
    batch, col = 4, 512
    dtype = torch.bfloat16
    device = "npu"

    x = (torch.rand([batch, col], device=device, dtype=dtype) - 0.5).requires_grad_(True)
    u = (torch.rand([batch, col], device=device, dtype=dtype) - 0.5).requires_grad_(True)
    weight = (torch.rand([col], device=device, dtype=dtype) - 0.5).requires_grad_(True)
    bias = (torch.rand([col], device=device, dtype=dtype) - 0.5).requires_grad_(True)

    # 前向 + loss.backward() 自动触发 norm_multiply_dropout_backward
    output, _, _ = ops_rec.ascendc.activation.norm_multiply_dropout(
        x, u, weight, bias, eps=1e-5, dropout_ratio=0.0)
    loss = output.float().sum()
    loss.backward()

    print("d_x:", x.grad.shape, x.grad.dtype)
    print("d_u:", u.grad.shape, u.grad.dtype)
    print("d_weight:", weight.grad.shape, weight.grad.dtype)
    print("d_bias:", bias.grad.shape, bias.grad.dtype)


if __name__ == "__main__":
    main()
