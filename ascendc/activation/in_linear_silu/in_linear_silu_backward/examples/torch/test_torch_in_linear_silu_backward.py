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

"""
in_linear_silu_backward 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_in_linear_silu_backward.py
"""

import torch
import torch.nn.functional as F

import ops_rec

torch.npu.config.allow_internal_format = False


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def silu_grad(linear_output):
    sig = sigmoid(linear_output.to(torch.float32))
    return sig + linear_output.to(torch.float32) * sig * (1.0 -sig)


def run_manual(seq, h, embed, split_list, data_type):
    torch.manual_seed(0)
    x = torch.randn(seq, embed, dtype=data_type).uniform_(-1, 1)
    weight = torch.randn(h, embed, dtype=data_type).uniform_(-1, 1)
    bias = torch.randn(h, dtype=torch.float32).uniform_(-1, 1)
    u = torch.randn(seq, split_list[0], dtype=data_type)
    v = torch.randn(seq, split_list[1], dtype=data_type)
    q = torch.randn(seq, split_list[2], dtype=data_type)
    k = torch.randn(seq, split_list[3], dtype=data_type)
    linear_output = torch.randn(seq, h, dtype=data_type).uniform_(-1, 1)

    mixed = torch.cat((u, v, q, k), dim = -1)
    lo_grad = mixed.to(torch.float32) * silu_grad(linear_output)
    x_grad = lo_grad.to(data_type) @ weight.to(data_type)
    w_grad = lo_grad.t().to(data_type) @ x.to(data_type)
    b_grad = lo_grad.sum(dim=0)

    return x, weight, bias, u, v, q, k, linear_output, (x_grad, w_grad, b_grad)


def test_in_linear_silu_backward():
    """
    测试in_linear_silu反向传播接口
    """
    seq, h, embed = 4, 128, 32
    split = h // 4
    split_list = [split] * 4
    dtype = torch.float16
    x, weight, bias, u, v, q, k, linear_output, golden = run_manual(seq, h, embed, split_list, dtype)
    xg, wg, bg = golden

    res = ops_rec.ascendc.activation.in_linear_silu_backward(x.npu(), weight.npu(), bias.npu(),
        u.npu(), v.npu(), q.npu(), k.npu(), linear_output.npu(), split_list)

    x_grad, weights_grad, bias_grad, _ = res

    diff_x = (x_grad.cpu().float() - xg.float()).abs().max().item()
    diff_w = (weights_grad.cpu().float() - wg.float()).abs().max().item()
    diff_b = (bias_grad.cpu().float() - bg.float()).abs().max().item()

    print(f"[torch] x_grad max diff: {diff_x:.6f}")
    print(f"[torch] weights_grad max diff: {diff_w:.6f}")
    print(f"[torch] bias_grad max diff: {diff_b:.6f}")

    assert diff_x < 1e-2, f"x_grad diff {diff_x} exceeds 1e-2"
    assert diff_w < 1e-2, f"weights_grad diff {diff_w} exceeds 1e-2"
    assert diff_b < 1e-2, f"bias_grad diff {diff_b} exceeds 1e-2"
    print("[torch] in_linear_silu_backward torch example PASSED")


if __name__ == "__main__":
    test_in_linear_silu_backward()