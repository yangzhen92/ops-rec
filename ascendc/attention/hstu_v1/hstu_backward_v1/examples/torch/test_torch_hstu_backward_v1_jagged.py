#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
hstu_jagged_backward 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_hstu_backward_v1_jagged.py
"""

import torch
import ops_rec
import torch_npu
import random
import numpy as np

# pylint: disable=duplicate-code
torch.npu.config.allow_internal_format = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN自动优化


def _describe(name, tensor):
    """打印张量的形状、dtype 与数值统计（从 NPU 取回 CPU 展示）。"""
    if tensor is None:
        print(f"  {name}: None")
        return
    t = tensor.cpu()
    desc = (
        f"shape={list(t.shape)}, dtype={t.dtype}, min={t.min().item():.4f}, "
        f"max={t.max().item():.4f}, mean={t.float().mean().item():.4f}"
    )
    print(f"  {name}: {desc}")


def _generate_jagged_data(batch_size, max_seq_len, head_num, head_dim, dtype, num_context=None, num_target=None):
    """生成变长序列数据，返回 grad/q/k/v 及相关的 offset 信息。"""
    min_seq_len = 1
    if num_context is not None:
        min_seq_len += num_context
    if num_target is not None:
        min_seq_len += num_target

    seq_lens_q = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32)
    seq_lens_k = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32)
    seq_lens_q = torch.where(seq_lens_k < seq_lens_q, seq_lens_k, seq_lens_q)

    seq_offset_q = torch.concat((torch.zeros((1,), dtype=torch.int32), torch.cumsum(seq_lens_q, axis=0)))
    seq_offset_k = torch.concat((torch.zeros((1,), dtype=torch.int32), torch.cumsum(seq_lens_k, axis=0)))

    total_len_q = torch.sum(seq_lens_q).item()
    total_len_k = torch.sum(seq_lens_k).item()

    grad = torch.rand(total_len_q, head_num, head_dim, dtype=dtype).uniform_(-1, 1).npu()
    q = torch.rand(total_len_q, head_num, head_dim, dtype=dtype).uniform_(-1, 1).npu()
    k = torch.rand(total_len_k, head_num, head_dim, dtype=dtype).uniform_(-1, 1).npu()
    v = torch.rand(total_len_k, head_num, head_dim, dtype=dtype).uniform_(-1, 1).npu()

    return grad, q, k, v, seq_offset_q.npu(), seq_offset_k.npu()


def test_hstu_jagged_backward(
    label, mask_type, enable_bias, silu_scale=0.0, num_context=None, num_target=None, target_group_size=None
):
    batch_size, max_seq_len, head_num, head_dim = 2, 64, 4, 32
    dtype = torch.float32
    alpha = 1.0

    # 根据 mask_type 构造掩码
    if mask_type == 0:
        mask = torch.tril(torch.ones(batch_size, max_seq_len, max_seq_len, dtype=dtype)).npu()
    elif mask_type == 3:
        mask = torch.empty(batch_size, head_num, max_seq_len, max_seq_len, dtype=dtype).uniform_(-1, 1).npu()
    else:  # mask_type == 2
        mask = None

    bias = None
    if enable_bias:
        bias = torch.empty(batch_size, head_num, max_seq_len, max_seq_len, dtype=dtype).uniform_(-1, 1).npu()

    grad, q, k, v, seq_offset_q, seq_offset_k = _generate_jagged_data(
        batch_size, max_seq_len, head_num, head_dim, dtype, num_context, num_target
    )

    q_grad, k_grad, v_grad, attn_bias_grad = ops_rec.ascendc.attention.hstu_jagged_backward(
        grad,
        q,
        k,
        v,
        mask,
        bias,
        mask_type,
        max_seq_len,
        max_seq_len,
        silu_scale,
        seq_offset_q,
        seq_offset_k,
        num_context,
        num_target,
        target_group_size,
        alpha,
    )

    torch.npu.synchronize()
    print(f"[{label}] hstu_jagged_backward (mask_type={mask_type}, enable_bias={enable_bias}):")
    _describe("q_grad", q_grad)
    _describe("k_grad", k_grad)
    _describe("v_grad", v_grad)
    _describe("attn_bias_grad", attn_bias_grad)
    print()


def main():
    torch.npu.set_device(2)
    set_seed(1234)
    # 无掩码、无 bias
    test_hstu_jagged_backward("无掩码、无 bias", mask_type=2, enable_bias=False)
    # 无掩码 + bias
    test_hstu_jagged_backward("无掩码 + bias", mask_type=2, enable_bias=True)
    # 下三角掩码
    test_hstu_jagged_backward("下三角掩码", mask_type=0, enable_bias=False)
    # 自定义掩码 + bias
    test_hstu_jagged_backward("自定义掩码 + bias", mask_type=3, enable_bias=True)
    print("hstu_backward_v1 (jagged) 示例全部执行完成。")


if __name__ == "__main__":
    main()
