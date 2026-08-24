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
in_linear_silu 算子 PyTorch 接口使用样例

功能：计算 mixed = x @ weight^T + bias，对 mixed 做 Silu 激活后按
split_arg_list 拆分为 user、value、query、key，并额外输出 linear_output。

用法:
    python3 test_torch_in_linear_silu.py
"""

import torch
import ops_rec

torch.npu.config.allow_internal_format = False

def get_user_value_query_key_tensors_uniattn(x, weight, bias, split_arg_list):
    mixed_uvqk = torch.addmm(bias, x, weight.t())
    linear_output = mixed_uvqk
    mixed_uvqk = torch.nn.functional.silu(mixed_uvqk)
    (user, value, query, key) = torch.split(
        mixed_uvqk,
        split_arg_list,
        dim=-1,
    )
    value = value.view(-1, split_arg_list[1]).contiguous()
    query = query.view(-1, split_arg_list[2]).contiguous()
    key = key.view(-1, split_arg_list[3]).contiguous()
    user = user.contiguous()
    return user, value, query, key, linear_output


def get_user_value_query_key_tensors_fused(x, weight, bias, split_arg_list):
    result = ops_rec.ascendc.activation.distance_in_linear_silu(
        x,
        weight,
        bias.to(torch.float32).contiguous(),
        split_arg_list
    )
    return result


def compare_result(result, golden, dtype):
    atol = 1e-4
    rtol = 1e-4
    if dtype == torch.float16:
        atol = 1e-3
        rtol = 1e-3
    elif dtype == torch.bfloat16:
        atol = 5e-3
        rtol = 5e-3
    for res, gold in zip(result, golden):
        assert torch.allclose(res, gold, atol=atol, rtol=rtol)


def test_in_linear_silu():
    """验证 distance_in_linear_silu 算子（仅前向）的推理结果。"""
    m, n, k = 256, 1024, 16
    device = "npu:0"
    dtype = torch.float16
    torch.npu.set_device(device)
    seq_len = int(n / 4)
    split_arg_list = [seq_len, seq_len, seq_len, seq_len]
    x = torch.randint(low=-100, high=101, size=(m, k), dtype=dtype, device=device)
    w = torch.randint(low=-100, high=101, size=(n, k), dtype=dtype, device=device)
    w = torch.nn.init.xavier_uniform_(w)
    bias = torch.ones(n, dtype=dtype, device=device)

    golden = get_user_value_query_key_tensors_uniattn(x, w, bias, split_arg_list)
    result = get_user_value_query_key_tensors_fused(x, w, bias, split_arg_list)
    compare_result(golden, result, dtype)
    print("[torch] in_linear_silu torch example PASSED")


if __name__ == "__main__":
    test_in_linear_silu()
