# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# dense_jagged_cat_jagged_out_kernel_api.py
"""
dense_jagged_cat_jagged_out_kernel 算子的 ATK API 适配器（Autotune 版本）

语义（对齐 FBGEMM sll.dense_jagged_cat_jagged_out）：
  dense a[B] 与 jagged b[sum(seq_len)] 按 batch 拼接为 c[sum(seq_len)+B]，
  每个 batch 输出 [a[i], b[offsets[i]:offsets[i+1]]]，段长可为 0。
  grid = (B,)，BLOCK_SIZE 由 autotune 自动选择。

数值来源：generator 只保证 shape/dtype 自洽，本文件在运行期依据 shape 确定性地
推导 offsets / 每段长度 / max_seq_len（无 random，保证 GPU/NPU 一致）。
  - B         = a_ptr.shape[0]
  - total_seq = b_ptr.shape[0]
  - lengths   = 由 (B, total_seq) 确定性切分得到（散乱但可复现）
  - offsets   = cumsum(lengths)
"""

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

# 从 autotune 目录导入 autotune 版本的 kernel
from triton_dense_jagged_cat_jagged_out import dense_jagged_cat_jagged_out_kernel


def get_device():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


@register("dense_jagged_cat_jagged_out_kernel")
class DenseJaggedCatJaggedOutKernelAPI(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        device = get_device()

        a_ptr_input = input_data.kwargs.get("a_ptr", None)
        b_ptr_input = input_data.kwargs.get("b_ptr", None)
        b_offsets_ptr_input = input_data.kwargs.get("b_offsets_ptr", None)

        if a_ptr_input is None or b_ptr_input is None or b_offsets_ptr_input is None:
            raise ValueError("a_ptr, b_ptr or b_offsets_ptr not found")

        a_ptr_raw = self._to_tensor(a_ptr_input, device)
        b_ptr_raw = self._to_tensor(b_ptr_input, device)
        b_offsets_ptr_raw = self._to_tensor(b_offsets_ptr_input, device, torch.int32)

        # B 与 total_seq 直接取自框架采样得到的 shape（唯一随机源）
        B = a_ptr_raw.reshape(-1).shape[0]
        if B <= 0:
            # b_offsets shape 为 [B+1]，可作兜底
            B = max(1, b_offsets_ptr_raw.shape[0] - 1)
        total_seq = b_ptr_raw.reshape(-1).shape[0]

        dtype = a_ptr_raw.dtype

        # 依据 (B, total_seq) 确定性切分出每段长度（段长可为 0，对齐 FBGEMM）
        lengths_seq = self._split_lengths(total_seq, B)
        total_seq = sum(lengths_seq)
        max_seq_len = max(lengths_seq) if lengths_seq else 0

        # 由 lengths 现算 offsets
        b_offsets_ptr = self._cumsum_tensor(lengths_seq, device)
        a_ptr = self._rebuild_1d(a_ptr_raw, B, dtype, device)
        b_ptr = self._rebuild_1d(b_ptr_raw, total_seq, dtype, device)

        # 输出 c[total_seq + B] 与 c_offsets[B+1]
        c_ptr = torch.zeros((total_seq + B,), dtype=dtype, device=device)
        c_offsets_ptr = torch.empty((B + 1,), dtype=torch.int32, device=device)

        # pylint: disable=attribute-defined-outside-init
        self._device = device
        self._a_ptr = a_ptr
        self._b_ptr = b_ptr
        self._c_ptr = c_ptr
        self._b_offsets_ptr = b_offsets_ptr
        self._c_offsets_ptr = c_offsets_ptr
        self._max_seq_len = max_seq_len
        self._B = B
        # pylint: enable=attribute-defined-outside-init

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 注意：BLOCK_SIZE 是 autotune 参数，不再显式传递
        dense_jagged_cat_jagged_out_kernel[(self._B,)](
            self._a_ptr,
            self._b_ptr,
            self._c_ptr,
            self._b_offsets_ptr,
            self._c_offsets_ptr,
            self._max_seq_len,
        )

        # 补全 c_offsets 最后一个值（对齐 wrapper：c_offsets[-1] = b_offsets[-1] + B）
        self._c_offsets_ptr[-1] = self._b_offsets_ptr[-1] + self._B

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        return self._c_ptr

    def _split_lengths(self, total_seq, B):
        """把 total_seq 确定性地切成 B 段（段长可为 0，均匀铺开、可复现、无 random）"""
        if B <= 0:
            return []
        if total_seq <= 0:
            return [0] * B
        base = total_seq // B
        rem = total_seq - base * B
        # 用确定性 LCG 打散 base 附近的抖动，同时把余数分摊到前 rem 段，保证 sum 精确等于 total_seq
        lengths = []
        for i in range(B):
            jitter = self._hash(i, B) % (base + 1) if base > 0 else 0
            seg = base - jitter // 2 + (1 if i < rem else 0)
            lengths.append(max(0, seg))
        # 修正累计误差，保证总和 == total_seq
        diff = total_seq - sum(lengths)
        i = 0
        while diff != 0 and B > 0:
            idx = i % B
            if diff > 0:
                lengths[idx] += 1
                diff -= 1
            elif lengths[idx] > 0:
                lengths[idx] -= 1
                diff += 1
            i += 1
        return lengths

    def _hash(self, i, salt):
        """确定性 LCG（无 random，GPU/NPU 一致）"""
        x = ((i + 1) * 2654435761 + salt * 40503) & 0xFFFFFFFF
        x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
        return (x ^ (x >> 16)) & 0x7FFFFFFF

    def _cumsum_tensor(self, lengths, device):
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, dtype=torch.int32, device=device)

    def _rebuild_1d(self, raw, target_len, dtype, device):
        raw_flat = raw.reshape(-1).to(device=device, dtype=dtype)
        if target_len <= 0:
            return torch.zeros((0,), dtype=dtype, device=device)
        if raw_flat.shape[0] >= target_len:
            return raw_flat[:target_len].contiguous()
        if raw_flat.shape[0] == 0:
            return torch.zeros((target_len,), dtype=dtype, device=device)
        reps = (target_len + raw_flat.shape[0] - 1) // raw_flat.shape[0]
        tiled = raw_flat.repeat(reps)[:target_len]
        return tiled.contiguous()

    def _to_tensor(self, data, device, dtype=None):
        if isinstance(data, torch.Tensor):
            tensor = data.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
        if dtype is not None:
            return torch.tensor(data, dtype=dtype, device=device)
        return torch.tensor(data, device=device)
