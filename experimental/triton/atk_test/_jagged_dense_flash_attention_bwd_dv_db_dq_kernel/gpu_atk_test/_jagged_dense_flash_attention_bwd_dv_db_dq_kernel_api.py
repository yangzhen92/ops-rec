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

import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

# 从被测代码直接引用 triton kernel
from _jagged_dense_flash_attention_bwd_dv_db_dq_kernel import (
    _jagged_dense_flash_attention_bwd_dv_db_dq_kernel,
    _bwd_preprocess_do_o_dot,
)


def get_device():
    """获取可用的设备，优先NPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


@register("_jagged_dense_flash_attention_bwd_dv_db_dq_kernel")
class JaggedDenseFlashAttentionBwdDvDbDqApi(BaseApi):
    """_jagged_dense_flash_attention_bwd_dv_db_dq_kernel 算子的 API 封装"""

    def __init__(self):
        super().__init__()
        # 实例属性声明：实际值由 init_by_input_data 填充
        self._device = None
        self._Q = None
        self._K = None
        self._V = None
        self._Out = None
        self._lse = None
        self._do = None
        self._attn_bias = None
        self._jagged_offsets = None
        self._max_seq_len = None
        self._allow_tf32 = None
        self._B = None
        self._D = None
        self._T = None
        self._sum_B = None
        self._dq = None
        self._dv = None
        self._dbias = None
        self._delta = None
        self._BLOCK_T = None
        self._BLOCK_L = None
        self._BLOCK_D = None
        self._num_blocks_l = None
        self._grid = None

    def init_by_input_data(self, input_data: InputDataset):
        """
        初始化操作，不计入性能统计
        """
        device = get_device()

        # 同步操作
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 获取输入参数
        Q_input = input_data.kwargs.get("Q", None)
        K_input = input_data.kwargs.get("K", None)
        V_input = input_data.kwargs.get("V", None)
        Out_input = input_data.kwargs.get("Out", None)
        lse_input = input_data.kwargs.get("lse", None)
        do_input = input_data.kwargs.get("do", None)
        attn_bias_input = input_data.kwargs.get("attn_bias", None)
        jagged_offsets_input = input_data.kwargs.get("jagged_offsets", None)
        max_seq_len = input_data.kwargs.get("max_seq_len", 128)
        allow_tf32 = input_data.kwargs.get("allow_tf32", False)

        # 参数校验
        if Q_input is None:
            raise ValueError("Q not found")
        if K_input is None:
            raise ValueError("K not found")
        if V_input is None:
            raise ValueError("V not found")
        if Out_input is None:
            raise ValueError("Out not found")
        if lse_input is None:
            raise ValueError("lse not found")
        if do_input is None:
            raise ValueError("do not found")
        if attn_bias_input is None:
            raise ValueError("attn_bias not found")
        if jagged_offsets_input is None:
            raise ValueError("jagged_offsets not found")

        # 转换 Q 到正确设备 (jagged tensor: [sum_B, D])
        if not isinstance(Q_input, torch.Tensor):
            Q = torch.tensor(Q_input, dtype=torch.float32, device=device)
        else:
            Q = Q_input.to(device)

        # 转换 K 到正确设备 (dense tensor: [B, D, T])
        if not isinstance(K_input, torch.Tensor):
            K = torch.tensor(K_input, dtype=torch.float32, device=device)
        else:
            K = K_input.to(device)

        # 转换 V 到正确设备 (jagged tensor: [sum_B, D])
        if not isinstance(V_input, torch.Tensor):
            V = torch.tensor(V_input, dtype=torch.float32, device=device)
        else:
            V = V_input.to(device)

        # 转换 Out 到正确设备 (dense tensor: [B, T, D])
        if not isinstance(Out_input, torch.Tensor):
            Out = torch.tensor(Out_input, dtype=torch.float32, device=device)
        else:
            Out = Out_input.to(device)

        # 转换 lse 到正确设备 (dense tensor: [B, T])
        if not isinstance(lse_input, torch.Tensor):
            lse = torch.tensor(lse_input, dtype=torch.float32, device=device)
        else:
            lse = lse_input.to(device)

        # 转换 do 到正确设备 (dense tensor: [B, T, D])
        if not isinstance(do_input, torch.Tensor):
            do = torch.tensor(do_input, dtype=torch.float32, device=device)
        else:
            do = do_input.to(device)

        # 转换 attn_bias 到正确设备 (dense tensor: [B, N, T])
        if not isinstance(attn_bias_input, torch.Tensor):
            attn_bias = torch.tensor(attn_bias_input, dtype=torch.float32, device=device)
        else:
            attn_bias = attn_bias_input.to(device)

        # 转换 jagged_offsets 到正确设备 (tensor: [B + 1])
        if not isinstance(jagged_offsets_input, torch.Tensor):
            jagged_offsets = torch.tensor(jagged_offsets_input, dtype=torch.int64, device=device)
        else:
            jagged_offsets = jagged_offsets_input.to(device)

        # 确保所有输入是连续的
        if not Q.is_contiguous():
            Q = Q.contiguous()
        if not K.is_contiguous():
            K = K.contiguous()
        if not V.is_contiguous():
            V = V.contiguous()
        if not Out.is_contiguous():
            Out = Out.contiguous()
        if not do.is_contiguous():
            do = do.contiguous()
        if not attn_bias.is_contiguous():
            attn_bias = attn_bias.contiguous()
        if not jagged_offsets.is_contiguous():
            jagged_offsets = jagged_offsets.contiguous()

        # 保存实例变量
        self._device = device
        self._Q = Q
        self._K = K
        self._V = V
        self._Out = Out
        self._lse = lse
        self._do = do
        self._attn_bias = attn_bias
        self._jagged_offsets = jagged_offsets
        self._max_seq_len = max_seq_len
        self._allow_tf32 = allow_tf32

        # 获取维度信息
        self._B = K.shape[0]  # batch size
        self._D = K.shape[1]  # hidden dimension
        self._T = K.shape[2]  # max sequence length (key/value length)

        # ======== 约束：jagged_offsets[-1] = sum_B，且每个序列长度 <= max_seq_len ========
        # 使用确定性方式生成 offset，确保不同机器结果一致
        # 规则：jagged_offsets[0] = 0, jagged_offsets[i] = sum of sequence lengths up to i
        sum_B_from_q = Q.shape[0]

        # 使用确定性算法生成 jagged_offsets
        step = max(1, self._T // self._B)
        segment_lengths = []
        remaining = sum_B_from_q
        for i in range(self._B):
            # 每个 batch 的序列长度不能超过 max_seq_len（kernel 会截断）
            seg_len = min(step + (i % 2), self._max_seq_len, remaining - (self._B - i - 1))
            seg_len = max(1, seg_len)
            segment_lengths.append(seg_len)
            remaining -= seg_len

        # 调整使总和等于 sum_B
        diff = sum_B_from_q - sum(segment_lengths)
        if diff != 0:
            segment_lengths[0] = segment_lengths[0] + diff

        # 生成累计偏移量
        cumulative = [0]
        for seg_len in segment_lengths:
            cumulative.append(cumulative[-1] + seg_len)

        jagged_offsets = torch.tensor(cumulative, dtype=torch.int64, device=device)
        jagged_offsets = jagged_offsets.contiguous()

        self._jagged_offsets = jagged_offsets
        self._sum_B = jagged_offsets[-1].item()  # sum of all sequence lengths

        # 创建输出 tensor
        self._dq = torch.zeros_like(Q)
        self._dv = torch.zeros_like(V)
        self._dbias = torch.zeros_like(attn_bias)

        # 创建 delta 中间结果 (与 lse 形状相同)
        self._delta = torch.empty_like(lse)

        # 设置 block 大小
        self._BLOCK_T = 32
        self._BLOCK_L = 32
        self._BLOCK_D = self._D

        # 计算 grid
        self._num_blocks_l = triton.cdiv(self._max_seq_len, self._BLOCK_L)
        self._grid = (self._num_blocks_l, self._B)

        # 预计算 delta (调用 _bwd_preprocess_do_o_dot kernel)
        num_blocks_t = triton.cdiv(self._T, self._BLOCK_T)
        _bwd_preprocess_do_o_dot[(num_blocks_t, self._B)](
            Out,
            do,
            self._delta,
            self._T,
            Out.stride(0),
            Out.stride(1),
            Out.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            self._BLOCK_T,
            self._BLOCK_D,
        )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """
        执行 _jagged_dense_flash_attention_bwd_dv_db_dq_kernel kernel
        """
        # 同步
        if self._device == 'npu':
            torch.npu.synchronize()
        elif self._device == 'cuda':
            torch.cuda.synchronize()

        # 调用 kernel
        _jagged_dense_flash_attention_bwd_dv_db_dq_kernel[self._grid](
            self._Q,
            self._K,
            self._V,
            self._attn_bias,
            self._jagged_offsets,
            self._Out,
            self._do,
            self._lse,
            self._delta,
            self._dq,
            None,  # dk (not computed by this kernel)
            self._dv,
            self._dbias,
            self._max_seq_len,
            self._Q.stride(0),
            self._Q.stride(1),
            self._K.stride(0),
            self._K.stride(1),
            self._K.stride(2),
            self._V.stride(0),
            self._V.stride(1),
            self._attn_bias.stride(0),
            self._attn_bias.stride(1),
            self._attn_bias.stride(2),
            self._Out.stride(0),
            self._Out.stride(1),
            self._Out.stride(2),
            self._dq.stride(0),
            self._dq.stride(1),
            self._dv.stride(0),
            self._dv.stride(1),
            self._dbias.stride(0),
            self._dbias.stride(1),
            self._dbias.stride(2),
            self._do.stride(0),
            self._do.stride(1),
            self._do.stride(2),
            self._T,
            self._BLOCK_T,
            self._BLOCK_L,
            self._BLOCK_D,
            self._allow_tf32,
        )

        # 同步并停止 profiler
        if self._device == 'npu':
            torch.npu.synchronize()
        elif self._device == 'cuda':
            torch.cuda.synchronize()

        return self._dq, self._dv, self._dbias
