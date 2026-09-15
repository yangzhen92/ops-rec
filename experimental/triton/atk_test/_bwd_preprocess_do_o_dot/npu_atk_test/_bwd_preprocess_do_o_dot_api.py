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

# _bwd_preprocess_do_o_dot_api.py
"""
_bwd_preprocess_do_o_dot 算子的ATK自定义API执行文件

用于 Flash Attention 反向传播的预处理步骤：
- 输入：Out [B, T, D], do [B, T, D]
- 输出：delta [B, T]

delta = sum(Out * do, axis=1)，即每个时间步的 dot product 求和

计算公式：
- o: [B, T, D] - 前向传播的输出
- do: [B, T, D] - 反向传播传入的梯度
- delta: [B, T] - delta[t] = sum_d(o[b, t, d] * do[b, t, d])
"""

import torch
import triton
from _bwd_preprocess_do_o_dot_kernel import _bwd_preprocess_do_o_dot
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def get_device():
    """获取可用的设备，优先NPU"""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except (AttributeError, RuntimeError):
        pass
    return "cpu"


# BaseApi 通过 init_by_input_data 生命周期方法初始化运行时状态。
# pylint: disable=attribute-defined-outside-init
@register("_bwd_preprocess_do_o_dot")
class BwdPreprocessDoODotApi(BaseApi):
    """_bwd_preprocess_do_o_dot 算子的 API 封装"""

    def _to_tensor(self, input_tensor, device, dtype):
        return input_tensor.to(device=device, dtype=dtype)

    def init_by_input_data(self, input_data: InputDataset):
        """
        初始化操作，不计入性能统计
        """
        device = get_device()

        # 同步操作
        if device == "npu":
            torch.npu.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        # 获取输入参数
        out_input = input_data.kwargs.get("Out", None)
        do_input = input_data.kwargs.get("do", None)

        if out_input is None:
            raise ValueError("Out not found")
        if do_input is None:
            raise ValueError("do not found")

        # 转换到正确设备
        if not isinstance(out_input, torch.Tensor):
            Out = torch.tensor(out_input, dtype=torch.float32, device=device)
        else:
            Out = out_input.to(device)

        if not isinstance(do_input, torch.Tensor):
            do = torch.tensor(do_input, dtype=torch.float32, device=device)
        else:
            do = do_input.to(device)

        # 执行双标杆时精度会发生提升（fp16 / bf16 -> fp32, fp32 -> fp64）
        # 输入强转 fp32 让 kernel 编译通过；输出再转回原始 dtype
        if isinstance(Out, torch.Tensor):
            self._orig_dtype = Out.dtype
        else:
            self._orig_dtype = torch.float32
        if Out.dtype == torch.float64:
            Out = self._to_tensor(Out, device, dtype=torch.float32)
            do = self._to_tensor(do, device, dtype=torch.float32)

        # 确保 do 是连续的
        if not do.is_contiguous():
            do = do.contiguous()

        # 保存实例变量
        self._device = device
        self._Out = Out
        self._do = do
        self._B, self._T, self._D = Out.shape

        # 创建 delta 输出
        self._delta = torch.empty((self._B, self._T), dtype=Out.dtype, device=device)

        self._BLOCK_D = self._D

        def grid(meta):
            return (triton.cdiv(self._T, meta["BLOCK_T"]), self._B)

        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """
        执行 _bwd_preprocess_do_o_dot kernel
        """
        # 同步
        if self._device == "npu":
            torch.npu.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()

        # 调用 kernel
        _bwd_preprocess_do_o_dot[self._grid](
            self._Out,
            self._do,
            self._delta,
            self._T,
            self._Out.stride(0),
            self._Out.stride(1),
            self._Out.stride(2),
            self._do.stride(0),
            self._do.stride(1),
            self._do.stride(2),
            BLOCK_D=self._BLOCK_D,
        )

        # 同步
        if self._device == "npu":
            torch.npu.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()

        if with_output:
            # 输入曾被强转 fp32 以通过编译；此处把输出转回原始 dtype
            orig_dtype = getattr(self, "_orig_dtype", None)
            if orig_dtype is not None and self._delta.dtype != orig_dtype:
                self._delta = self._delta.to(orig_dtype)
            return self._delta
        return None
