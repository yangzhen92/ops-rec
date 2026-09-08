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

# _group_norm_mul_dropout_fwd_api.py
"""_group_norm_mul_dropout_fwd 算子的 ATK 自定义 API。"""

import torch
import triton
from _group_norm_mul_dropout_fwd_kernel import _group_norm_mul_dropout_fwd
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
@register("_group_norm_mul_dropout_fwd")
class GroupNormMulDropoutFwdApi(BaseApi):
    """_group_norm_mul_dropout_fwd 算子的 API 封装"""

    def _to_tensor(self, input_tensor, device, dtype):
        return input_tensor.to(device=device, dtype=dtype)

    def init_by_input_data(self, input_data: InputDataset):
        """初始化输入和输出，不计入性能统计。"""
        device = get_device()

        # 同步操作
        if device == "npu":
            torch.npu.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        # 获取输入参数
        X_input = input_data.kwargs.get("X", None)
        U_input = input_data.kwargs.get("U", None)
        W_input = input_data.kwargs.get("W", None)
        Bias_input = input_data.kwargs.get("Bias", None)

        if X_input is None:
            raise ValueError("X not found")
        if U_input is None:
            raise ValueError("U not found")
        if W_input is None:
            raise ValueError("W not found")
        if Bias_input is None:
            raise ValueError("Bias not found")

        # 转换到正确设备
        if not isinstance(X_input, torch.Tensor):
            X = torch.tensor(X_input, dtype=torch.float32, device=device)
        else:
            X = X_input.to(device)

        if not isinstance(U_input, torch.Tensor):
            U = torch.tensor(U_input, dtype=torch.float32, device=device)
        else:
            U = U_input.to(device)

        if not isinstance(W_input, torch.Tensor):
            W = torch.tensor(W_input, dtype=torch.float32, device=device)
        else:
            W = W_input.to(device)

        if not isinstance(Bias_input, torch.Tensor):
            Bias = torch.tensor(Bias_input, dtype=torch.float32, device=device)
        else:
            Bias = Bias_input.to(device)

        # 执行双标杆时精度会发生提升（fp16 / bf16 -> fp32, fp32 -> fp64）
        # 输入强转 fp32 让 kernel 编译通过；输出再转回原始 dtype
        if isinstance(X, torch.Tensor):
            self._orig_dtype = X.dtype
        else:
            self._orig_dtype = torch.float32
        if X.dtype == torch.float64:
            X = self._to_tensor(X, device, dtype=torch.float32)
            U = self._to_tensor(U, device, dtype=torch.float32)
            W = self._to_tensor(W, device, dtype=torch.float32)
            Bias = self._to_tensor(Bias, device, dtype=torch.float32)

        # 确保数据是连续的
        if not X.is_contiguous():
            X = X.contiguous()
        if not U.is_contiguous():
            U = U.contiguous()
        if not W.is_contiguous():
            W = W.contiguous()
        if not Bias.is_contiguous():
            Bias = Bias.contiguous()

        # 保存实例变量
        self._device = device
        self._X = X
        self._U = U
        self._W = W
        self._Bias = Bias
        self._B, self._H, self._D = X.shape

        # 创建输出
        self._Mean = torch.empty((self._B, self._H), dtype=X.dtype, device=device)
        self._Rstd = torch.empty((self._B, self._H), dtype=X.dtype, device=device)
        self._Y = torch.empty((self._B, self._H, self._D), dtype=X.dtype, device=device)

        # NPU kernel 的 BLOCK_H 由 autotune 选择
        self._BLOCK_D = self._D

        # 计算 grid
        def grid(meta):
            return (self._B, triton.cdiv(self._H, meta["BLOCK_H"]))

        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行 _group_norm_mul_dropout_fwd kernel。"""
        # 同步
        if self._device == "npu":
            torch.npu.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()

        eps = 1e-6
        seed = 42
        dropout_ratio = 0

        # 调用 kernel
        _group_norm_mul_dropout_fwd[self._grid](
            self._X,
            self._U,
            self._Y,
            self._W,
            self._Bias,
            self._Mean,
            self._Rstd,
            self._D,
            self._H,
            eps,
            seed,
            dropout_ratio,
            self._X.stride(0),
            self._U.stride(0),
            self._Y.stride(0),
            BLOCK_D=self._BLOCK_D,
            TRAINING=False,  # GPU和NPU两端设置seed后会生成不一致的数据，导致精度无法通过，因此不测试dropout
            CONCAT_UX=False,
        )

        # 同步
        if self._device == "npu":
            torch.npu.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()

        if with_output:
            # 输入曾被强转 fp32 以通过编译；此处把输出转回原始 dtype
            orig_dtype = getattr(self, "_orig_dtype", None)
            if orig_dtype is not None and self._Y.dtype != orig_dtype:
                self._Y = self._Y.to(orig_dtype)
                self._Mean = self._Mean.to(orig_dtype)
                self._Rstd = self._Rstd.to(orig_dtype)
            return self._Y, self._Mean, self._Rstd
        return None
