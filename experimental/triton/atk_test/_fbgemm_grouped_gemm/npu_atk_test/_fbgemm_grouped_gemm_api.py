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
#     limitations under the License.
# ==============================================================================
# _fbgemm_grouped_gemm_api.py
"""_fbgemm_grouped_gemm 算子的 ATK 自定义 API 执行文件（bf16 基础版）

被测入口（triton/_fbgemm_grouped_gemm/kernel.py）：
    grouped_gemm(x, w, m_sizes, use_fast_accum=True, *,
                 _use_warp_specialization=True,
                 _output_tensor=None, _scatter_add_indices=None) -> y

本文件固定 _use_warp_specialization=False，确保走基础 kernel
`_fbgemm_grouped_gemm`；WS 路径由 _fbgemm_grouped_gemm_ws 套件单独覆盖。
_output_tensor / _scatter_add_indices 保持 None：传了会切换成 fused
scatter-add（就地 atomic 累加）语义，与本算子要测的 y = A_i @ B_i^T 不是一回事。
"""

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def _build_m_sizes(M: int, G: int):
    """确定性切分：把 M 行分成 G 段，保证 sum == M

    均分 + 余数摊到前 remainder 段。不引入随机，GPU 与 NPU 两侧结果逐位一致。
    """
    base = M // G
    remainder = M - base * G
    return [base + (1 if i < remainder else 0) for i in range(G)]


def _get_device():
    """获取可用设备，优先 NPU"""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# BaseApi 通过 init_by_input_data 生命周期方法初始化运行时状态。
# pylint: disable=attribute-defined-outside-init
@register("_fbgemm_grouped_gemm")
class _FbgemmGroupedGemmApi(BaseApi):
    """_fbgemm_grouped_gemm 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理（不计时）：获取输入、重建 m_sizes、搬到设备"""
        device = _get_device()

        x_input = input_data.kwargs.get("x", None)
        w_input = input_data.kwargs.get("w", None)
        m_sizes_input = input_data.kwargs.get("m_sizes", None)

        if x_input is None or w_input is None or m_sizes_input is None:
            raise ValueError("x, w, m_sizes 三个输入都是必需的")

        # 转 tensor（原生版本：保持输入原始精度）
        if isinstance(x_input, torch.Tensor):
            x = x_input.to(device=device)
        else:
            x = torch.tensor(x_input, device=device)

        if isinstance(w_input, torch.Tensor):
            w = w_input.to(device=device)
        else:
            w = torch.tensor(w_input, device=device)

        if isinstance(m_sizes_input, torch.Tensor):
            m_sizes_raw = m_sizes_input.to(device)
        else:
            m_sizes_raw = torch.tensor(m_sizes_input, dtype=torch.int32, device=device)

        if x.dim() != 2 or w.dim() != 2:
            raise ValueError(f"x/w 必须是二维: x.dim={x.dim()}, w.dim={w.dim()}")

        M, K = x.shape
        # 只取长度定 G，值不可信（见文件头第 2 条）
        G = m_sizes_raw.numel()
        if G <= 0:
            raise ValueError("m_sizes 长度必须为正")

        if w.shape[0] % G != 0:
            raise ValueError(f"w.shape[0]={w.shape[0]} 必须能被 G={G} 整除")

        if w.shape[1] != K:
            raise ValueError(f"K 维不匹配: x.shape[1]={K}, w.shape[1]={w.shape[1]}")

        # w 与 x 同 dtype：kernel 用单一 dtsize 做 config 剪枝，混 dtype 会误剪
        if w.dtype != x.dtype:
            w = w.to(x.dtype)

        # 双标杆 / 单标杆真值路径会把输入升精度（fp16/bf16 -> fp32，fp32 -> fp64）。
        # kernel 的 accumulator 硬编码 fp32，fp64 输入会使 tl.dot 返回 fp64，
        # 触发 "Loop-carried variable accumulator ... re-assigned to ... fp64" 编译错误。
        # 与 _bwd_preprocess_do_o_dot/acc 的约定一致：fp64 一律强转 fp32 让编译通过。
        # （kernel 输出恒为 bf16，与输入 dtype 无关，故无需把输出转回 fp64。）
        # （此为 atk 测试框架的问题，与代码本身无关）
        if x.dtype == torch.float64:
            x = x.to(torch.float32)
            w = w.to(torch.float32)

        # 重建 m_sizes（API 内确定性生成，不信任 ATK 传入值）
        m_sizes = torch.tensor(_build_m_sizes(M, G), dtype=torch.int32, device=device)
        if int(m_sizes.sum().item()) != M:
            raise ValueError(f"m_sizes 重建失败: sum={int(m_sizes.sum().item())}, M={M}")

        # 确保连续性
        self._x = x.contiguous()
        self._w = w.contiguous()
        self._m_sizes = m_sizes.contiguous()
        self._device = device

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行算子（计时）"""
        # kernel 模块顶层有设备探测，推迟到此处 import
        from _fbgemm_grouped_gemm_kernel import grouped_gemm

        if self._device == "npu":
            torch.npu.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()

        y = grouped_gemm(
            self._x,
            self._w,
            self._m_sizes,
            use_fast_accum=True,
            _use_warp_specialization=False,  # 基础 kernel 路径，见文件头
        )

        if self._device == "npu":
            torch.npu.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()

        return y
