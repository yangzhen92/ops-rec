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

# _fbgemm_grouped_gemm_ws_api.py
"""_fbgemm_grouped_gemm_ws 算子的 ATK 自定义 API 执行文件（bf16 + Warp Specialization）

被测入口（triton/_fbgemm_grouped_gemm_ws/kernel.py）：
    grouped_gemm(x, w, m_sizes, use_fast_accum=True, *,
                 _use_warp_specialization=True, ...) -> y

kernel.py 里**没有**独立的 grouped_gemm_ws() 函数：WS 是同一个入口的
关键字开关。本文件传 _use_warp_specialization=True 请求 WS 路径。

⚠️ 在 Ascend NPU 上这个请求会被驳回：_grouped_gemm 检测到 HAS_NPU 时 warn
   "Warp specialization is disabled as it is not supported on Ascend NPU"
   并强制置 False，实际走 kernel _fbgemm_grouped_gemm（基础版）。
   K 或 N 不是 8 的倍数时同样会降级。所以本套件在 NPU 上采到的是基础版数据，
   用途是与 GPU 侧真 WS 结果逐 case 对齐，不是验证 NPU 的 WS 能力。

两条踩过的坑，改动前请先读：

1. 顶层不 import grouped_gemm（它的模块顶层会探测设备）。
   torch 本身可以安全地顶层 import（参考其他成功的算子 API），
   但 grouped_gemm 模块初始化时会调设备探测逻辑，必须推迟到 __call__ 内才 import。

2. m_sizes 的值由本文件重建，不信任 ATK 传入值。
   kernel 要求 sum(m_sizes) == M，而 ATK 只能注入 shape、无法注入精确序列。
   这里只取长度定 G，值一律确定性重算——重建算法与非 WS 版逐字相同，保证两边对齐。
"""

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def _build_m_sizes(M: int, G: int):
    """确定性切分：把 M 行分成 G 段，保证 sum == M

    与非 WS 版实现逐字一致，两套用例才能逐 case 比对。
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


@register("_fbgemm_grouped_gemm_ws")
class _FbgemmGroupedGemmWsApi(BaseApi):
    """_fbgemm_grouped_gemm_ws 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理（不计时）：获取输入、重建 m_sizes、搬到设备"""
        device = _get_device()

        x_input = input_data.kwargs.get("x", None)
        w_input = input_data.kwargs.get("w", None)
        m_sizes_input = input_data.kwargs.get("m_sizes", None)

        if x_input is None or w_input is None or m_sizes_input is None:
            raise ValueError("x, w, m_sizes 三个输入都是必需的")

        # 转 tensor（保留 ATK 给的 dtype）
        if isinstance(x_input, torch.Tensor):
            x = x_input.to(device)
        else:
            x = torch.tensor(x_input, dtype=torch.bfloat16, device=device)

        if isinstance(w_input, torch.Tensor):
            w = w_input.to(device)
        else:
            w = torch.tensor(w_input, dtype=torch.bfloat16, device=device)
        if isinstance(m_sizes_input, torch.Tensor):
            m_sizes_raw = m_sizes_input.to(device)
        else:
            m_sizes_raw = torch.tensor(m_sizes_input, dtype=torch.int32, device=device)

        if x.dim() != 2 or w.dim() != 2:
            raise ValueError(f"x/w 必须是二维: x.dim={x.dim()}, w.dim={w.dim()}")

        M, K = x.shape
        G = m_sizes_raw.numel()  # 只取长度，值不可信（见文件头第 2 条）
        if G <= 0:
            raise ValueError("m_sizes 长度必须为正")

        if w.shape[0] % G != 0:
            raise ValueError(f"w.shape[0]={w.shape[0]} 必须能被 G={G} 整除")
        if w.shape[1] != K:
            raise ValueError(f"K 维不匹配: x.shape[1]={K}, w.shape[1]={w.shape[1]}")

        if w.dtype != x.dtype:
            w = w.to(x.dtype)

        # 重建 m_sizes（API 内确定性生成，不信任 ATK 传入值）
        m_sizes = torch.tensor(_build_m_sizes(M, G), dtype=torch.int32, device=device)
        if int(m_sizes.sum().item()) != M:
            raise ValueError(f"m_sizes 重建失败: sum={int(m_sizes.sum().item())}, M={M}")

        # 确保连续性
        # pylint: disable=attribute-defined-outside-init
        self._x = x.contiguous()
        self._w = w.contiguous()
        self._m_sizes = m_sizes.contiguous()
        self._device = device
        # pylint: enable=attribute-defined-outside-init

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行算子（计时）"""
        # kernel 模块顶层有设备探测，推迟到此处 import
        from _fbgemm_grouped_gemm_ws_kernel import grouped_gemm

        if self._device == "npu":
            torch.npu.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()

        y = grouped_gemm(
            self._x,
            self._w,
            self._m_sizes,
            use_fast_accum=True,
            _use_warp_specialization=True,  # 请求 WS；NPU 上会被驳回，见文件头
        )

        if self._device == "npu":
            torch.npu.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()

        return y
