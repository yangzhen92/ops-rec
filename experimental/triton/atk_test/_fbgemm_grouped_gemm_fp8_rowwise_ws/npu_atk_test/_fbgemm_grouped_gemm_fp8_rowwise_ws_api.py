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


import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

_FP8_EPS = 1e-12


def _build_m_sizes(M: int, G: int):
    base = M // G
    remainder = M - base * G
    return [base + (1 if i < remainder else 0) for i in range(G)]


def _get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _pick_fp8_dtype(torch):  # pylint: disable=redefined-outer-name
    if torch.version.hip:
        dtype = getattr(torch, "float8_e4m3fnuz", None)
        if dtype is not None:
            return dtype
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("torch 不支持 float8_e4m3fn/fnuz。需要 PyTorch 2.1+ 且设备支持 fp8。")
    return dtype


def _quantize_fp8_row(torch, a, pt_fp8_dtype):  # pylint: disable=redefined-outer-name
    max_fp8 = float(torch.finfo(pt_fp8_dtype).max)  # Python float
    eps = float(_FP8_EPS)

    row_max = torch.max(torch.abs(a), dim=-1)[0].to(torch.float32)
    row_max = torch.clamp(row_max, min=eps)

    scale = torch.tensor(max_fp8, dtype=torch.float32, device=a.device) / row_max
    scale = torch.where(torch.isinf(scale), torch.ones_like(scale), scale)

    a_fp8 = (a * scale[..., None]).to(pt_fp8_dtype)
    return a_fp8, (1.0 / scale)


@register("_fbgemm_grouped_gemm_fp8_rowwise_ws")
class _FbgemmGroupedGemmFp8RowwiseWsApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        device = _get_device()

        x_input = input_data.kwargs.get("x", None)
        w_input = input_data.kwargs.get("w", None)
        m_sizes_input = input_data.kwargs.get("m_sizes", None)

        if x_input is None or w_input is None or m_sizes_input is None:
            raise ValueError("x, w, m_sizes 三个输入都是必需的")

        if isinstance(x_input, torch.Tensor):
            x = x_input.to(device=device, dtype=torch.float32)
        else:
            x = torch.tensor(x_input, dtype=torch.float32, device=device)

        if isinstance(w_input, torch.Tensor):
            w = w_input.to(device=device, dtype=torch.float32)
        else:
            w = torch.tensor(w_input, dtype=torch.float32, device=device)

        if isinstance(m_sizes_input, torch.Tensor):
            m_sizes_raw = m_sizes_input.to(device)
        else:
            m_sizes_raw = torch.tensor(m_sizes_input, dtype=torch.int32, device=device)

        if x.dim() != 2 or w.dim() != 2:
            raise ValueError(f"x/w 必须是二维: x.dim={x.dim()}, w.dim={w.dim()}")

        M, K = x.shape
        G = m_sizes_raw.numel()  # 只取长度，值不可信
        if G <= 0:
            raise ValueError("m_sizes 长度必须为正")

        if w.shape[0] % G != 0:
            raise ValueError(f"w.shape[0]={w.shape[0]} 必须能被 G={G} 整除")
        N = w.shape[0] // G

        if w.shape[1] != K:
            raise ValueError(f"K 维不匹配: x.shape[1]={K}, w.shape[1]={w.shape[1]}")

        if K % 8 != 0 or N % 8 != 0:
            raise ValueError(f"FP8 路径要求 K 与 N 均为 8 的倍数: K={K}, N={N}")

        if w.dtype != x.dtype:
            w = w.to(x.dtype)

        m_sizes = torch.tensor(_build_m_sizes(M, G), dtype=torch.int32, device=device)
        if int(m_sizes.sum().item()) != M:
            raise ValueError(f"m_sizes 重建失败: sum={int(m_sizes.sum().item())}, M={M}")

        fp8_dtype = _pick_fp8_dtype(torch)
        x_fp8, x_scale = _quantize_fp8_row(torch, x, fp8_dtype)
        w_fp8, w_scale = _quantize_fp8_row(torch, w, fp8_dtype)

        # pylint: disable=attribute-defined-outside-init
        self._x_fp8 = x_fp8.contiguous()
        self._w_fp8 = w_fp8.contiguous()
        self._x_scale = x_scale.to(torch.float32).contiguous()
        self._w_scale = w_scale.to(torch.float32).contiguous()
        self._m_sizes = m_sizes.contiguous()
        self._device = device
        # pylint: enable=attribute-defined-outside-init

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        from _fbgemm_grouped_gemm_fp8_rowwise_ws_kernel import grouped_gemm_fp8_rowwise

        if self._device == "npu":
            torch.npu.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()

        y = grouped_gemm_fp8_rowwise(
            self._x_fp8,
            self._w_fp8,
            self._m_sizes,
            self._x_scale,
            self._w_scale,
            use_fast_accum=True,
            _use_warp_specialization=True,  # 请求 WS；NPU 上会被驳回，见文件头
        )

        if self._device == "npu":
            torch.npu.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()

        return y
