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

# _fbgemm_grouped_gemm_fp8_rowwise_api.py
"""_fbgemm_grouped_gemm_fp8_rowwise 算子的 ATK 自定义 API 执行文件（FP8 行级量化）

被测入口（triton/_fbgemm_grouped_gemm_fp8_rowwise/kernel.py）：
    grouped_gemm_fp8_rowwise(x, w, m_sizes, x_scale, w_scale,
                             use_fast_accum=True, *,
                             _use_warp_specialization=True, ...) -> y

本文件固定 _use_warp_specialization=False 走基础 kernel
`_fbgemm_grouped_gemm_fp8_rowwise`；WS 路径由 _fp8_rowwise_ws 套件覆盖。

三条设计决定，改动前请先读：

1. FP8 张量与 scale 都在本文件里生成，不从 yaml 来。
   ATK 的 DEFAULT_DTYPES 里没有任何 fp8 类型，声明 fp8_e4m3 会在造数据阶段 KeyError。
   所以 yaml 只给 bf16，这里按行量化成 fp8 并导出倒数尺度——与 FBGEMM 原生单测
   （grouped_gemm_test.py 里 quantize_fp8_row(a) 的用法）同一套路。
   scale 与张量数值严格耦合（A_fp8 * a_scale ≈ A_bf16），独立随机的 scale
   会让输出在数学上失去意义。

2. 量化实现内联在本文件，不 import fbgemm_gpu。
   原生 quantize_fp8_row 在 fp8_gemm.py 里，import 它会拖进整个 fbgemm_gpu 包依赖。
   下面 _quantize_fp8_row 复刻其 torch fallback 分支（fp8_gemm.py:2976-2994），
   逐行等价：row_max → clamp(min=eps) → scale = max_fp8/row_max → 返回 1/scale。

3. 顶层不 import kernel（它的模块顶层会探测设备）。
   torch 本身可以安全地顶层 import（参考其他成功的算子 API），
   但 kernel 模块初始化时会调设备探测逻辑，必须推迟到 __call__ 内才 import。

   注意 fp8 量化必须在 device 上做（CPU 不支持 fp8 算子），所以量化也放在
   首次 __call__ 里，init 阶段直接搬 bf16 到设备。
"""

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


# fp8 量化的最小分母，对齐 get_fp8_constants() 的 eps
_FP8_EPS = 1e-12


def _build_m_sizes(M: int, G: int):
    """确定性切分：把 M 行分成 G 段，保证 sum == M"""
    base = M // G
    remainder = M - base * G
    return [base + (1 if i < remainder else 0) for i in range(G)]


def _quantize_fp8_row(torch_module, a, pt_fp8_dtype):
    """按行量化到 fp8，返回 (a_fp8, 倒数尺度)

    复刻 FBGEMM quantize_fp8_row 的 torch fallback 分支
    （fp8_gemm.py:2976-2994），保持逐行等价：

        row_max = max(|a|, dim=-1)         行内绝对值最大
        row_max = clamp(row_max, min=eps)  防除零
        scale   = max_fp8 / row_max        放大到 fp8 满量程
        a_fp8   = (a * scale).to(fp8)
        return a_fp8, 1 / scale            倒数尺度：a_fp8 * (1/scale) ≈ a

    kernel 侧用的是这个倒数尺度：c = acc * a_scale[行] * b_scale[列]。
    """
    max_fp8 = torch_module.finfo(pt_fp8_dtype).max

    row_max = torch_module.max(torch_module.abs(a), dim=-1)[0]
    row_max = torch_module.clamp(row_max, min=_FP8_EPS)

    scale = max_fp8 / row_max.to(torch_module.float32)
    # row_max 极小时 scale 会溢出成 inf，原生实现在这里回落到 1.0
    scale[scale == float("inf")] = 1.0

    a_fp8 = (a * scale[..., None]).to(pt_fp8_dtype)
    return a_fp8, (1.0 / scale)


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


def _pick_fp8_dtype(torch_module):
    """选择平台支持的 fp8 类型（ROCm 优先 fnuz，否则 fn）"""
    if torch_module.version.hip:
        dtype = getattr(torch, "float8_e4m3fnuz", None)
        if dtype is not None:
            return dtype
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("torch 不支持 float8_e4m3fn/fnuz。需要 PyTorch 2.1+ 且设备支持 fp8。")
    return dtype


@register("_fbgemm_grouped_gemm_fp8_rowwise")
class _FbgemmGroupedGemmFp8RowwiseApi(BaseApi):
    """_fbgemm_grouped_gemm_fp8_rowwise 算子的 API 封装"""

    # 走基础 kernel。_fp8_rowwise_ws 目录是独立的 ATK 工作目录（各有自己的
    # grouped_gemm.py 副本），无法跨目录继承，那边是一份对称的独立实现。
    _USE_WARP_SPECIALIZATION = False

    def init_by_input_data(self, input_data: InputDataset):
        """预处理（不计时）：获取输入、重建 m_sizes、搬到设备、量化"""
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
        G = m_sizes_raw.numel()  # 只取长度，值不可信
        if G <= 0:
            raise ValueError("m_sizes 长度必须为正")

        if w.shape[0] % G != 0:
            raise ValueError(f"w.shape[0]={w.shape[0]} 必须能被 G={G} 整除")
        N = w.shape[0] // G

        if w.shape[1] != K:
            raise ValueError(f"K 维不匹配: x.shape[1]={K}, w.shape[1]={w.shape[1]}")

        # fp8 路径要求 K/N 都是 8 的倍数：否则 _grouped_gemm 会 assert x_scale is None，
        # 即量化路径根本走不通。generator 已保证，这里显式拦一道便于定位。
        if K % 8 != 0 or N % 8 != 0:
            raise ValueError(f"FP8 路径要求 K 与 N 均为 8 的倍数: K={K}, N={N}")

        if w.dtype != x.dtype:
            w = w.to(x.dtype)

        # 双标杆 / 单标杆真值路径会把输入升精度（fp16/bf16 -> fp32，fp32 -> fp64）。
        # 与 _bwd_preprocess_do_o_dot/acc 的约定一致：fp64 一律强转 fp32 再量化，
        # 避免量化路径在 fp64 上数值/类型不一致。
        if x.dtype == torch.float64:
            x = x.to(torch.float32)
            w = w.to(torch.float32)

        # 重建 m_sizes（API 内确定性生成，不信任 ATK 传入值）
        m_sizes = torch.tensor(_build_m_sizes(M, G), dtype=torch.int32, device=device)
        if int(m_sizes.sum().item()) != M:
            raise ValueError(f"m_sizes 重建失败: sum={int(m_sizes.sum().item())}, M={M}")

        # FP8 量化（必须在 device 上做）
        fp8_dtype = _pick_fp8_dtype(torch)
        x_fp8, x_scale = _quantize_fp8_row(torch, x, fp8_dtype)
        w_fp8, w_scale = _quantize_fp8_row(torch, w, fp8_dtype)

        # 确保连续性，scale 转 fp32
        # BaseApi initializes execution state from the generated input dataset.
        # pylint: disable=attribute-defined-outside-init
        self._x_fp8 = x_fp8.contiguous()
        self._w_fp8 = w_fp8.contiguous()
        self._x_scale = x_scale.to(torch.float32).contiguous()
        self._w_scale = w_scale.to(torch.float32).contiguous()
        self._m_sizes = m_sizes.contiguous()
        self._device = device
        # pylint: enable=attribute-defined-outside-init

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行算子（计时）"""
        # kernel 模块顶层有设备探测，推迟到此处 import
        from _fbgemm_grouped_gemm_fp8_rowwise_kernel import grouped_gemm_fp8_rowwise

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
            _use_warp_specialization=self._USE_WARP_SPECIALIZATION,
        )

        if self._device == "npu":
            torch.npu.synchronize()
        elif self._device == "cuda":
            torch.cuda.synchronize()

        return y
