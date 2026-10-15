#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
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

# kernel.py
"""_fbgemm_grouped_gemm_fp8_rowwise 的算子 kernel 与分发逻辑（FP8 行级量化）

被测入口（triton/_fbgemm_grouped_gemm_fp8_rowwise/kernel.py）：
    grouped_gemm_fp8_rowwise(x, w, m_sizes, x_scale, w_scale, use_fast_accum=True, *,
                             _use_warp_specialization=False, _output_tensor=None,
                             _scatter_add_indices=None) -> y

语义：对 G 组分别做 y = A_i @ B_i^T，输入为按行量化的 FP8 张量，输出按行/列
反量化 scale 还原：
    x        [M, K]      FP8，按 m_sizes 切成 G 段
    w        [G*N, K]    FP8 stacked 权重（转置存储），第 i 段为 B_i [N, K]
    x_scale  [M]         每行倒数尺度（A_fp8 * x_scale ≈ A）
    w_scale  [G*N]       每行倒数尺度
    m_sizes  [G]         sum(m_sizes) == M
    y        [M, N]      dtype 恒为 bf16
"""

import sys
import warnings
from typing import Optional

import torch
import triton.language as tl

import triton

# Ascend NPU support
try:
    import torch_npu  # noqa: F401

    HAS_NPU = torch.npu.is_available()
except ImportError:
    HAS_NPU = False


# ============================================================================
# Utils functions (inlined from fbgemm_gpu.experimental.gemm.triton_gemm.utils)
# Modified to support Ascend NPU and avoid CUDA initialization at module load
# ============================================================================


def map_dtype_to_triton(dtype: torch.dtype) -> tl.dtype:
    """Maps torch dtype to triton dtype."""
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.float32:
        return tl.float32
    elif dtype == torch.int32:
        return tl.int32
    elif dtype == torch.float8_e4m3fn:
        # NPU and AMD use E4M3 (float8e4b8), NVIDIA uses E4NV
        if HAS_NPU or torch.version.hip:
            return tl.float8e4b8
        else:
            return tl.float8e4nv
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


# Check if we have TMA descriptor support (Triton 3.5+ uses make_tensor_descriptor)
# Note: TMA is only supported on NVIDIA Hopper GPUs, not on NPU or AMD
HAS_TMA_DESC = hasattr(tl, "make_tensor_descriptor") and not HAS_NPU

if HAS_TMA_DESC:
    print(
        "TMA benchmarks will be running with experimental grid constant TMA descriptor.",
        file=sys.stderr,
    )
else:
    if HAS_NPU:
        print(
            "Running on Ascend NPU: TMA is not supported.",
            file=sys.stderr,
        )
    else:
        print(
            "TMA benchmarks will be running without grid constant TMA descriptor.",
            file=sys.stderr,
        )


class TmaAutoTuneHelper:
    """
    Helper class for TMA descriptor management.

    Note: This helper is only used on NVIDIA GPUs with TMA support.
    On NPU, TMA operations are disabled and this helper won't be instantiated
    (USE_TMA_LOAD is forced False on NPU / non-Hopper).
    """

    # Duck typing wrapper to implement the same interface as TmaDescKernelParam
    class KernelParamWrapper:
        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        if HAS_NPU:
            raise RuntimeError(
                "TmaAutoTuneHelper cannot be used on Ascend NPU. "
                "TMA (Tensor Memory Accelerator) is a NVIDIA Hopper-specific feature."
            )

        # Triton API 兼容性：新版本改名 fill_1d_tma_descriptor → fill_tma_descriptor
        utils = triton.runtime.driver.active.utils
        if hasattr(utils, "fill_1d_tma_descriptor"):
            self.fill_1d_tma_descriptor_inner = utils.fill_1d_tma_descriptor
            self.fill_2d_tma_descriptor_inner = utils.fill_2d_tma_descriptor
        else:
            # 新版 Triton 使用统一的 fill_tma_descriptor
            self.fill_1d_tma_descriptor_inner = utils.fill_tma_descriptor
            self.fill_2d_tma_descriptor_inner = utils.fill_tma_descriptor

        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    def init_tma_descriptor(self, name):
        """Initialize TMA descriptor (call outside grid lambda)"""
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8)
        else:
            self.cuda_descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8)

    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        """Fill 1D TMA descriptor (call inside grid lambda)"""
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    def fill_2d_tma_descriptor(self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size):
        """Fill 2D TMA descriptor (call inside grid lambda)"""
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        """Get TMA descriptor kernel parameter"""
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]


# FP8 dtype selection - use E4M3 for Ascend NPU and AMD, E4NV for NVIDIA
TT_FP8_DTYPE = tl.float8e4b8 if torch.version.hip else tl.float8e4nv


@triton.jit
def _fbgemm_grouped_gemm_fp8_rowwise(
    a_desc_ptr,
    a_scale_ptr,
    b_desc_ptr,
    b_scale_ptr,
    b_scale_desc_ptr,
    c_ptr,
    scatter_add_indices,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FUSE_SCATTER_ADD: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    # Triton DSL control flow is evaluated by the compiler.
    # pylint: disable=too-many-nested-blocks
    tl.static_assert(
        not (FUSE_SCATTER_ADD and USE_TMA_STORE),
        "Cannot fuse scatter add with TMA store!",
    )

    tidx = tl.program_id(0)

    dtype = TT_FP8_DTYPE

    M_end_offset = 0
    M_end_offset = M_end_offset.to(tl.int64)  # pyre-ignore  # pylint: disable=no-member
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        m_size = tl.load(m_sizes + g)

        if m_size > 0:
            M_start_offset = M_end_offset
            M_end_offset = M_start_offset + m_size
            N_start_offset = g.to(tl.int64) * N
            n_size = N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                c_desc_ptr = tl.make_tensor_descriptor(
                    c_ptr + M_start_offset * N,
                    shape=[m_size, n_size],
                    # pyre-ignore
                    strides=[n_size, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            # Move across tiles
            # pylint: disable-next=chained-comparison
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                tl.static_assert(K % BLOCK_SIZE_K == 0)
                if USE_TMA_LOAD:
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = tl._experimental_descriptor_load(
                            a_desc_ptr,
                            [m_offset, k_offset],
                            [BLOCK_SIZE_M, BLOCK_SIZE_K],
                            dtype,
                        )
                        b = tl._experimental_descriptor_load(
                            b_desc_ptr,
                            [n_offset, k_offset],
                            [BLOCK_SIZE_N, BLOCK_SIZE_K],
                            dtype,
                        )
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)
                    a_ptrs = a_desc_ptr + (M_start_offset + offs_am[:, None]) * K + offs_k[None, :]
                    b_ptrs = b_desc_ptr + (N_start_offset + offs_bn[:, None]) * K + offs_k[None, :]
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size)
                        b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size)
                        accumulator += tl.dot(a, b.T)
                        a_ptrs += BLOCK_SIZE_K
                        b_ptrs += BLOCK_SIZE_K

                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                a_scale = tl.load(
                    a_scale_ptr + M_start_offset + offs_am[:, None],
                    mask=offs_am[:, None] < m_size,
                )
                b_scale = tl.load(
                    b_scale_ptr + N_start_offset + offs_bn[None, :],
                    mask=offs_bn[None, :] < n_size,
                )
                c = accumulator.to(tl.float32) * a_scale * b_scale

                if USE_TMA_STORE:
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    # pyre-ignore
                    c_desc_ptr.store([m_offset, n_offset], c.to(c_ptr.dtype.element_ty))
                elif FUSE_SCATTER_ADD:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    mask = offs_am < m_size
                    m_offsets = tl.load(
                        scatter_add_indices + M_start_offset + offs_am,
                        mask=mask,
                        cache_modifier=".ca",
                    )
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    tl.atomic_add(
                        c_ptr + m_offsets[:, None] * N + offs_bn[None, :],
                        c.to(c_ptr.dtype.element_ty),
                        mask=(mask[:, None]) & (offs_bn[None, :] < n_size),
                        sem="relaxed",
                    )
                else:
                    tl.store(
                        c_ptr + (M_start_offset + offs_am[:, None]) * N + offs_bn[None, :],
                        c,
                        mask=(offs_am[:, None] < m_size) & (offs_bn[None, :] < n_size),
                    )
                tidx += NUM_SMS

            iterated_tiles += num_tiles


def _grouped_gemm(
    *,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: Optional[torch.Tensor],
    w_scale: Optional[torch.Tensor],
    use_fast_accum: bool,
    use_warp_specialization: bool,
    output_tensor: Optional[torch.Tensor],
    scatter_add_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    # 本套件只支持 FP8 行级量化路径：warp specialization 恒走本 kernel。
    if use_warp_specialization:
        warnings.warn(
            "Warp specialization is not supported in this fp8_rowwise suite "
            "(_fbgemm_grouped_gemm_fp8_rowwise); falling back to "
            "_fbgemm_grouped_gemm_fp8_rowwise.",
            stacklevel=2,
        )
        use_warp_specialization = False

    USE_TMA_LOAD = not torch.version.hip and not HAS_NPU
    USE_TMA_STORE = False

    # TMA 仅在 Hopper+ (compute capability >= 9.0) 上可用
    # L20/A100/V100 等旧架构不支持，强制禁用避免 KernelParamWrapper 错误
    if USE_TMA_LOAD:
        try:
            cc = torch.cuda.get_device_capability()
            if cc[0] < 9:  # 非 Hopper 架构
                USE_TMA_LOAD = False
                USE_TMA_STORE = False
        except Exception:
            USE_TMA_LOAD = False
            USE_TMA_STORE = False

    if USE_TMA_LOAD and not HAS_TMA_DESC:
        USE_TMA_LOAD = False
        warnings.warn("TMA load is disabled as there is no TMA descriptor support!", stacklevel=2)

    if USE_TMA_STORE and not HAS_TMA_DESC:
        USE_TMA_STORE = False
        warnings.warn("TMA store is disabled as there is no TMA descriptor support!", stacklevel=2)

    G = m_sizes.shape[0]

    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    M, K = x.shape
    N = w.shape[0] // G
    assert K == w.shape[1]

    if K % 8 != 0 or N % 8 != 0:
        # API 侧已保证 K/N 是 8 的倍数（FP8 路径前提）；此处仅防御性禁用 TMA。
        USE_TMA_LOAD = False
        USE_TMA_STORE = False
        warnings.warn(
            f"TMA load is disabled since K or N is not a multiple of 8: {K=}, {N=}.",
            stacklevel=2,
        )
        assert output_tensor is None, (
            f"Fused scatter add has large rounding error when K or N is not a multiple of 8: {K=}, {N=}."
        )

    if output_tensor is None:
        FUSE_SCATTER_ADD = False
        assert scatter_add_indices is None
        y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    else:
        FUSE_SCATTER_ADD = True
        assert scatter_add_indices is not None
        assert scatter_add_indices.is_contiguous()
        assert scatter_add_indices.shape == (M,)
        y = output_tensor
    if M == 0 or N == 0:
        return y

    # Get number of SMs based on device type
    if HAS_NPU:
        NUM_SMS = torch.npu.device_count()  # Fallback, may not be accurate
        try:
            props = torch.npu.get_device_properties(x.device)
            NUM_SMS = props.multi_processor_count // 2  # 28 AI Cores for Ascend 950
        except Exception:
            NUM_SMS = 28  # Default for Ascend 950
    else:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    desc_helper = None
    desc_x = x
    desc_w = w
    desc_ws = w_scale

    if USE_TMA_LOAD:
        desc_helper = TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("x")
        desc_helper.init_tma_descriptor("w")
        desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
        desc_w = desc_helper.get_tma_descriptor_kernel_param("w")

    if USE_TMA_STORE:

        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    # === 硬编码固定config ===
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    NUM_CONSUMER_GROUPS = 1
    # =====================================

    def grid(META):
        if USE_TMA_LOAD:
            nonlocal desc_helper
            desc_helper.fill_2d_tma_descriptor(
                "x",
                x.data_ptr(),
                M,
                K,
                META["BLOCK_SIZE_M"] // META["NUM_CONSUMER_GROUPS"],
                META["BLOCK_SIZE_K"],
                x.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "w",
                w.data_ptr(),
                N * G,
                K,
                META["BLOCK_SIZE_N"],
                META["BLOCK_SIZE_K"],
                w.element_size(),
            )

        return (NUM_SMS,)

    M_BUCKET_CAP = 16384
    M_BUCKET = min(triton.next_power_of_2(M), M_BUCKET_CAP)

    # FP8 行级量化路径：必须有行/列尺度（API 恒传入）。
    assert x_scale is not None and w_scale is not None
    assert x_scale.is_contiguous()
    assert w_scale.is_contiguous()
    fn = _fbgemm_grouped_gemm_fp8_rowwise
    args = (
        desc_x,
        x_scale,
        desc_w,
        w_scale,
        desc_ws,
        y,
        scatter_add_indices,
        m_sizes,
        G,
        M_BUCKET,
        N,
        K,
        NUM_SMS,
        FUSE_SCATTER_ADD,
        USE_TMA_LOAD,
        USE_TMA_STORE,
        use_fast_accum,
    )
    args += (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_CONSUMER_GROUPS)
    fn[grid](*args)

    return y


def grouped_gemm_fp8_rowwise(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    use_fast_accum: bool = True,
    *,
    _use_warp_specialization: bool = True,
    _output_tensor: Optional[torch.Tensor] = None,
    _scatter_add_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _grouped_gemm(
        x=x,
        w=w,
        m_sizes=m_sizes,
        x_scale=x_scale,
        w_scale=w_scale,
        use_fast_accum=use_fast_accum,
        use_warp_specialization=_use_warp_specialization,
        output_tensor=_output_tensor,
        scatter_add_indices=_scatter_add_indices,
    )
