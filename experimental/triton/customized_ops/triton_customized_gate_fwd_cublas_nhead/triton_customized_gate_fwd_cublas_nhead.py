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
import triton.language as tl
from triton.runtime import driver


def _get_ncores() -> int:
    """Query the number of AI cores from NPU hardware at runtime."""
    device = torch.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    num_vectorcore = props["num_vectorcore"]
    return num_vectorcore


# ========== Autotune config（BLOCK_B + multibuffer） ==========
def get_autotune_config():
    return [
        triton.Config({"BLOCK_B": 1, "multibuffer": True}),
        triton.Config({"BLOCK_B": 2, "multibuffer": True}),
        triton.Config({"BLOCK_B": 4, "multibuffer": True}),
        triton.Config({"BLOCK_B": 8, "multibuffer": True}),
        triton.Config({"BLOCK_B": 16, "multibuffer": True}),
        triton.Config({"BLOCK_B": 32, "multibuffer": True}),
    ]


# ============================================================
# Fused Kernel: clamp + SiLU + sum over heads + gate multiply + RMSNorm
#   Input:  proj_all   (H, B, Din)  — all head projections (bf16/f16)
#           proj_bias  (H, Din)     — per-head bias (f32)
#           input      (B, Din)     — original input
#           rms_weight (Din,)       — RMSNorm weight
#   Output: output     (B, Din)     — RMSNorm(gated)
# ============================================================
@triton.autotune(
    configs=get_autotune_config(),
    key=['B', 'Din', 'H'],
)
@triton.jit
def _fused_gate_rmsnorm_kernel(
    proj_all_ptr,
    input_ptr,
    rms_weight_ptr,
    out_ptr,
    B,
    Din,
    H,
    stride_pa_h,
    stride_pa_b,
    stride_pa_din,
    stride_in_b,
    stride_in_din,
    stride_out_b,
    stride_out_din,
    eps: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_DIN: tl.constexpr,
    NUM_CORES: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)

    tiles_b = tl.cdiv(B, BLOCK_B)
    tiles_per_core = tl.cdiv(tiles_b, NUM_CORES)
    start_tid_b = pid * tiles_per_core
    end_tid_b = tl.minimum(start_tid_b + tiles_per_core, tiles_b)

    for tid_b in range(start_tid_b, end_tid_b):
        offs_b = tid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = offs_b < B

        offs_din = tl.arange(0, BLOCK_DIN)
        din_mask = offs_din < Din
        load_mask = b_mask[:, None] & din_mask[None, :]

        # Step 1: 对每个 head 执行 clamp + SiLU 激活并累加，再乘以门控输入
        # Note: bias 已通过前端的 baddbmm 注入 proj_all，kernel 中不再额外加 bias。
        acc = tl.zeros([BLOCK_B, BLOCK_DIN], dtype=tl.float32)
        for h in range(H):
            proj_h = tl.load(
                proj_all_ptr + h * stride_pa_h + offs_b[:, None] * stride_pa_b + offs_din[None, :] * stride_pa_din,
                mask=load_mask,
                other=0.0,
            )
            proj_h = tl.minimum(tl.maximum(proj_h, -10.0), 10.0)
            proj_h = proj_h * tl.sigmoid(proj_h)
            acc = acc + proj_h

        x_tile = tl.load(
            input_ptr + offs_b[:, None] * stride_in_b + offs_din[None, :] * stride_in_din,
            mask=load_mask,
            other=0.0,
        )
        gated = acc * x_tile

        # Step 2: 计算 RMS 归一化因子（归约前升精度到 FP32 保证数值稳定）
        squared_sum = tl.sum(gated.to(tl.float32) * gated.to(tl.float32), axis=1)
        rms = tl.sqrt(squared_sum / Din + eps)

        # Step 3: 直接对寄存器中的 gated 施加 RMSNorm，一次写回最终结果
        w = tl.load(rms_weight_ptr + offs_din, mask=din_mask, other=0.0)
        out = gated / rms[:, None] * w[None, :]
        tl.store(
            out_ptr + offs_b[:, None] * stride_out_b + offs_din[None, :] * stride_out_din,
            out.to(OUT_DTYPE),
            mask=load_mask,
        )


# pylint: disable=redefined-builtin
def customized_gate_fwd_cublas_nhead(
    input: torch.Tensor,  # (B, in_size)  bf16
    gate_layer_weight: torch.Tensor,  # (E, in_size)  bf16, E = H * Dh
    gate_layer_bias: torch.Tensor,  # (E,)  bf16/f32
    proj_weights: torch.Tensor,  # (H, in_size, head_size)  bf16  ← 与 CUDA 接口一致
    proj_biases: torch.Tensor,  # (H, in_size)  bf16/f32
    rms_norm_weight: torch.Tensor,  # (in_size,)  bf16/f32
    eps: float = 1e-5,
) -> torch.Tensor:
    B, Din = input.shape
    E = gate_layer_weight.shape[0]
    H = proj_weights.shape[0]
    Dh = proj_weights.shape[2]

    assert E == H * Dh, f"embedding_dim ({E}) != num_heads ({H}) * head_size ({Dh})"
    assert proj_weights.shape[1] == Din, "proj_weights dim 1 must match in_size"
    assert proj_biases.shape == (H, Din), "proj_biases shape mismatch"
    assert gate_layer_weight.shape == (E, Din), "gate_layer_weight shape mismatch"
    assert gate_layer_bias.shape == (E,), "gate_layer_bias shape mismatch"

    dtype = input.dtype
    out_dtype = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16
    device = input.device

    # ---- Kernel 1 (torch.addmm): X @ W_gate^T + b_gate → (B, E) ----
    gate_out = torch.addmm(gate_layer_bias, input, gate_layer_weight.t())

    # ---- Kernel 2 (torch.baddbmm): 用 CUDA 一致的 baddbmm 将 bias 注入 proj_all ----
    # proj_weights: (H, in_size, head_size) → transpose → (H, head_size, in_size)
    # heads: (H, B, head_size)
    # proj_all = heads @ proj_weights^T + bias → (H, B, in_size)
    heads = gate_out.reshape(B, H, Dh).permute(1, 0, 2).contiguous()
    proj_all = torch.baddbmm(
        proj_biases.unsqueeze(1),
        heads,
        proj_weights.transpose(-2, -1),
    )

    BLOCK_DIN = Din

    # ---- Fused Kernel: act + gate + RMSNorm (single kernel) ----
    num_cores = _get_ncores()
    output = torch.empty((B, Din), device=device, dtype=dtype)
    grid = (num_cores,)
    _fused_gate_rmsnorm_kernel[grid](
        proj_all,
        input,
        rms_norm_weight,
        output,
        B,
        Din,
        H,
        proj_all.stride(0),
        proj_all.stride(1),
        proj_all.stride(2),
        input.stride(0),
        input.stride(1),
        output.stride(0),
        output.stride(1),
        eps=eps,
        BLOCK_DIN=BLOCK_DIN,
        NUM_CORES=num_cores,
        OUT_DTYPE=out_dtype,
        hfusion_enable_multiple_consumer_fusion=True,
    )

    return output
