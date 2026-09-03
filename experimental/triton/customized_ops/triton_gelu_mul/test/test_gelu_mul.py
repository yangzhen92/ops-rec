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

# pylint: disable=redefined-outer-name, redefined-builtin

"""
Pytest for triton_gelu_mul: Triton NPU vs PyTorch CPU reference.
The reference (标杆) is generated on CPU via torch.nn.functional.gelu.
"""

import pytest
import torch

from triton_gelu_mul import triton_gelu_mul

# ============================================================
# CPU reference (标杆)
# ============================================================


def gelu_mul_cpu_ref(input: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    d = input.shape[-1] // 2
    x1 = input[..., :d].float()
    x2 = input[..., d:].float()
    x1_gelu = torch.nn.functional.gelu(x1, approximate=approximate)
    return (x1_gelu * x2).to(input.dtype)


@pytest.fixture(scope="module")
def device():
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")
    torch.npu.set_device(2)
    return torch.device("npu:2")


def _build_input(rows, d, dtype, device):
    # Single input [rows, 2d]; last dim 2d=3072 -> d=1536.
    torch.manual_seed(42)
    input_cpu = (torch.randn((rows, d), device="cpu", dtype=torch.float32) * 3.0).to(dtype)
    input_dev = input_cpu.to(device)
    return input_cpu, input_dev


# ============================================================
# Test cases: (rows, d, approximate)  ->  8 shapes x 2 modes = 16 cases
# ============================================================


@pytest.mark.parametrize(
    "rows, d, approximate",
    [
        # bs1
        (1000, 3072, "none"),
        (3072, 3072, "none"),
        (1000, 3072, "tanh"),
        (3072, 3072, "tanh"),
        # bs4
        (4000, 3072, "none"),
        (12288, 3072, "none"),
        (4000, 3072, "tanh"),
        (12288, 3072, "tanh"),
        # bs8
        (8000, 3072, "none"),
        (24576, 3072, "none"),
        (8000, 3072, "tanh"),
        (24576, 3072, "tanh"),
        # bs16
        (16000, 3072, "none"),
        (49152, 3072, "none"),
        (16000, 3072, "tanh"),
        (49152, 3072, "tanh"),
    ],
)
def test_gelu_mul_case(rows, d, approximate, device):
    """Compare Triton NPU vs CPU reference for a single input shape."""
    dtype = torch.bfloat16

    input_cpu, input_dev = _build_input(rows, d, dtype, device)
    d_half = d // 2
    out_shape = (rows, d_half)

    # ---- CPU reference (标杆) ----
    ref = gelu_mul_cpu_ref(input_cpu, approximate=approximate)

    # ---- Triton ----
    tri_out = triton_gelu_mul(input_dev, approximate=approximate)

    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    tri_cpu = tri_out.cpu()

    # ---- Shape check ----
    assert tuple(tri_cpu.shape) == out_shape, f"shape mismatch: expected {out_shape}, got {tuple(tri_cpu.shape)}"

    # ---- Compare ----
    ref_f = ref.float()
    tri_f = tri_cpu.float()

    max_diff = (ref_f - tri_f).abs().max().item()
    mean_diff = (ref_f - tri_f).abs().mean().item()

    print(f"  [rows={rows}, d={d}, approx={approximate}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    torch.testing.assert_close(
        tri_f,
        ref_f,
        atol=8e-2,
        rtol=1e-1,
        msg=f"Triton vs CPU mismatch (approx={approximate}): max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}",
    )
