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

# pylint: disable=redefined-outer-name, redefined-builtin, attribute-defined-outside-init

"""
Pytest for CustomizedGateFwdCublasNhead: Triton NPU vs PyTorch reference.
"""

import pytest
import torch
from typing import List, Optional

from ops_rec.experimental.triton.triton_customized_gate_fwd_cublas_nhead import customized_gate_fwd_cublas_nhead

# ============================================================
# PyTorch reference implementation (from mlp.py, simplified for NPU)
# ============================================================


def _init_mlp_weights_optional_bias(module):
    """Initialize MLP weights."""
    if hasattr(module, 'weight'):
        torch.nn.init.xavier_uniform_(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.zeros_(module.bias)


class MLP(torch.nn.Module):
    """MLP module wrapper."""

    def __init__(
        self,
        in_size: int,
        layer_sizes: List[int],
        activation: str = "relu",
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if activation == "relu":
            activation_fn = torch.nn.ReLU
        elif activation == "gelu":
            activation_fn = torch.nn.GELU
        elif activation == "silu":
            activation_fn = torch.nn.SiLU
        elif activation == "identity":
            activation_fn = torch.nn.Identity
        else:
            raise ValueError(f"Activation function {activation} not supported")

        layers = []
        for i in range(len(layer_sizes)):
            layers.extend(
                [
                    torch.nn.Linear(
                        layer_sizes[i - 1] if i > 0 else in_size,
                        layer_sizes[i],
                        bias=bias,
                        device=device,
                        dtype=dtype,
                    ),
                    activation_fn() if i < len(layer_sizes) - 1 else torch.nn.Identity(),
                ]
            )

        self._mlp = torch.nn.Sequential(*layers)
        self._mlp.apply(_init_mlp_weights_optional_bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 2, "Tensor must be 2-dimensional"
        return self._mlp(input)


class CustomizedGate(torch.nn.Module):
    """Customized Gate module matching mlp.py implementation."""

    def __init__(
        self,
        in_size: int,
        layer_sizes: List[int],
        input_gate_head: int,
        device: Optional[torch.device] = None,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self._in_size = in_size
        self._input_gate_head = input_gate_head
        self._eps = eps

        # Create gate layer
        self._gate_layer = MLP(
            in_size,
            layer_sizes=layer_sizes,
            bias=True,
            activation="silu",
            device=device,
            dtype=dtype,
        )

        # Calculate the output size of gate_layer (last element in layer_sizes)
        gate_output_size = layer_sizes[-1] if layer_sizes else in_size
        # Calculate the size of each head
        head_size = gate_output_size // input_gate_head

        # Create projection layers for each head
        self._gate_proj_layers = torch.nn.ModuleList()
        for _ in range(input_gate_head):
            gate_layer_proj = MLP(
                head_size,
                layer_sizes=[in_size],
                bias=True,
                activation="identity",
                device=device,
                dtype=dtype,
            )
            self._gate_proj_layers.append(gate_layer_proj)

        # Create RMS norm module
        if device is None:
            device = torch.device("cpu")
        self._rms_norm = torch.nn.RMSNorm(
            normalized_shape=in_size,
            eps=eps,
        ).to(device=device, dtype=dtype)

        self._fused_proj_ready = False

    def _build_fused_proj(self):
        """Build fused projection weight from individual proj layers for grouped bmm."""
        with torch.no_grad():
            w_list, b_list = [], []
            for proj in self._gate_proj_layers:
                linear = proj._mlp[0]
                w_list.append(linear.weight.T)  # [head_size, in_size]
                b_list.append(linear.bias)
            # [num_heads, head_size, in_size]
            self._fused_proj_w = torch.stack(w_list).contiguous()
            self._fused_proj_b = torch.stack(b_list).unsqueeze(1).contiguous()
        self._fused_proj_ready = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._fused_proj_ready:
            self._build_fused_proj()

        input_gate = self._gate_layer(input)  # [B, gate_out]

        # Split into heads and reshape for grouped bmm
        H = self._input_gate_head
        head_size = input_gate.shape[-1] // H
        # [B, gate_out] → [B, H, head_size] → [H, B, head_size]
        heads = input_gate.reshape(-1, H, head_size).permute(1, 0, 2)

        # Grouped bmm: [H, B, head_size] × [H, head_size, in_size] → [H, B, in_size]
        proj = torch.bmm(heads, self._fused_proj_w) + self._fused_proj_b
        proj = torch.clamp(proj, -10, 10)
        proj = proj * torch.sigmoid(proj)  # Swish

        # Sum over heads: [H, B, in_size] → [B, in_size]
        gate_output = proj.sum(dim=0)
        gate_output = input * gate_output
        gate_output = self._rms_norm(gate_output)
        return gate_output


@pytest.fixture(scope="module")
def device():
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")
    torch.npu.set_device(2)
    return torch.device("npu:2")


def _make_tensors(B, in_size, embedding_dim, num_heads, dtype=torch.bfloat16):
    """Create random tensors for the operator with CustomizedGate reference module."""
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")
    torch.npu.set_device(2)
    DEVICE = torch.device("npu:2")

    torch.manual_seed(42)

    # 创建 CustomizedGate 模型 (使用与 mlp.py 中相同的结构)
    # gate MLP 使用 [embedding_dim] 作为隐藏层
    model = CustomizedGate(
        in_size=in_size,
        layer_sizes=[embedding_dim],
        input_gate_head=num_heads,
        device=DEVICE,
        eps=1e-5,
        dtype=dtype,
    )

    # 构建投影权重
    model._build_fused_proj()

    # 在 CPU 上创建输入 tensor，然后移到 NPU
    x_cpu = torch.randn((B, in_size), device="cpu", dtype=dtype)
    x = x_cpu.to(DEVICE)

    # 提取 NPU 上的权重
    gate_w = model._gate_layer._mlp[0].weight.data  # (E, in_size)
    gate_b = model._gate_layer._mlp[0].bias.data  # (E,)
    proj_w = model._fused_proj_w.data  # (H, head_size, in_size)
    proj_b = model._fused_proj_b.data  # (H, 1, in_size)
    rms_w = model._rms_norm.weight.data  # (in_size,)

    # proj_b 形状调整: (H, 1, in_size) -> (H, in_size)
    proj_b = proj_b.squeeze(1)

    # kernel 期望 proj_w 形状为 (H, in_size, head_size)，转置 dim1/dim2
    proj_w = proj_w.transpose(-2, -1).contiguous()

    # 获取 rms eps
    rms_eps = model._eps

    return x, gate_w, gate_b, proj_w, proj_b, rms_w, model, rms_eps


# ============================================================
# Customer cases (matching gr_inference test_customized_gate_nhead.py)
#   his_gate:  in_size=256,  heads=4, embedding_dim=2048
#   req_gate:  in_size=6519, heads=4, embedding_dim=2048
#   H=2 sanity: in_size=256, heads=2, embedding_dim=2048
# ============================================================


@pytest.mark.parametrize(
    "B, in_size, embedding_dim, num_heads",
    [
        # bs1: his_gate  (input [1000, 256],  gate E=2048 H=4 Dh=512)
        (1000, 256, 2048, 4),
        # bs1: req_gate  (input [512, 6519],  gate E=2048 H=4 Dh=512)
        (512, 6519, 2048, 4),
        # bs4: his_gate  (input [4000, 256],  gate E=2048 H=4 Dh=512)
        (4000, 256, 2048, 4),
        # bs4: req_gate  (input [2048, 6519], gate E=2048 H=4 Dh=512)
        (2048, 6519, 2048, 4),
        # bs8: his_gate  (input [8000, 256],  gate E=2048 H=4 Dh=512)
        (8000, 256, 2048, 4),
        # bs8: req_gate  (input [4096, 6519], gate E=2048 H=4 Dh=512)
        (4096, 6519, 2048, 4),
        # bs16: his_gate (input [16000, 256], gate E=2048 H=4 Dh=512)
        (16000, 256, 2048, 4),
        # bs16: req_gate (input [8192, 6519],  gate E=2048 H=4 Dh=512)
        (8192, 6519, 2048, 4),
    ],
)
def test_customer_case(B, in_size, embedding_dim, num_heads):
    """Customer cases matching gr_inference test_customized_gate_nhead.py shapes."""
    dtype = torch.bfloat16
    x, gate_w, gate_b, proj_w, proj_b, rms_w, model, rms_eps = _make_tensors(
        B, in_size, embedding_dim, num_heads, dtype=dtype
    )

    # ---- Reference ----
    ref_out = model(x)

    # ---- Triton ----
    tri_out = customized_gate_fwd_cublas_nhead(
        x,
        gate_w,
        gate_b,
        proj_w,
        proj_b,
        rms_w,
        eps=rms_eps,
    )

    # ---- Compare ----
    max_diff = (ref_out.float() - tri_out.float()).abs().max().item()
    mean_diff = (ref_out.float() - tri_out.float()).abs().mean().item()
    print(
        f"  [B={B}, Din={in_size}, E={embedding_dim}, H={num_heads}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    )

    torch.testing.assert_close(
        tri_out.float(),
        ref_out.float(),
        atol=8e-2,
        rtol=1e-1,
        msg=f"Triton vs PyTorch mismatch: max_diff={max_diff:.6f}",
    )
