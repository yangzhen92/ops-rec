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

"""
triton_gelu_mul 算子的 ATK 自定义 API 执行文件

双注册：
  - @register("triton_gelu_mul") — 走 Triton 实现（--backend triton 时使用）
  - @register("npu_gelu_mul")   — 走 AscendC / CPU 实现（--backend npu 或 cpu 时使用）

双标杆模式：
  npu 后端 → AscendC npu_geglu（标杆1）
  cpu 后端 → PyTorch 标准 GELU（真值标杆2）

功能：输入 [..., 2d] → GELU(x1) * x2, 其中 x1 = input[..., :d], x2 = input[..., d:]
"""
import torch
import torch_npu
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

# Triton 实现
from triton_gelu_mul import triton_gelu_mul

def get_device():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except:
        pass
    return 'cpu'


class GeluMulBase(BaseApi):
    """共享的 init_by_input_data 逻辑"""

    def init_by_input_data(self, input_data: InputDataset):
        # 不自动将 tensor 搬到 device，让 __call__ 按需处理
        input_tensor = input_data.kwargs.get("input", None)
        approximate = input_data.kwargs.get("approximate", "none")

        if input_tensor is None:
            raise ValueError("input not found in input_data")

        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        # else: 保持 ATK 框架分配的 device 不动

        last_dim = input_tensor.shape[-1]
        if last_dim % 2 != 0:
            raise ValueError(f"last dim must be even, got {last_dim}")

        if approximate is None or approximate == "null" or approximate == "none":
            approximate = "none"
        elif approximate not in ("none", "tanh"):
            approximate = "none"

        self._input_tensor = input_tensor
        self._approximate = approximate
        d = input_tensor.shape[-1] // 2
        self._expected_shape = input_tensor.shape[:-1] + (d,)


@register("triton_gelu_mul")
class TritonGeluMulApi(GeluMulBase):
    """Triton 算子实现路径（--backend triton 时使用）"""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        device = get_device()
        if device == 'npu':
            input_tensor = self._input_tensor.npu()
            torch.npu.synchronize()
        elif device == 'cuda':
            input_tensor = self._input_tensor.cuda()
            torch.cuda.synchronize()
        else:
            input_tensor = self._input_tensor

        output = triton_gelu_mul(
            input_tensor,
            approximate=self._approximate,
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        return output


@register("npu_gelu_mul")
class NpuGeluMulApi(GeluMulBase):
    """
    npu/cpu 统一实现路径（--backend npu 或 --backend cpu 时使用）

    双标杆模式：
      - npu 后端：AscendC npu_geglu（标杆1）
      - cpu 后端：PyTorch 标准 GELU（真值标杆2）
    """

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_tensor = self._input_tensor
        approximate = self._approximate

        if input_tensor.device.type == 'npu':
            torch.npu.synchronize()
            approximate_val = 1 if approximate == "tanh" else 0
            output, _ = torch_npu.npu_geglu(
                input_tensor, dim=-1, approximate=approximate_val,
            )
            torch.npu.synchronize()
        else:
            # CPU 后端：用 PyTorch 标准 GELU 作为真值标杆
            d = input_tensor.shape[-1] // 2
            x1 = input_tensor[..., :d]
            x2 = input_tensor[..., d:]
            output = torch.nn.functional.gelu(x1, approximate=approximate) * x2

        return output