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
add_position_embeddings 算子 ATK 自定义 API 执行文件

直接调用 _add_position_embeddings_kernel（含 @triton_autotune 包装）。

值生成策略（确定性，不依赖随机数）：
- seq_offsets: 单调递增整数，首 0 尾 sum(N_i)
- high_inds: 非负整数，每个 batch 的 max dense lookup index
- ind_offsets: 存在时为非负整数（可选参数）
"""
import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from triton_add_position_embeddings_kernel import _add_position_embeddings_kernel


def get_device():
    """获取可用的设备，优先 NPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


def _generate_offsets_deterministic(b_size: int, total_n: int) -> list:
    """确定性生成 seq_offsets: 单调递增，首 0 尾 total_n"""
    if b_size <= 0:
        return [0, total_n]
    min_per_batch = 1
    base = (total_n - b_size * min_per_batch) // b_size if b_size > 0 else 0
    extra = max(0, (total_n - b_size * min_per_batch) - base * b_size)

    offsets = [0]
    current = 0
    for i in range(b_size):
        seg_len = min_per_batch + base + (1 if i < extra else 0)
        current += seg_len
        offsets.append(current)

    if offsets[-1] != total_n:
        offsets[-1] = total_n
    return offsets


def _generate_high_inds_deterministic(b_size: int, offsets: list, K: int) -> list:
    """确定性生成 high_inds，不超过 K-1"""
    high_inds = []
    max_allowed = K - 1
    ratios = [0.3, 0.5, 0.7, 0.9, 1.0]
    for i in range(b_size):
        if max_allowed <= 0:
            high_inds.append(0)
            continue
        ratio = ratios[i % len(ratios)]
        val = max(0, int(max_allowed * ratio))
        val = min(val, max_allowed)
        high_inds.append(val)
    return high_inds


@register("add_position_embeddings")
class AddPositionEmbeddingsApi(BaseApi):
    """add_position_embeddings 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理操作（不计时）"""
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ---- 1. 读取输入参数 ----
        jagged_input = input_data.kwargs.get("Jagged", None)
        seq_offsets_input = input_data.kwargs.get("seq_offsets", None)
        high_inds_input = input_data.kwargs.get("high_inds", None)
        dense_input = input_data.kwargs.get("Dense", None)
        scale_input = input_data.kwargs.get("scale", None)
        ind_offsets_input = input_data.kwargs.get("ind_offsets", None)
        autotune_max_seq_len = input_data.kwargs.get("AUTOTUNE_MAX_SEQ_LEN", 64)
        scale_jagged = input_data.kwargs.get("SCALE_JAGGED", False)

        if jagged_input is None or dense_input is None:
            raise ValueError("Missing required input: Jagged and Dense are required")

        # ---- 2. 将 tensor 输入转为 torch.Tensor 并移至目标设备 ----
        if not isinstance(jagged_input, torch.Tensor):
            jagged = torch.tensor(jagged_input, dtype=torch.float32, device=device)
        else:
            jagged = jagged_input.to(device)

        if not isinstance(dense_input, torch.Tensor):
            dense = torch.tensor(dense_input, dtype=jagged.dtype, device=device)
        else:
            dense = dense_input.to(device)

        # ---- 3. 从 shapes 推断 batch 维度和总长度 ----
        total_n = jagged.shape[0]
        d_dim = jagged.shape[1]

        # 从 seq_offsets 的 shape 推断 B
        if seq_offsets_input is not None:
            if isinstance(seq_offsets_input, torch.Tensor):
                b_size = seq_offsets_input.shape[0] - 1
                seq_offsets_input = seq_offsets_input.to(device).to(torch.int32)
            elif hasattr(seq_offsets_input, 'shape'):
                b_size = seq_offsets_input.shape[0] - 1
            else:
                b_size = 2
        else:
            b_size = 2

        b_size = max(1, b_size)

        # ---- 4. 确定性生成 seq_offsets ----
        offsets_values = _generate_offsets_deterministic(b_size, total_n)
        seq_offsets = torch.tensor(offsets_values, dtype=torch.int32, device=device)
        total_n = offsets_values[-1]

        if jagged.shape[0] != total_n:
            jagged = jagged[:total_n]

        # ---- 5. 确定性生成 high_inds ----
        K = dense.shape[0]
        high_inds_values = _generate_high_inds_deterministic(b_size, offsets_values, K)
        if high_inds_input is not None:
            if isinstance(high_inds_input, torch.Tensor):
                high_inds = torch.tensor(high_inds_values, dtype=high_inds_input.dtype, device=device).to(torch.int32)
            else:
                high_inds = torch.tensor(high_inds_values, dtype=torch.int32, device=device)
        else:
            high_inds = torch.tensor(high_inds_values, dtype=torch.int32, device=device)

        # ---- 6. 处理可选参数 ----
        if scale_input is not None:
            if isinstance(scale_input, torch.Tensor):
                scale = scale_input.item()
            else:
                scale = float(scale_input)
        else:
            scale = 1.0

        # ind_offsets：kernel 中 `if ind_offsets is not None` 需要非 None 才能通过编译
        if ind_offsets_input is not None:
            if isinstance(ind_offsets_input, torch.Tensor):
                ind_tensor = ind_offsets_input.to(device).to(torch.int32)
            else:
                ind_tensor = torch.tensor(ind_offsets_input, dtype=torch.int32, device=device)

            if ind_tensor.shape[0] != b_size:
                ind_tensor = torch.zeros(b_size, dtype=torch.int32, device=device)
            ind_offsets = ind_tensor
        else:
            # 用全 0 张量替代，逻辑等价于不加偏移
            ind_offsets = torch.zeros(b_size, dtype=torch.int32, device=device)

        # AUTOTUNE_MAX_SEQ_LEN
        if isinstance(autotune_max_seq_len, torch.Tensor):
            autotune_max_seq_len = int(autotune_max_seq_len.item())
        else:
            autotune_max_seq_len = int(autotune_max_seq_len)
        actual_max_seq_len = max(offsets_values[i+1] - offsets_values[i] for i in range(b_size))
        autotune_max_seq_len = max(autotune_max_seq_len, actual_max_seq_len)

        # SCALE_JAGGED
        if isinstance(scale_jagged, torch.Tensor):
            scale_jagged = bool(scale_jagged.item())
        elif isinstance(scale_jagged, str):
            scale_jagged = scale_jagged.lower() == "true"
        else:
            scale_jagged = bool(scale_jagged)

        # ---- 7. 确保 Dense 的 D 维度与 Jagged 一致 ----
        if dense.shape[1] != d_dim:
            if dense.shape[1] > d_dim:
                dense = dense[:, :d_dim]
            else:
                pad = torch.zeros(dense.shape[0], d_dim - dense.shape[1], dtype=dense.dtype, device=device)
                dense = torch.cat([dense, pad], dim=1)

        # ---- 8. 计算 strides ----
        stride_jn = jagged.stride(0)
        stride_dk = dense.stride(0)
        stride_on = stride_jn  # Out 与 Jagged 同形状

        # ---- 9. 创建输出 tensor ----
        out = torch.empty_like(jagged, device=device)

        # ---- 10. 硬件核数 ----
        try:
            device_props = triton.runtime.driver.active.utils.get_device_properties(device)
            ncores = int(device_props.get("num_vectorcore", 56))
        except Exception:
            ncores = 56
        ncores = min(b_size, ncores)

        # ---- 11. BLOCK_D = D（不再对齐） ----
        block_d = d_dim

        # ---- 12. 保存实例变量 ----
        self._device = device
        self._jagged = jagged
        self._seq_offsets = seq_offsets
        self._high_inds = high_inds
        self._dense = dense
        self._out = out
        self._scale = scale
        self._ind_offsets = ind_offsets
        self._d_dim = d_dim
        self._stride_jn = stride_jn
        self._stride_dk = stride_dk
        self._stride_on = stride_on
        self._scale_jagged = scale_jagged
        self._block_d = block_d
        self._total_n = total_n
        self._ncores = ncores

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行 kernel（计时）"""
        if self._device == 'npu':
            torch.npu.synchronize()
        elif self._device == 'cuda':
            torch.cuda.synchronize()

        grid = (self._ncores,)

        # 调用正确的 kernel 函数名：_add_position_embeddings_kernel
        # BLOCK_N 不传，由 autotune 自动选择
        _add_position_embeddings_kernel[grid](
            self._jagged,
            self._seq_offsets,
            self._high_inds,
            self._dense,
            self._out,
            self._total_n,   # B（总行数，用于按 core 划分）
            self._d_dim,     # D
            self._ncores,    # ncores
            self._scale,     # scale
            self._ind_offsets,
            self._stride_jn,
            self._stride_dk,
            self._stride_on,
            SCALE_JAGGED=self._scale_jagged,
            BLOCK_D=self._block_d,   # 固定为 D
            hfusion_enable_multiple_consumer_fusion=True,
        )

        if self._device == 'npu':
            torch.npu.synchronize()
        elif self._device == 'cuda':
            torch.cuda.synchronize()

        return self._out