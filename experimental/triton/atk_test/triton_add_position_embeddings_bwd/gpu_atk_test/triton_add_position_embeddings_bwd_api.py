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

# add_position_embeddings_bwd_api.py
"""
_add_position_embeddings_bwd_kernel 的 ATK 自定义 API 执行文件
"""

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from triton_add_position_embeddings_bwd_kernel import _add_position_embeddings_bwd_kernel


def get_device():
    """获取可用的设备，优先NPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except:  # noqa: E722
        pass
    return 'cpu'


@register("add_position_embeddings_bwd")
class AddPositionEmbeddingsBwdApi(BaseApi):
    """_add_position_embeddings_bwd_kernel 算子的 API 封装"""

    # pylint: disable=attribute-defined-outside-init
    def init_by_input_data(self, input_data: InputDataset):
        """预处理（不计时）"""
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # === 获取输入参数 ===
        Jagged_input = input_data.kwargs.get("Jagged", None)
        seq_offsets_input = input_data.kwargs.get("seq_offsets", None)
        high_inds_input = input_data.kwargs.get("high_inds", None)
        DenseOut_input = input_data.kwargs.get("DenseOut", None)
        scale_input = input_data.kwargs.get("scale", 1.0)

        if Jagged_input is None:
            raise ValueError("Jagged is required")
        if seq_offsets_input is None:
            raise ValueError("seq_offsets is required")
        if high_inds_input is None:
            raise ValueError("high_inds is required")
        if DenseOut_input is None:
            raise ValueError("DenseOut is required")

        # === 转换为tensor ===
        if not isinstance(Jagged_input, torch.Tensor):
            Jagged = torch.tensor(Jagged_input, dtype=torch.float32, device=device)
        else:
            Jagged = Jagged_input.to(device)

        if not isinstance(seq_offsets_input, torch.Tensor):
            seq_offsets = torch.tensor(seq_offsets_input, dtype=torch.int32, device=device)
        else:
            seq_offsets = seq_offsets_input.to(device)

        if not isinstance(high_inds_input, torch.Tensor):
            high_inds = torch.tensor(high_inds_input, dtype=torch.int32, device=device)
        else:
            high_inds = high_inds_input.to(device)

        # scale
        scale = float(scale_input)

        # === 推导参数 ===
        N, D = Jagged.shape
        B = seq_offsets.shape[0] - 1

        # 验证B
        assert B > 0, f"B must be > 0, got {B}"
        assert high_inds.shape[0] == B, f"high_inds.shape[0] {high_inds.shape[0]} != B {B}"

        # === 生成seq_offsets的值（确定性算法） ===
        # 确保 seq_offsets 从0到N，单调递增
        seq_offsets = self._generate_seq_offsets(seq_offsets, N, B, device)

        # === 生成high_inds的值（确定性算法） ===
        # high_inds[i] < seq_offsets[i+1] - seq_offsets[i]
        high_inds = self._generate_high_inds(high_inds, seq_offsets, B, device)

        # === 创建DenseOut（预分配输出） ===
        # DenseOut shape: [max_high_ind+1, D]
        max_high_ind = int(high_inds.max().item())
        DenseOut = torch.zeros((max_high_ind + 1, D), dtype=Jagged.dtype, device=device)

        # === 确定grid大小 ===
        grid = (max_high_ind + 1,)

        # === strides ===
        stride_jn = Jagged.stride(0)
        stride_jon = stride_jn  # JaggedOut stride与Jagged相同

        # DenseOut stride: [D, 1]
        stride_don = DenseOut.stride(0)

        # === BLOCK_D ===
        # BLOCK_D 必须是 2 的幂且 >= D
        BLOCK_D = 1
        while BLOCK_D < D:
            BLOCK_D <<= 1

        # === SCALE_JAGGED: 当scale != 1.0时为True ===
        SCALE_JAGGED = scale != 1.0

        # === 保存实例变量 ===
        self._device = device
        self._grid = grid
        self._Jagged = Jagged
        self._seq_offsets = seq_offsets
        self._high_inds = high_inds
        self._DenseOut = DenseOut
        self._B = B
        self._D = D
        self._scale = scale
        self._stride_jn = stride_jn
        self._stride_jon = stride_jon
        self._stride_don = stride_don
        self._SCALE_JAGGED = SCALE_JAGGED
        self._BLOCK_D = BLOCK_D
        self._max_high_ind = max_high_ind

    def _generate_seq_offsets(self, seq_offsets_template, N, B, device):
        """
        确定性生成 seq_offsets 的值
        seq_offsets 必须是单调递增，从0到N
        每段长度在 [N//(2*B), N//B*2] 之间
        """
        # 计算每段的近似长度
        base_len = N // B
        remainder = N - base_len * B

        offsets = [0]
        for i in range(B):
            # 尾部多余的分配给前几个batch
            extra = 1 if i < remainder else 0
            seg_len = max(base_len + extra, 1)
            offsets.append(offsets[-1] + seg_len)

        # 确保最后一个元素正好是N
        offsets[-1] = N

        result = torch.tensor(offsets, dtype=seq_offsets_template.dtype, device=device)
        return result

    def _generate_high_inds(self, high_inds_template, seq_offsets, B, device):
        """
        多样化生成 high_inds 的值
        ratios = [0.2, 0.5, 0.8, 1.0] 覆盖多种分界比例
        high_inds[i] < seg_len，至少为1
        """
        seq_offsets_cpu = seq_offsets.cpu()
        ratios = [0.2, 0.5, 0.8, 1.0]
        vals = []
        for i in range(B):
            seg_len = int(seq_offsets_cpu[i + 1].item() - seq_offsets_cpu[i].item())
            ratio = ratios[i % len(ratios)]
            high_ind = max(1, min(seg_len - 1, int(seg_len * ratio)))
            vals.append(high_ind)

        result = torch.tensor(vals, dtype=high_inds_template.dtype, device=device)
        return result

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行 kernel（计时）"""
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        _add_position_embeddings_bwd_kernel[self._grid](
            self._Jagged,
            self._seq_offsets,
            self._high_inds,
            self._DenseOut,
            self._Jagged,  # JaggedOut: 复用Jagged（JaggedOut在SCALE_JAGGED时才写）
            self._B,
            self._D,
            self._scale,
            self._stride_jn,
            self._stride_jon,
            self._stride_don,
            SCALE_JAGGED=self._SCALE_JAGGED,
            BLOCK_D=self._BLOCK_D,
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        return self._DenseOut
