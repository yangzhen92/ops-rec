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
# pylint: disable=attribute-defined-outside-init
"""
group_norm_bwd_dwdb 算子的ATK自定义API执行文件（Autotune版本）

该算子对 DW [GROUP_N, num_heads] 按行累加得到 FINAL_DW [num_heads]
"""

from __future__ import annotations
import sys
from pathlib import Path

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def _import_local_kernel():
    source_dir = str(Path(__file__).resolve().parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from group_norm_bwd_dwdb import _group_norm_bwd_dwdb

    return _group_norm_bwd_dwdb


_group_norm_bwd_dwdb = _import_local_kernel()


def get_device():
    """获取可用的设备，优先NPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


@register("group_norm_bwd_dwdb")
class GroupNormBwdDwdbApi(BaseApi):
    """group_norm_bwd_dwdb 算子的 API 封装（Autotune版本）"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理阶段（不计入性能）"""
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 获取输入参数
        dw_input = input_data.kwargs.get("DW", None)
        db_input = input_data.kwargs.get("DB", None)

        if dw_input is None:
            raise ValueError("DW not found")
        if db_input is None:
            raise ValueError("DB not found")

        # 处理 DW 和 DB：转换为 tensor 并放到目标设备
        if not isinstance(dw_input, torch.Tensor):
            dw = torch.tensor(dw_input, dtype=torch.float32, device=device)
        else:
            dw = dw_input.to(device)

        if not isinstance(db_input, torch.Tensor):
            db = torch.tensor(db_input, dtype=torch.float32, device=device)
        else:
            db = db_input.to(device)

        # 确保连续性
        if not dw.is_contiguous():
            dw = dw.contiguous()
        if not db.is_contiguous():
            db = db.contiguous()

        # DW shape: [GROUP_N, num_heads]
        GROUP_N = dw.shape[0]
        num_heads_val = dw.shape[1]

        # 创建输出 tensor
        final_dw = torch.empty((num_heads_val,), dtype=dw.dtype, device=device)
        final_db = torch.empty((num_heads_val,), dtype=db.dtype, device=device)

        # grid: 启动 num_heads 个 program，每个处理一列
        self._grid = (num_heads_val,)

        # 保存实例变量
        self._device = device
        self._dw = dw
        self._db = db
        self._final_dw = final_dw
        self._final_db = final_db
        self._GROUP_N = GROUP_N

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行 group_norm_bwd_dwdb 算子（计入性能）"""
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ============ Autotune 版本调用 ============
        # 使用关键字参数，不传 BLOCK_N，由 autotune 自动注入
        _group_norm_bwd_dwdb[self._grid](
            DW=self._dw,
            DB=self._db,
            FINAL_DW=self._final_dw,
            FINAL_DB=self._final_db,
            N=self._GROUP_N,
            # BLOCK_N 不传，由 autotune 自动选择最优值
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        if with_output:
            return self._final_dw, self._final_db
        return self._final_dw, self._final_db
