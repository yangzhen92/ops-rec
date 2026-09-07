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
_rms_norm_bwd_dwdb 算子的ATK自定义API执行文件

该算子对 DW [N, D] 按行累加得到 FINAL_DW [D]
"""

import torch
import triton
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

NCORES = 64


def get_device():
    """获取可用的设备，优先NPU"""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except:
        pass
    return "cpu"


from _rms_norm_bwd_dwdb_kernel import _rms_norm_bwd_dwdb_kernel


@register("_rms_norm_bwd_dwdb")
class RMSNormBwdDwdbApi(BaseApi):
    """_rms_norm_bwd_dwdb 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理阶段（不计入性能）"""
        device = get_device()
        if device == "npu":
            torch.npu.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        # 获取输入参数
        dw_input = input_data.kwargs.get("DW", None)

        if dw_input is None:
            raise ValueError("DW not found")

        # 处理 DW：转换为 tensor 并放到目标设备
        if not isinstance(dw_input, torch.Tensor):
            dw = torch.tensor(dw_input, dtype=torch.float32, device=device)
        else:
            dw = dw_input.to(device)

        # 确保连续性
        if not dw.is_contiguous():
            dw = dw.contiguous()

        # DW shape: [N, D]
        N = dw.shape[0]
        D = dw.shape[1]

        # 创建输出 tensor
        final_dw = torch.empty((D,), dtype=dw.dtype, device=device)

        # grid: 启动 cdiv(D, BLOCK_D) 个 program，每个处理BLOCK_D列
        BLOCK_D = 32

        grid = lambda meta: (min(NCORES, triton.cdiv(D, meta["BLOCK_D"])),)  # NPU上小，ncores

        # 保存实例变量
        self._device = device
        self._dw = dw
        self._final_dw = final_dw
        self._N = N
        self._D = D
        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行 _rms_norm_bwd_dwdb 算子（计入性能）"""
        device = self._device

        if device == "npu":
            torch.npu.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        _rms_norm_bwd_dwdb_kernel[self._grid](
            self._dw,
            self._final_dw,
            self._N,
            self._D,
            # BLOCK_N=BLOCK_N,
            # BLOCK_D=BLOCK_D,
        )

        if device == "npu":
            torch.npu.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        # # 修改：同时返回 FINAL_DW 和 FINAL_DB
        # if with_output:
        #     return self._final_dw, self._final_db

        return self._final_dw
