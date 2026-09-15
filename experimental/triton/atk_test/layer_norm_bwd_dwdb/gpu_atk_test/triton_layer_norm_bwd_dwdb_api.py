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

from __future__ import annotations


import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from triton_layer_norm_bwd_dwdb_kernel import _layer_norm_bwd_dwdb

# pylint: disable=attribute-defined-outside-init


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def _get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("_layer_norm_bwd_dwdb requires an NPU or CUDA device")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def _compute_block_d(d: int, sms: int) -> int:
    blocks = triton.next_power_of_2(sms * 4)
    block_d = triton.next_power_of_2(triton.cdiv(d, blocks))
    block_d = min(max(block_d, 4), 128)
    return block_d


@register("triton_layer_norm_bwd_dwdb")
@register("_layer_norm_bwd_dwdb")
class LayerNormBwdDwdbApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        dw_input = input_data.kwargs.get("DW")
        db_input = input_data.kwargs.get("DB")
        final_dw_input = input_data.kwargs.get("FINAL_DW")
        final_db_input = input_data.kwargs.get("FINAL_DB")
        if not all(isinstance(v, torch.Tensor) for v in (dw_input, db_input, final_dw_input, final_db_input)):
            raise ValueError("DW, DB, FINAL_DW, FINAL_DB must all be tensors")

        dw_input = dw_input.to(torch.float32) if dw_input.dtype == torch.float64 else dw_input
        db_input = db_input.to(torch.float32) if db_input.dtype == torch.float64 else db_input
        if dw_input.ndim != 2 or db_input.ndim != 2:
            raise ValueError(f"Expected DW, DB to be 2D, got DW {dw_input.ndim}D, DB {db_input.ndim}D")
        if final_dw_input.ndim != 1 or final_db_input.ndim != 1:
            raise ValueError("FINAL_DW and FINAL_DB must be 1D")
        if dw_input.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dw_input.dtype}")
        if dw_input.shape != db_input.shape:
            raise ValueError(f"DW shape {dw_input.shape} != DB shape {db_input.shape}")
        if final_dw_input.shape != final_db_input.shape:
            raise ValueError(f"FINAL_DW shape {final_dw_input.shape} != FINAL_DB shape {final_db_input.shape}")
        if final_dw_input.shape[0] != dw_input.shape[1]:
            raise ValueError(f"FINAL_DW dim {final_dw_input.shape[0]} != DW second dim {dw_input.shape[1]}")

        tile_num, d = dw_input.shape

        n_val = tile_num

        device = _get_device()

        self._dw = dw_input.to(device).contiguous()
        self._db = db_input.to(device).contiguous()
        self._final_dw = torch.empty((d,), dtype=self._dw.dtype, device=device)
        self._final_db = torch.empty((d,), dtype=self._dw.dtype, device=device)

        sms = 0
        if device == "npu":
            sms = 64
        else:
            try:
                sms = torch.cuda.get_device_properties(device).multi_processor_count
            except Exception:
                sms = 64
        block_d = _compute_block_d(d, sms)

        block_n = 64

        grid = (triton.cdiv(d, block_d),)

        self._device = device
        self._d = d
        self._tile_num = n_val
        self._block_d = block_d
        self._block_n = block_n
        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        _synchronize(self._device)

        _layer_norm_bwd_dwdb[self._grid](
            self._dw,
            self._db,
            self._final_dw,
            self._final_db,
            self._tile_num,
            self._d,
            BLOCK_N=self._block_n,
            BLOCK_D=self._block_d,
        )

        _synchronize(self._device)

        return self._final_dw, self._final_db
