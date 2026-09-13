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


@register("triton_layer_norm_bwd_dwdb")
@register("_layer_norm_bwd_dwdb")
class LayerNormBwdDwdbApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        dw_input = input_data.kwargs.get("DW")
        db_input = input_data.kwargs.get("DB")
        final_dw_input = input_data.kwargs.get("FINAL_DW")
        final_db_input = input_data.kwargs.get("FINAL_DB")

        dw_input = dw_input.to(torch.float32) if dw_input.dtype == torch.float64 else dw_input
        db_input = db_input.to(torch.float32) if db_input.dtype == torch.float64 else db_input
        if not all(isinstance(v, torch.Tensor) for v in (dw_input, db_input, final_dw_input, final_db_input)):
            raise ValueError("DW, DB, FINAL_DW, FINAL_DB must all be tensors")
        if dw_input.ndim != 2 or db_input.ndim != 2:
            raise ValueError(f"Expected DW, DB to be 2D, got DW {dw_input.ndim}D, DB {db_input.ndim}D")
        if final_dw_input.ndim != 1 or final_db_input.ndim != 1:
            raise ValueError("FINAL_DW and FINAL_DB must be 1D")
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

        self._device = device
        self._d = d
        self._tile_num = n_val

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        _synchronize(self._device)

        def grid(META):
            return (triton.cdiv(self._d, META["BLOCK_D"]),)

        _layer_norm_bwd_dwdb[grid](
            DW=self._dw,
            DB=self._db,
            FINAL_DW=self._final_dw,
            FINAL_DB=self._final_db,
            N=self._tile_num,
            D=self._d,
        )

        _synchronize(self._device)

        return self._final_dw, self._final_db
