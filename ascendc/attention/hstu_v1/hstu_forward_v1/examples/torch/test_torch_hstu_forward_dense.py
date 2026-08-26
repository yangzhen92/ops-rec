#!/usr/bin/env python3
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
hstu_forward_v1 算子 PyTorch 接口使用样例 — dense（normal）格式

覆盖接口（参数名与 torch schema 一一对应）：
    - hstu_dense            : normal（dense）格式，q/k/v 为 4 维 [B, S, N, D]

用法:
    python3 test_torch_hstu_forward_dense.py
"""

import torch
import ops_rec

# 与 pytest 保持一致：关闭内部格式，保证输入按 ND 布局送入算子
torch.npu.config.allow_internal_format = False


def test_hstu_dense():
    """运行 hstu_dense 前向样例（normal 格式，内置下三角 mask），打印输出便于查看。"""
    torch.npu.set_device(0)
    B, S, N, D = 2, 16, 4, 32
    silu_scale = 1.0 / S

    q = torch.randn(B, S, N, D, dtype=torch.float16).npu()
    k = torch.randn(B, S, N, D, dtype=torch.float16).npu()
    v = torch.randn(B, S, N, D, dtype=torch.float16).npu()

    y = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, None, 0, S, silu_scale)

    torch.npu.synchronize()
    print(f"[torch] hstu_dense output shape: {tuple(y.shape)}")
    print(f"[torch] hstu_dense output sample: {y.flatten()[:4].cpu().tolist()}")


if __name__ == "__main__":
    test_hstu_dense()
    print("[torch] hstu_dense torch example done")
