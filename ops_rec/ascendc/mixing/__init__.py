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

# ops_rec.ascendc.mixing — AscendC mixing operators
#
# Usage:
#   import ops_rec
#   ops_rec.ascendc.mixing.token_mixing(x, gamma, beta, epsilon)
#   from ops_rec.ascendc.mixing import token_mixing
#
# All operators are registered in C++ via TORCH_LIBRARY under the ops_rec
# namespace. This module uses __getattr__ to dynamically proxy attribute
# access to torch.ops.ops_rec, and also exports operator names explicitly
# to support "from ... import <op_name>" syntax.

import torch

_OP_NAMESPACE = torch.ops.ops_rec

# 显式导入所有在 torch.ops.ops_rec 下注册的算子名，
# 支持 "from ops_rec.ascendc.mixing import token_mixing" 语法。
__all__ = [k for k in dir(_OP_NAMESPACE) if not k.startswith("_")]
for _name in __all__:
    globals()[_name] = getattr(_OP_NAMESPACE, _name)


def __getattr__(name: str):
    if hasattr(_OP_NAMESPACE, name):
        return getattr(_OP_NAMESPACE, name)

    available_ops = [k for k in dir(_OP_NAMESPACE) if not k.startswith("_")]
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Operator '{name}' is not supported or not compiled for the current NPU architecture. "
        f"Available ops: {available_ops}"
    )


def __dir__():
    own_attrs = set(globals().keys())
    op_attrs = {k for k in dir(_OP_NAMESPACE) if not k.startswith("_")}
    return sorted(own_attrs | op_attrs)
