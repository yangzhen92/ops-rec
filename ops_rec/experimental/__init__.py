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

# 实验性算子，按后端分组，与 ops_rec.ascendc / ops_rec.triton 结构一致。
# 使用方式：
#   from ops_rec.experimental.ascendc import xxx
#   from ops_rec.experimental.triton import xxx
#   ops_rec.experimental.ascendc.xxx(...)

import importlib
import pkgutil

__all__ = []

for _, name, is_pkg in pkgutil.iter_modules(__path__):
    if is_pkg:
        mod = importlib.import_module('.' + name, __package__)
        globals()[name] = mod
        __all__.append(name)
