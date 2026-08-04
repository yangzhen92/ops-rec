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

# AscendC 算子 Python 封装，每个算子组是一个子包（如 activation, attention 等），自动发现并导入。
# 使用方式：
#   from ops_rec.ascendc.activation import ln_mul
#   ops_rec.ascendc.activation.ln_mul(x, u, gamma, beta)

# pylint: disable=duplicate-code
import importlib
import logging
import pkgutil

__all__ = []

for _, name, is_pkg in pkgutil.iter_modules(__path__):
    if not is_pkg:
        continue
    try:
        mod = importlib.import_module('.' + name, __package__)
    except Exception:
        logging.warning("ops_rec.ascendc: failed to import subpackage '%s', skipping", name)
        continue
    globals()[name] = mod
    __all__.append(name)