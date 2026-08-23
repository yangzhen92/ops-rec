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

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)

__all__: list[str] = []


def _register_modules() -> None:
    # 逐模块注册：任一模块失败仅告警一次（warning），不阻断其余模块
    for _, mod_name, is_pkg in pkgutil.iter_modules(__path__):
        if is_pkg:
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{mod_name}")
        except Exception as exc:
            logger.warning("ops_rec.autograd: failed to import '%s': %s", mod_name, exc)
            continue
        register_fn = getattr(mod, "register_autograd", None)
        if register_fn is None:
            logger.warning("ops_rec.autograd: module '%s' has no register_autograd(), skipping", mod_name)
            continue
        try:
            register_fn()
        except Exception as exc:
            logger.warning("ops_rec.autograd: register_autograd() in '%s' failed: %s", mod_name, exc)
            continue
        __all__.append(mod_name)


_register_modules()
