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

import os
import logging
import warnings

import importlib
import pkgutil


def env_setup_path() -> str:
    """返回 env_setup.sh 的路径，供 shell 级配置使用。

    默认情况下，import ops_rec 时会在当前进程内自动刷新
    ASCEND_CUSTOM_OPP_PATH，日常使用通常无需再 source 该脚本。

    如需在 shell 级别预先配置环境（例如给其他进程复用），可执行：
        source $(python3 -c "import ops_rec; print(ops_rec.env_setup_path())")
    """
    pkg_dir = os.path.realpath(os.path.dirname(__file__))
    path = os.path.join(pkg_dir, "env_setup.sh")
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"env_setup.sh not found at {path}")


def _opp_dir() -> str:
    """返回 opp/ 目录的绝对路径（包含 vendors/ 的上级目录）。"""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg_dir, "opp")


def _detect_soc_version() -> str | None:
    """通过 acl.get_soc_name() 检测当前机器 SOC 版本。

    返回值示例: "atlas_950", "atlas_a2", "atlas_a3"
    """
    try:
        import acl

        soc = acl.get_soc_name()
    except Exception:
        logging.warning("ops_rec: acl.get_soc_name() SOC detection failed.")
        return None

    variant = _normalize_variant(soc)
    if variant:
        return variant
    return None


def _normalize_variant(value: str) -> str | None:
    """把各种写法归一化为 ops_rec 的 variant 名（atlas_950 / atlas_a2 / atlas_a3）。

    支持：
      - 抽象 variant：atlas_950 / atlas_a2 / atlas_a3（大小写不敏感）
      - 代号：A5 -> atlas_950，A2 -> atlas_a2，A3 -> atlas_a3
      - 真实芯片名：Ascend950* -> atlas_950，Ascend910B* -> atlas_a2，Ascend910_93* -> atlas_a3
    无法识别返回 None。
    """
    s = (value or "").strip()
    if not s:
        return None

    low = s.lower()
    if low.startswith("atlas_"):
        if low in ("atlas_950", "atlas_a2", "atlas_a3"):
            return low
        return None

    if low == "a5":
        return "atlas_950"
    if low == "a2":
        return "atlas_a2"
    if low == "a3":
        return "atlas_a3"

    if s.lower().startswith("ascend950"):
        return "atlas_950"
    if s.lower().startswith("ascend910b"):
        return "atlas_a2"
    if s.lower().startswith("ascend910_93"):
        return "atlas_a3"
    return None


def _select_variant() -> str:
    """选择当前环境使用的 variant。

    优先级：
    1. SOC_VERSION 环境变量（真实芯片名/代号/抽象 variant，通用写法）
    2. OPS_REC_FORCE_BUILD_VER 环境变量（build/装包强制指定，同 SOC_VERSION）
    3. acl.get_soc_name() 自动探测
    4. 默认 atlas_950
    """
    for var in ("SOC_VERSION", "OPS_REC_FORCE_BUILD_VER"):
        value = os.environ.get(var, "")
        if value:
            variant = _normalize_variant(value)
            if variant:
                return variant

    variant = _detect_soc_version()
    if variant:
        return variant
    logging.warning("ops_rec: SOC not detected, defaulting to atlas_950")
    return "atlas_950"


def _setup_custom_opp_path(variant: str):
    # 设置 ASCEND_CUSTOM_OPP_PATH 指向 whl 内对应 variant 的 opp/<variant>/vendors 目录。
    opp_dir = _opp_dir()
    variant_root = os.path.join(opp_dir, variant)
    vendors_dir = os.path.join(variant_root, "vendors")
    if not os.path.isdir(vendors_dir):
        logging.warning(
            "ops_rec: variant '%s' not packaged (expected %s)",
            variant,
            vendors_dir,
        )
        return

    new_paths = []
    try:
        for name in os.listdir(vendors_dir):
            if name == "config.ini":
                continue
            vendor_path = os.path.join(vendors_dir, name)
            if os.path.isdir(vendor_path):
                new_paths.append(vendor_path)
    except Exception as e:
        logging.error("ops_rec: failed to scan vendors dir '%s': %s", vendors_dir, e)
        return

    new_paths.append(variant_root)

    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    old_parts = [p for p in existing.split(os.pathsep) if p and not p.startswith(opp_dir)]
    merged = new_paths + old_parts
    seen = set()
    deduped: list[str] = []
    for p in merged:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(deduped)
    logging.info(
        "ops_rec: ASCEND_CUSTOM_OPP_PATH=%s (variant=%s)",
        os.environ["ASCEND_CUSTOM_OPP_PATH"],
        variant,
    )


def _so_name(variant: str) -> str:
    """variant 名 → .so 文件名，例如 atlas_950 → ops_rec_atlas_950.so"""
    suffix = variant.lower()
    if suffix.startswith("atlas_"):
        suffix = suffix[len("atlas_") :]
    return f"ops_rec_atlas_{suffix}.so"


def _load_custom_so(variant: str, no_throw: bool = False):
    # 加载对应 variant 的 ops_rec_atlas_<variant>.so (torch 适配层).
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    so_name = _so_name(variant)
    so_path = os.path.join(pkg_dir, so_name)
    if not os.path.isfile(so_path):
        msg = f"ops_rec: {so_name} not found (expected at {so_path})"
        if no_throw:
            logging.warning("%s; continuing without adapter .so", msg)
            return
        raise FileNotFoundError(msg)

    try:
        import torch

        torch.ops.load_library(so_path)
    except Exception as exc:
        if no_throw:
            warnings.warn(f"ops_rec: Failed to load {so_path}: {exc}")
            return
        raise RuntimeError(f"ops_rec: Failed to load {so_path}: {exc}") from exc


def _load_plugins(*plugin_pkgs: str) -> None:
    """自动加载指定的注册子包，顺次导入包内所有 py 模块。"""
    for pkg_name in plugin_pkgs:
        try:
            pkg = importlib.import_module(f"{__name__}.{pkg_name}")
        except Exception as exc:
            logging.warning("ops_rec: failed to import plugin package '%s': %s", pkg_name, exc)
            continue
        for _, mod_name, is_pkg in pkgutil.iter_modules(pkg.__path__):
            try:
                importlib.import_module(f"{pkg.__name__}.{mod_name}")
            except Exception as exc:
                logging.warning("ops_rec: failed to import plugin module '%s.%s': %s", pkg.__name__, mod_name, exc)


def _bootstrap() -> None:
    from . import ascendc  # noqa: F401
    from . import triton  # noqa: F401
    from . import experimental  # noqa: F401

    # 加载各功能注册子包（autograd 等）
    _load_plugins("autograd")


# 模块加载时自动检测 SOC 并加载对应变体
_variant = _select_variant()
_setup_custom_opp_path(_variant)
_load_custom_so(_variant)
_bootstrap()
