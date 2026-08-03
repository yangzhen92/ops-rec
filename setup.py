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

# 使用方式：
#   pip install . --no-build-isolation
# 依赖：torch, torch_npu, scikit-build

import os
import sys
import setuptools
from setuptools.command.build_py import build_py as _build_py
from skbuild import setup as skbuild_setup

class _silent_build_py(_build_py):
    def analyze_manifest(self):
        import setuptools.command.build_py as bpm

        old = getattr(bpm._IncludePackageDataAbuse, "warn", None)
        try:
            bpm._IncludePackageDataAbuse.warn = lambda *args, **kwargs: None
            super().analyze_manifest()
        finally:
            if old is not None:
                bpm._IncludePackageDataAbuse.warn = old

def _get_torch_version():
    import torch

    return torch.__version__.split("+")[0]

def _get_torch_prefix():
    import torch

    return os.path.dirname(torch.__file__)

def _get_cxx11_abi():
    import torch

    return int(torch._C._GLIBCXX_USE_CXX11_ABI)

def cmake_args():
    torch_root = _get_torch_prefix()

    os.environ.setdefault(
        "CMAKE_BUILD_PARALLEL_LEVEL",
        str((os.cpu_count() or 4) // 2),
    )

    # 安全编译选项
    compiler_flags = "-fstack-protector-strong"
    link_flags = "-Wl,-z,relro -Wl,-z,now -Wl,-s"

    # ASCEND_HOME_PATH 检查（在 cmake 之前设好环境变量）
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_home:
        for candidate in ("/usr/local/Ascend/latest",):
            if os.path.isdir(candidate):
                ascend_home = candidate
                break
    if ascend_home:
        os.environ["ASCEND_HOME_PATH"] = ascend_home

    args = [
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DCMAKE_PREFIX_PATH={torch_root}",
        f"-D_GLIBCXX_USE_CXX11_ABI={_get_cxx11_abi()}",
        f"-DCMAKE_C_FLAGS={compiler_flags}",
        f"-DCMAKE_CXX_FLAGS={compiler_flags}",
        f"-DCMAKE_EXE_LINKER_FLAGS={link_flags}",
        f"-DCMAKE_SHARED_LINKER_FLAGS={link_flags}",
        "-DCMAKE_SKIP_BUILD_RPATH=FALSE",
        "-DCMAKE_SKIP_INSTALL_RPATH=FALSE",
    ]

    # 通过 CMAKE_TOOLCHAIN_FILE 引用 toolchain
    script_dir = os.path.dirname(os.path.abspath(__file__))
    toolchain_path = os.path.join(script_dir, "cmake", "toolchain_ascend.cmake")
    if os.path.isfile(toolchain_path):
        args.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain_path}")

    # 从环境变量传递 OPS_REC_SOC_VARIANT 到 cmake（多 SOC 构建支持）
    soc_variant = os.environ.get("OPS_REC_SOC_VARIANT", "")
    if soc_variant:
        args.append(f"-DOPS_REC_SOC_VARIANT={soc_variant}")
    else:
        # 兼容旧式 ASCEND_COMPUTE_UNIT 环境变量
        ascend_compute_unit = os.environ.get("ASCEND_COMPUTE_UNIT", "")
        if ascend_compute_unit:
            args.append(f"-DASCEND_COMPUTE_UNIT={ascend_compute_unit}")

    return args

skbuild_setup(
    cmdclass={"build_py": _silent_build_py},
    name="ops-rec",
    version=_get_torch_version(),
    description="Ascend NPU Custom Operations for Recommendation Systems",
    packages=setuptools.find_namespace_packages(
        where=".",
        include=["ops_rec*"],
        exclude=[],
    ),
    package_dir={"ops_rec": "ops_rec"},
    include_package_data=True,
    package_data={
        "ops_rec": ["opp/**", "env_setup.sh"],
    },
    python_requires=">=3.10",
    cmake_args=cmake_args(),
    cmake_install_dir="ops_rec",
)
