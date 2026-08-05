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

"""Extract op_api artifacts from installed ops_rec package for aclnn example compilation.

Usage:
    python3 scripts/extract_op_api.py --pkg-dir <site-packages/ops_rec> --run-tmp <dir>
"""

import argparse
import logging
import os
import shutil
import sys
import traceback



def _exit_error(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract op_api from installed ops_rec package.")
    parser.add_argument("--pkg-dir", required=True, help="Path to ops_rec package directory in site-packages.")
    parser.add_argument("--run-tmp", required=True, help="Path to run_tmp output directory.")
    args = parser.parse_args()

    os.makedirs(args.run_tmp, exist_ok=True)
    extract_op_api(args.pkg_dir, args.run_tmp)


def _detect_soc_version() -> str | None:
    """通过 acl.get_soc_name() 检测当前机器 SOC 版本。"""
    try:
        import acl

        soc = acl.get_soc_name()
    except Exception:
        logging.debug("acl.get_soc_name() unavailable, trying no path")
        return None

    if soc.startswith("Ascend950"):
        return "atlas_950"
    if soc.startswith("Ascend910B"):
        return "atlas_a2"
    if soc.startswith("Ascend910_93"):
        return "atlas_a3"
    logging.debug("acl.get_soc_name returned unrecognized SoC: %s", soc)
    return None


def extract_op_api(pkg_dir: str, run_tmp_dir: str) -> None:
    print(f"[INFO] Using ops_rec from {pkg_dir}")

    if not os.path.isdir(pkg_dir):
        _exit_error(f"--pkg-dir not found or not a directory: {pkg_dir}")

    opp_root = os.path.join(pkg_dir, "opp")
    if not os.path.isdir(opp_root):
        _exit_error(
            f"ops_rec package has no opp/ subdir: {pkg_dir} (is the package properly installed with --install?)"
        )

    soc = _detect_soc_version()
    if not soc:
        _exit_error("failed to detect SOC variant (acl.get_soc_name returned unrecognized SoC)")

    variant_dir = os.path.join(opp_root, soc)
    if not os.path.isdir(variant_dir):
        _exit_error(f"variant dir not found under {opp_root}: expected {variant_dir} (detected soc={soc!r})")

    print(f"[INFO] Detected SOC variant: {soc}")

    vendors_dir = os.path.join(variant_dir, "vendors")
    if not os.path.isdir(vendors_dir):
        print(f"[WARN] vendors dir not found: {vendors_dir}; nothing to extract")
        return

    for vname in os.listdir(vendors_dir):
        if vname == "config.ini":
            continue
        vdir = os.path.join(vendors_dir, vname)
        op_api = os.path.join(vdir, "op_api")
        if os.path.isdir(op_api):
            dst = os.path.join(run_tmp_dir, vname)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(op_api, dst)
            print(f"[INFO] Extracted {vname} op_api to run_tmp")

    print(f"[INFO] Extracted aclnn headers and libs to {run_tmp_dir}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        _exit_error(f"file not found: {exc.filename} ({exc})")
    except Exception:
        traceback.print_exc()
        _exit_error("extract_op_api failed; see traceback above")
