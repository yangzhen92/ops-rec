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

"""Merge single-SOC whl artifacts into a shared directory for multi-SOC packaging.

Usage:
    python3 scripts/merge_wheel.py --src <whl_path> --soc <variant> --merge-dir <dir>
    python3 scripts/merge_wheel.py --merge-dir <dir> --build-final --out-dir <out_dir>
"""

import argparse
import logging
import shutil
import sys
import sysconfig
import traceback
from pathlib import Path
import tempfile
import zipfile


def _exit_error(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge single-SOC wheel into multi-SOC wheel")
    parser.add_argument("--src", default=None, help="Path to single-SOC .whl file")
    parser.add_argument("--soc", default=None, help="SOC variant name (e.g. atlas_950)")
    parser.add_argument("--merge-dir", required=True, help="Merge directory for extracted artifacts")
    parser.add_argument("--out-dir", default=None, help="Output directory for final wheel (only for final step)")
    parser.add_argument("--build-final", action="store_true", help="Build final multi-SOC wheel from merge_dir")
    parser.add_argument("--version", default=None, help="Package version (detected automatically if not provided)")

    args = parser.parse_args()

    if args.build_final:
        build_final_wheel(args.merge_dir, args.out_dir or str(Path.cwd() / "dist"), args.version)
    else:
        if not args.src or not args.soc:
            parser.error("--src and --soc are required when not using --build-final")
        merge_single(args.src, args.soc, args.merge_dir)


def build_final_wheel(merge_dir: str, out_dir: str, version: str | None = None) -> str:
    """组装多 SOC 产物并打包生成最终的 Wheel 安装包"""
    from packaging.version import Version

    merge_path = Path(merge_dir).resolve()
    out_path_dir = Path(out_dir).resolve()

    if not merge_path.is_dir():
        _exit_error(f"merge_dir invalid: {merge_path}")

    if not version:
        version = _detect_version(merge_path)
    if not version:
        _exit_error(
            "cannot determine package version: pass --version explicitly "
            f"or ensure {merge_path} contains an ops_rec-<ver>.dist-info/ directory"
        )

    try:
        Version(version)  # 验证版本号格式合法性
    except Exception as exc:
        _exit_error(f"invalid version '{version}': {exc}")

    py_tag = _python_tag()
    abi_tag = _abi_tag()
    plat_tag = _platform_tag()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pkg_dst = tmp_path / "ops_rec"

        # 复制合并后的内容
        shutil.copytree(merge_path, pkg_dst)

        dist_info_dir = tmp_path / f"ops_rec-{version}.dist-info"
        dist_info_dir.mkdir(parents=True, exist_ok=True)

        # 写入元数据文件
        (dist_info_dir / "METADATA").write_text(_generate_metadata(version), encoding="utf-8")
        (dist_info_dir / "WHEEL").write_text(_generate_wheel_meta(py_tag, abi_tag, plat_tag), encoding="utf-8")
        (dist_info_dir / "top_level.txt").write_text("ops_rec\n", encoding="utf-8")

        # 生成 RECORD 文件
        _write_record(tmp_path, dist_info_dir, version)

        # 打包最终的 Wheel 压缩包
        out_path_dir.mkdir(parents=True, exist_ok=True)
        final_out_path = out_path_dir / f"ops_rec-{version}-{py_tag}-{abi_tag}-{plat_tag}.whl"

        if final_out_path.exists():
            final_out_path.unlink()

        with zipfile.ZipFile(final_out_path, "w", zipfile.ZIP_DEFLATED) as z:
            for filepath in tmp_path.rglob("*"):
                if filepath.is_file():
                    z.write(filepath, filepath.relative_to(tmp_path).as_posix())

    print(f"[INFO] Merged whl: {final_out_path}")
    return str(final_out_path)


def merge_single(src: str, soc: str, merge_dir: str) -> None:
    """提取单 SOC 算子包并将其归档到共享合并目录中"""
    src_path = Path(src).resolve()
    merge_path = Path(merge_dir).resolve()

    if not src_path.is_file():
        _exit_error(f"--src not found or not a file: {src_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        with zipfile.ZipFile(src_path, "r") as z:
            z.extractall(tmp_path)

        # 解包并对齐数据目录
        _extract_and_align_data(tmp_path)

        pkg_dir = tmp_path / "ops_rec"
        if not pkg_dir.is_dir():
            _exit_error(
                f"ops_rec package dir not found in wheel: {src_path} "
                f"(top-level dirs: {sorted(p.name for p in tmp_path.iterdir())})"
            )

        merge_path.mkdir(parents=True, exist_ok=True)

        for item in pkg_dir.iterdir():
            if _skip_item(item.name):
                continue
            dst_item = merge_path / item.name

            if item.is_dir():
                if dst_item.exists():
                    for src_file in item.rglob("*"):
                        if src_file.is_file():
                            dst_file = dst_item / src_file.relative_to(item)
                            dst_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_file, dst_file)
                else:
                    shutil.copytree(item, dst_item)
            elif item.is_file():
                shutil.copy2(item, dst_item)

    print(f"[INFO] Extracted {soc} artifacts to merge dir")


def _extract_and_align_data(tmp_path: Path) -> None:
    """解析单 SOC 包中的 data 目录，并将算子产物对齐到标准包路径下"""
    data_dir_candidates = list(tmp_path.glob("ops_rec-*.data/data/ops_rec"))
    if not data_dir_candidates:
        return

    data_dir = data_dir_candidates[0]
    pkg_dir = tmp_path / "ops_rec"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    for item in data_dir.iterdir():
        if item.name in ("lib", "include"):
            continue
        dst_item = pkg_dir / item.name
        if item.is_dir():
            if dst_item.exists():
                shutil.rmtree(dst_item)
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_item)

    for d in tmp_path.glob("ops_rec-*.data"):
        shutil.rmtree(d)


def _detect_version(merge_dir: Path) -> str | None:
    """从已解压的 .dist-info 目录中安全检测并验证版本号"""
    from packaging.version import Version

    pkg_dir = merge_dir / "ops_rec"
    if not pkg_dir.is_dir():
        return None

    for entry in merge_dir.iterdir():
        if entry.name.startswith("ops_rec-") and entry.name.endswith(".dist-info"):
            ver = entry.name[len("ops_rec-") : -len(".dist-info")]
            try:
                Version(ver)
                return ver
            except Exception as exc:  # nosec B110 - 跳过非 version 格式的目录项
                logging.debug("Skipping non-version dist-info entry %s: %s", entry.name, exc)
    return None


def _write_record(tmp_path: Path, dist_info_dir: Path, version: str) -> None:
    """计算所有打包文件的哈希与大小，生成 Wheel 必需的 RECORD 清单"""
    record_lines = []
    for filepath in tmp_path.rglob("*"):
        if filepath.is_file():
            arcname = filepath.relative_to(tmp_path).as_posix()
            size = filepath.stat().st_size
            record_lines.append(f"{arcname},sha256=,{size}")
    record_lines.append(f"ops_rec-{version}.dist-info/RECORD,,")
    (dist_info_dir / "RECORD").write_text("\n".join(record_lines), encoding="utf-8")


def _skip_item(name: str) -> bool:
    """过滤在多 SOC 合并时需要跳过的特定文件或目录"""
    return name in frozenset({"lib", "include", ".dist-info", "_available_ops.py"})


def _python_tag() -> str:
    """获取当前 Python 主副版本标签（如 cp310）"""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _abi_tag() -> str:
    """获取 ABI 兼容标签"""
    return _python_tag()


def _platform_tag() -> str:
    """获取系统平台标签（如 manylinux 或 linux_x86_64）"""
    return f"linux_{sysconfig.get_platform().split('-')[-1]}"


def _generate_metadata(version: str) -> str:
    """生成 Wheel METADATA 内容模板"""
    return (
        f"Metadata-Version: 2.1\n"
        f"Name: ops-rec\n"
        f"Version: {version}\n"
        f"Summary: Ascend NPU Custom Operations for Recommendation Systems\n"
        f"Requires-Python: >=3.10\n"
    )


def _generate_wheel_meta(py_tag: str, abi_tag: str, plat_tag: str) -> str:
    """生成 Wheel 配置文件内容模板"""
    return (
        f"Wheel-Version: 1.0\nGenerator: ops-rec build.sh\nRoot-Is-Purelib: false\nTag: {py_tag}-{abi_tag}-{plat_tag}\n"
    )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        _exit_error(f"file not found: {exc.filename} ({exc})")
    except zipfile.BadZipFile as exc:
        _exit_error(f"corrupted .whl file: {exc}")
    except Exception:
        traceback.print_exc()
        _exit_error("merge_wheel failed; see traceback above")
