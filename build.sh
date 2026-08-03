#!/usr/bin/env bash
# ============================================================================
# 用法：
#   bash build.sh -j16
#   bash build.sh --socs atlas_950 --install -j16
# ============================================================================

set -e

# ---------------------------------------------------------------------------
# 默认值
# ---------------------------------------------------------------------------
INSTALL_MODE=false
JOBS=""
SOCS="atlas_950,atlas_a2,atlas_a3"  # 默认构建全部 SOC 变体

# ---------------------------------------------------------------------------
# 解析命令行参数
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --install)
            INSTALL_MODE=true
            shift
            ;;
        -j)
            JOBS="$2"
            shift 2
            ;;
        -j*)
            JOBS="${1#-j}"
            shift
            ;;
        --socs)
            SOCS="$2"
            shift 2
            ;;
        --socs=*)
            SOCS="${1#*=}"
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: bash build.sh [flags]

Flags:
  --install                 Compile and then install the merged whl
  -j[N]                     Compile thread nums (default: CPU/2)
  --socs <list>             SOC variants to build, comma-separated
                            eg: --socs atlas_950
                                --socs atlas_a2,atlas_a3,atlas_950 (default: all)

Environment variables:
  ASCEND_HOME_PATH          CANN install path (default: /usr/local/Ascend/latest)
  OPS_REC_FORCE_VARIANT     Force runtime SOC variant (for testing)

Examples:
  bash run_examples.sh      Run operator examples
  bash build.sh -j16
  bash build.sh --install -j16
EOF
            exit 0
            ;;
        *)
            echo "[WARN] Unknown argument: $1 (ignored)"
            shift
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 环境初始化与检查
# ---------------------------------------------------------------------------
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/latest}"

if [ -n "$JOBS" ]; then
    export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
fi

# CANN 的 npu_op_kernel_library 生成 $(MAKE) 引用，与 Ninja 不兼容
export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Unix Makefiles}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "${SCRIPT_DIR}" || exit 1

echo "==========================================================="
echo "[INFO] Build Configuration:"
echo "  - Install Mode        : ${INSTALL_MODE}"
echo "  - SOCs                : ${SOCS}"
echo "  - Ascend Home         : ${ASCEND_HOME_PATH}"
echo "  - Parallel Jobs       : ${CMAKE_BUILD_PARALLEL_LEVEL:-<auto>}"
echo "  - CMake Generator     : ${CMAKE_GENERATOR}"
echo "==========================================================="

# ---------------------------------------------------------------------------
# 工具函数：构建单个 SOC variant
# ---------------------------------------------------------------------------
build_single_soc() {
    local soc="$1"
    echo ""
    echo "-----------------------------------------------------------"
    echo "[INFO] Building SOC variant: ${soc}"
    echo "-----------------------------------------------------------"

    # 清理旧编译产物（每 SOC 必须重新编译）
    rm -rf _skbuild/ build/ dist_tmp/

    # 同步 experimental/triton/ -> ops_rec/experimental/triton/
    # 仅拷贝 .py 文件（不含 test/、README.md 等），清除非 .py 文件
    local triton_src="${SCRIPT_DIR}/experimental/triton"
    local triton_dst="${SCRIPT_DIR}/ops_rec/experimental/triton"
    if [ -d "${triton_src}" ]; then
        rm -rf "${triton_dst}"
        cp -a "${triton_src}" "${triton_dst}"
        find "${triton_dst}" -type f ! -name '*.py' -exec rm -f {} +
        find "${triton_dst}" -type d -name 'test' -exec rm -rf {} + 2>/dev/null || true
        find "${triton_dst}" -type d -empty -delete 2>/dev/null || true
        echo "[INFO] Synced experimental/triton/ -> ${triton_dst} (source only)"
    fi

    export OPS_REC_SOC_VARIANT="${soc}"

    # 执行构建
    pip wheel . --no-build-isolation --no-deps -v -w dist_tmp/ 2>&1

    # 确认 whl 已生成
    local whl_file
    whl_file=$(ls dist_tmp/ops_rec-*.whl 2>/dev/null | head -1)
    if [ -z "${whl_file}" ]; then
        echo "[ERROR] wheel not found in dist_tmp/ for variant ${soc}"
        exit 1
    fi

    # 构建完成，删除同步过来的临时文件
    if [ -d "${triton_dst}" ]; then
        rm -rf "${triton_dst}"
    fi

    # 提取此 variant 的产物到合并目录
    local merge_dir="${SCRIPT_DIR}/dist_merge"
    mkdir -p "${merge_dir}"

    python3 "${SCRIPT_DIR}/scripts/merge_wheel.py" \
        --src "${whl_file}" \
        --soc "${soc}" \
        --merge-dir "${merge_dir}"
}

# ---------------------------------------------------------------------------
# 主流程执行
# ---------------------------------------------------------------------------
main() {
    # 清理历史残余目录
    rm -rf dist_merge/ dist_tmp/ dist/

    # 1. 循环编译各个 SOC 变体
    IFS=',' read -ra SOC_ARRAY <<< "${SOCS}"
    for soc in "${SOC_ARRAY[@]}"; do
        soc="$(echo "${soc}" | xargs)"
        [ -z "${soc}" ] && continue
        build_single_soc "${soc}"
    done

    echo ""
    echo "==========================================================="
    echo "[INFO] Building Final Multi-SOC wheel..."
    echo "==========================================================="

    # 从当前 torch 获取版本号，确保最终 whl 版本与 torch 版本一致
    local torch_ver
    torch_ver=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")

    python3 "${SCRIPT_DIR}/scripts/merge_wheel.py" \
        --build-final \
        --merge-dir "${SCRIPT_DIR}/dist_merge" \
        --out-dir "${SCRIPT_DIR}/dist" \
        --version "${torch_ver}"

    # 清理中间过渡目录
    rm -rf dist_merge/ dist_tmp/

    # 检查最终 wheel 是否创建成功
    local final_whl
    final_whl=$(ls dist/ops_rec-*.whl 2>/dev/null | head -1)
    if [ -z "${final_whl}" ]; then
        echo "[ERROR] Build failed: no wheel file found in dist/"
        exit 1
    fi

    echo "[INFO] Build successful: ${final_whl}"

    # 2. 如果开启了 --install 模式，则自动安装
    if [ "${INSTALL_MODE}" = true ]; then
        echo ""
        echo "==========================================================="
        echo "[INFO] Installing ops_rec package..."
        echo "==========================================================="
        cd "${SCRIPT_DIR}/dist" && pip uninstall -y ops_rec >/dev/null 2>&1 || true
        pip install --force-reinstall --no-deps "${SCRIPT_DIR}/${final_whl}"
    fi

    echo ""
    echo "==========================================================="
    echo "[INFO] All Tasks Completed Successfully!"
    echo "==========================================================="
}

# 执行主函数
main "$@"
