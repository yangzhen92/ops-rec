#!/usr/bin/env bash
# ============================================================================
# 依赖：ops_rec 已安装（bash build.sh --install）
# 用法：
#   bash run_examples.sh                      # 跑所有算子所有类型
#   bash run_examples.sh --type torch         # 只跑 torch 样例
#   bash run_examples.sh --type aclnn --ops ln_mul   # 只跑 ln_mul 的 aclnn 样例
# ============================================================================

set -e

# ---------------------------------------------------------------------------
# 默认值
# ---------------------------------------------------------------------------
EXAMPLE_TYPE="all"   # aclnn | torch | all
EXAMPLE_OPS=""

# ---------------------------------------------------------------------------
# 解析命令行参数
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --type)
            EXAMPLE_TYPE="$2"
            shift 2
            ;;
        --type=*)
            EXAMPLE_TYPE="${1#*=}"
            shift
            ;;
        --ops)
            EXAMPLE_OPS="$2"
            shift 2
            ;;
        --ops=*)
            EXAMPLE_OPS="${1#*=}"
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: bash run_examples.sh [flags]

Flags:
  --type <aclnn|torch|all>  Example type to run (default: all)
  --ops <list>              Operator names, comma-separated (default: all)
                            eg: --ops ln_mul
                                --ops ln_mul,rmsnorm
EOF
            exit 0
            ;;
        *)
            echo "[WARN] Unknown argument: $1 (ignored)"
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

# ---------------------------------------------------------------------------
# 依赖检查：验证 ops_rec 是否已正确安装
# ---------------------------------------------------------------------------
if ! python3 -c "import ops_rec" 2>/dev/null; then
    echo "[ERROR] ops_rec is not installed. Run 'bash build.sh --install' first."
    exit 1
fi

# ---------------------------------------------------------------------------
# 工具函数：动态收集需要测试的算子列表
# ---------------------------------------------------------------------------
collect_ops() {
    if [ -n "${EXAMPLE_OPS}" ]; then
        IFS=',' read -ra parts <<< "${EXAMPLE_OPS}"
        for op in "${parts[@]}"; do
            echo "$(echo "${op}" | xargs)"
        done
    else
        # 根据 --type 从 ascendc/examples 目录发现算子
        (
            if [ "${EXAMPLE_TYPE}" = "aclnn" ] || [ "${EXAMPLE_TYPE}" = "all" ]; then
                find "${SCRIPT_DIR}/ascendc" -type f -name "test_aclnn_*.cpp" 2>/dev/null
            fi
            if [ "${EXAMPLE_TYPE}" = "torch" ] || [ "${EXAMPLE_TYPE}" = "all" ]; then
                find "${SCRIPT_DIR}/ascendc" -type f -name "test_torch_*.py" 2>/dev/null
            fi
        ) | sed 's/.*[/]test_[^_]*_//; s/[.].*//' \
          | sort -u
    fi
}

# ---------------------------------------------------------------------------
# aclnn 样例：提取算子 API -> g++ 编译 C++ 样例 -> 执行
# ---------------------------------------------------------------------------
run_aclnn_examples() {
    echo ""
    echo "==========================================================="
    echo "[INFO] Running ACLNN C++ Examples..."
    echo "==========================================================="

    local run_tmp="${SCRIPT_DIR}/build/run_tmp"

    # 仅在开始时清理并准备目录
    rm -rf "${run_tmp}"
    mkdir -p "${run_tmp}"

    local ops_rec_pkg
    ops_rec_pkg=$(mkdir -p "${SCRIPT_DIR}/build" && cd "${SCRIPT_DIR}/build" && python3 -c "import ops_rec; import os; print(os.path.dirname(ops_rec.__file__))")

    # 提取算子 API 产物
    python3 "${SCRIPT_DIR}/scripts/extract_op_api.py" \
        --pkg-dir "${ops_rec_pkg}" \
        --run-tmp "${run_tmp}"

    # 获取 ops_rec 设置的 ASCEND_CUSTOM_OPP_PATH（cd build/ 避免源码目录遮蔽 whl）
    local ascend_opp_path
    ascend_opp_path=$(cd "${SCRIPT_DIR}/build" && python3 -c "import ops_rec; import os; print(os.environ['ASCEND_CUSTOM_OPP_PATH'])")

    echo ""

    # 自动搜寻系统中的 CANN 头文件与动态库路径
    local arch_suffix
    case "$(uname -m)" in
        aarch64|arm64) arch_suffix="aarch64-linux" ;;
        *)             arch_suffix="x86_64-linux"  ;;
    esac

    local ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/latest}"

    local acl_include
    acl_include=$(find "${ascend_home}" -name "acl.h" -path "*/include/acl/*.h" \
        -not -path "*/tools/*" 2>/dev/null | head -1)
    if [ -z "${acl_include}" ]; then
        acl_include="${ascend_home}/${arch_suffix}/include"
    else
        acl_include=$(dirname "$(dirname "${acl_include}")")
    fi

    local ascend_lib
    local ascend_lib_extra=""
    ascend_lib=$(find "${ascend_home}" -name "libascendcl.so" -path "*/lib64/*" \
        -not -path "*/tools/*" 2>/dev/null | head -1)
    if [ -z "${ascend_lib}" ]; then
        ascend_lib="${ascend_home}/${arch_suffix}/lib64"
    else
        ascend_lib=$(dirname "${ascend_lib}")
    fi

    # aarch64 平台可能有额外的 ${arch_suffix}/lib64 依赖（如 libascend_hal.so）
    local ascend_arch_lib="${ascend_home}/${arch_suffix}/lib64"
    if [ -d "${ascend_arch_lib}" ] && [ "${ascend_lib}" != "${ascend_arch_lib}" ]; then
        ascend_lib_extra="${ascend_arch_lib}"
    fi

    # 聚合所有厂商目录的 include 与 lib 路径
    local all_include_dirs=""
    local all_lib_dirs=""
    for vd in "${run_tmp}"/*/; do
        [ -d "${vd}/include" ] && all_include_dirs="${all_include_dirs} -I${vd}/include"
        [ -d "${vd}/include/aclnnop" ] && all_include_dirs="${all_include_dirs} -I${vd}/include/aclnnop"
        [ -d "${vd}/lib" ] && all_lib_dirs="${all_lib_dirs} -L${vd}/lib"
    done

    for op in $(collect_ops); do
        local example_files=()
        while IFS= read -r f; do
            [ -n "$f" ] && example_files+=("$f")
        done < <(find "${SCRIPT_DIR}/ascendc" -path "*/${op}/examples/aclnn/test_aclnn_${op}*.cpp" 2>/dev/null | sort)

        if [ ${#example_files[@]} -eq 0 ]; then
            echo "[WARN] No aclnn example found for operator: ${op}, skipping."
            continue
        fi

        local lib_dir="${run_tmp}/${op}/lib"

        for example_file in "${example_files[@]}"; do
            local example_name
            example_name=$(basename "${example_file}" .cpp)
            local example_bin="${run_tmp}/${example_name}"

            echo "[INFO] Building aclnn example: ${example_name} (${op})..."
            local gcc_lib_dirs="${all_lib_dirs} -L${ascend_lib}"
            local run_lib_path="${lib_dir}:${ascend_lib}"
            [ -n "${ascend_lib_extra}" ] && gcc_lib_dirs="${gcc_lib_dirs} -L${ascend_lib_extra}" && run_lib_path="${run_lib_path}:${ascend_lib_extra}"

            g++ -std=c++17 \
                ${all_include_dirs} \
                -I"${acl_include}" \
                ${gcc_lib_dirs} \
                "${example_file}" \
                -lcust_opapi -lascendcl -lnnopbase \
                -o "${example_bin}"

            echo "[INFO] Running aclnn example: ${example_name}..."
            ASCEND_CUSTOM_OPP_PATH="${ascend_opp_path}" \
            LD_LIBRARY_PATH="${run_lib_path}:${LD_LIBRARY_PATH:-}" \
            "${example_bin}"

            echo "[INFO] [PASSED] ${example_name}"
            echo ""
        done
    done
}

# ---------------------------------------------------------------------------
# torch 样例：自动发现并运行 test_torch_*.py 脚本
# ---------------------------------------------------------------------------
run_torch_examples() {
    echo ""
    echo "==========================================================="
    echo "[INFO] Running Torch Python Examples..."
    echo "==========================================================="

    for op in $(collect_ops); do
        local example_files=()
        while IFS= read -r f; do
            [ -n "$f" ] && example_files+=("$f")
        done < <(find "${SCRIPT_DIR}/ascendc" -path "*/${op}/examples/torch/test_torch_${op}*.py" 2>/dev/null | sort)

        if [ ${#example_files[@]} -eq 0 ]; then
            echo "[WARN] No torch example found for operator: ${op}, skipping."
            continue
        fi

        for example_file in "${example_files[@]}"; do
            local example_name
            example_name=$(basename "${example_file}" .py)

            echo "[INFO] Running torch example: ${example_name} (${op})..."
            mkdir -p "${SCRIPT_DIR}/build" && cd "${SCRIPT_DIR}/build" && python3 "${example_file}"

            echo "[INFO] [PASSED] ${example_name}"
            echo ""
        done
    done
}

# ---------------------------------------------------------------------------
# 主流程控制
# ---------------------------------------------------------------------------
main() {
    if [ "${EXAMPLE_TYPE}" = "all" ] || [ "${EXAMPLE_TYPE}" = "aclnn" ]; then
        run_aclnn_examples
    fi

    if [ "${EXAMPLE_TYPE}" = "all" ] || [ "${EXAMPLE_TYPE}" = "torch" ]; then
        run_torch_examples
    fi

    echo ""
    echo "==========================================================="
    echo "[INFO] All Requested Examples Passed Successfully!"
    echo "==========================================================="
}

main "$@"
