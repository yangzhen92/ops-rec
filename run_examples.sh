#!/usr/bin/env bash
# ============================================================================
# 依赖：ops_rec 已安装（bash build.sh --install）
# 用法：
#   bash run_examples.sh                            # 跑所有算子所有类型
#   bash run_examples.sh --type torch               # 只跑 torch 样例
#   bash run_examples.sh --type aclnn --ops ln_mul  # 只跑 ln_mul 的 aclnn 样例
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
            if [ -z "${2:-}" ]; then
                echo "[ERROR] --type requires a value (aclnn | torch | all)" >&2
                exit 2
            fi
            EXAMPLE_TYPE="$2"
            shift 2
            ;;
        --type=*)
            if [ "${1}" = "--type=" ]; then
                echo "[ERROR] --type= requires a value (aclnn | torch | all)" >&2
                exit 2
            fi
            EXAMPLE_TYPE="${1#*=}"
            shift
            ;;
        --ops)
            if [ -z "${2:-}" ]; then
                echo "[ERROR] --ops requires a value (comma-separated op vendor names)" >&2
                exit 2
            fi
            EXAMPLE_OPS="$2"
            shift 2
            ;;
        --ops=*)
            if [ "${1}" = "--ops=" ]; then
                echo "[ERROR] --ops= requires a value (comma-separated op vendor names)" >&2
                exit 2
            fi
            EXAMPLE_OPS="${1#*=}"
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: bash run_examples.sh [flags]

Flags:
  --type <aclnn|torch|all>  Example type to run (default: all)
  --ops <list>              Op vendor names, comma-separated (default: all)
                            Values are the CANN vendor package names
                            eg: --ops ln_mul
                                --ops ln_mul,disentangle_attention
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
# 依赖检查：先确认 ops_rec 已安装，获取并 export ASCEND_CUSTOM_OPP_PATH
# ---------------------------------------------------------------------------
if [ ! -d "${SCRIPT_DIR}/build" ]; then
    mkdir -p "${SCRIPT_DIR}/build"
fi

cd "${SCRIPT_DIR}/build" || exit 1

if ! python3 -c "import ops_rec" 2>/dev/null; then
    echo "[ERROR] ops_rec is not installed. Run 'bash build.sh --install' first." >&2
    exit 1
fi

_ops_rec_opp_path=$(python3 -c '
import sys, os, ops_rec
v = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
sys.exit("[ERROR] ASCEND_CUSTOM_OPP_PATH not set by ops_rec; check ops_rec.__init__ SOC detection.") if not v else print(v)
')
if [ -z "${_ops_rec_opp_path}" ]; then
    # Step 1 已通过 import 检查，所以这里空说明 SOC 检测没设上变量
    echo "[ERROR] ASCEND_CUSTOM_OPP_PATH not set by ops_rec; check ops_rec.__init__ SOC detection." >&2
    exit 1
fi
export ASCEND_CUSTOM_OPP_PATH="${_ops_rec_opp_path}"
unset _ops_rec_opp_path

# ---------------------------------------------------------------------------
# 工具函数：动态收集需要测试的算子列表
# ---------------------------------------------------------------------------

# example 文件路径查找：命名约定 test_<kind>_<op>.<ext> under examples/<kind>/<op>/
_example_paths() {
    find "${SCRIPT_DIR}/ascendc" \
        -path "*/${2}/examples/${1}/test_${1}_*${3}" 2>/dev/null | sort
}

# 从 example 路径里提取 op 名（取路径里 <op>/examples/<kind>/test_<kind>_<op>.<ext> 的 op 段）
_op_from_example_path() {
    awk -F/ '{gsub(/\.[^.]+$/, "", $NF); sub(/^test_(aclnn|torch)_/, "", $NF); print $NF}'
}

# 查询聚合 vendor 在 soc_ops_config.cmake 中展开后的真实子 vendor 列表。
op_leaves_for_soc() {
    local op="$1"
    # soc_ops_config.cmake 用 message() 把查询结果打到 stderr，需 2>&1 合并捕获
    cmake -DSOC=atlas_950 -DOPS_REC_QUERY_OP="${op}" \
        -P "${SCRIPT_DIR}/cmake/soc_ops_config.cmake" 2>&1
}

collect_ops() {
    if [ -n "${EXAMPLE_OPS}" ]; then
        IFS=',' read -ra parts <<< "${EXAMPLE_OPS}"
        for op in "${parts[@]}"; do
            op="$(echo "${op}" | xargs)"
            [ -z "${op}" ] && continue
            local -a leaves=()
            while IFS= read -r leaf; do
                leaf="$(echo "${leaf}" | xargs)"
                [ -n "${leaf}" ] && leaves+=("${leaf}")
            done < <(op_leaves_for_soc "${op}")
            if [ "${#leaves[@]}" -gt 0 ] \
               && ! { [ "${#leaves[@]}" -eq 1 ] && [ "${leaves[0]}" = "${op}" ]; }; then
                printf '%s\n' "${leaves[@]}"
            else
                echo "${op}"
            fi
        done
    else
        {
            if [ "${EXAMPLE_TYPE}" = "aclnn" ] || [ "${EXAMPLE_TYPE}" = "all" ]; then
                _example_paths aclnn '*' .cpp
            fi
            if [ "${EXAMPLE_TYPE}" = "torch" ] || [ "${EXAMPLE_TYPE}" = "all" ]; then
                _example_paths torch '*' .py
            fi
        } | _op_from_example_path | sort -u
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

    rm -rf "${run_tmp}" && mkdir -p "${run_tmp}"

    local ops_rec_pkg=$(python3 -c "import ops_rec; import os; print(os.path.dirname(ops_rec.__file__))")

    # 提取算子 API 产物
    python3 "${SCRIPT_DIR}/scripts/extract_op_api.py" \
        --pkg-dir "${ops_rec_pkg}" \
        --run-tmp "${run_tmp}"

    # 自动搜寻系统中的 CANN 头文件与动态库路径
    local arch_suffix
    case "$(uname -m)" in
        aarch64|arm64) arch_suffix="aarch64-linux" ;;
        *)             arch_suffix="x86_64-linux"  ;;
    esac

    local ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/latest}"

    local acl_include=$(find "${ascend_home}" -name "acl.h" -path "*/include/acl/*.h" \
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

    local ascend_arch_lib="${ascend_home}/${arch_suffix}/lib64"
    if [ -d "${ascend_arch_lib}" ] && [ "${ascend_lib}" != "${ascend_arch_lib}" ]; then
        ascend_lib_extra="${ascend_arch_lib}"
    fi

    # 聚合所有vendor目录的 include 与 lib 路径
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
        done < <(_example_paths aclnn "${op}" .cpp)

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
                "${lib_dir}/libcust_opapi.so" \
                -lascendcl -lnnopbase \
                -o "${example_bin}"

            echo "[INFO] Running aclnn example: ${example_name}..."
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
        done < <(_example_paths torch "${op}" .py)

        if [ ${#example_files[@]} -eq 0 ]; then
            echo "[WARN] No torch example found for operator: ${op}, skipping."
            continue
        fi

        for example_file in "${example_files[@]}"; do
            local example_name
            example_name=$(basename "${example_file}" .py)

            echo "[INFO] Running torch example: ${example_name} (${op})..."
            python3 "${example_file}"

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
}

main "$@"
