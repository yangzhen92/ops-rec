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
OPS=""  # 可选：指定要构建的 vendor 名列表（逗号分隔）。为空 = 构建该 SOC 全部 vendor

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
            if [ -z "${2:-}" ]; then
                echo "[ERROR] -j requires a thread number (e.g. -j16)" >&2
                exit 2
            fi
            JOBS="$2"
            shift 2
            ;;
        -j*)
            JOBS="${1#-j}"
            shift
            ;;
        --socs)
            if [ -z "${2:-}" ]; then
                echo "[ERROR] --socs requires a value (e.g. --socs atlas_950)" >&2
                exit 2
            fi
            SOCS="$2"
            shift 2
            ;;
        --socs=*)
            SOCS="${1#*=}"
            shift
            ;;
        --ops)
            if [ -z "${2:-}" ]; then
                echo "[ERROR] --ops requires a value (e.g. --ops ln_mul,disentangle_attention)" >&2
                exit 2
            fi
            OPS="$2"
            shift 2
            ;;
        --ops=*)
            OPS="${1#*=}"
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: bash build.sh [flags]

Flags:
  --install                 Compile and then install the merged whl
  -j[N]                     Wall-clock concurrency: max parallel op builds
                            (default: number of CPU cores)
  --socs <list>             SOC variants to build, comma-separated
                            eg: --socs atlas_950
                                --socs atlas_a2,atlas_a3,atlas_950 (default: all)
  --ops <list>              op vendors to build, comma-separated (default: all)
                            Values are the CANN vendor package names
                            eg: --ops ln_mul,disentangle_attention

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
    OUTER_PARALLEL="$JOBS"
else
    OUTER_PARALLEL="$(nproc 2>/dev/null || echo 4)"
fi

# CANN 的 npu_op_kernel_library 生成 $(MAKE) 引用，与 Ninja 不兼容
export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Unix Makefiles}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "${SCRIPT_DIR}" || exit 1

MERGE_ROOT="${SCRIPT_DIR}/dist_merge/ops_rec"      # 最终包内容（merge_wheel 的 merge-dir 根）
STAGE_ROOT="${SCRIPT_DIR}/dist_merge/stage"        # 各算子/适配层独立 staging

echo "==========================================================="
echo "[INFO] Build Configuration:"
echo "  - Install Mode        : ${INSTALL_MODE}"
echo "  - SOCs                : ${SOCS}"
echo "  - op vendors         : ${OPS:-<all>}"
echo "  - Ascend Home         : ${ASCEND_HOME_PATH}"
echo "  - Parallel (outer)    : ${OUTER_PARALLEL}"
echo "  - CMake Generator     : ${CMAKE_GENERATOR}"
echo "==========================================================="

# ---------------------------------------------------------------------------
# 工具函数：获取某 SOC 的算子清单。
#   cmake -DSOC=atlas_950 -P cmake/soc_ops_config.cmake
# ---------------------------------------------------------------------------
ops_for_soc() {
    local soc="$1"
    cmake -DSOC="${soc}" -P "${SCRIPT_DIR}/cmake/soc_ops_config.cmake" 2>&1
}

# ---------------------------------------------------------------------------
# 工具函数：生成公共 cmake 参数（适配 torch / CANN 通用 flags）。
# ---------------------------------------------------------------------------
CMAKECOMMON=()
init_common_cmake_args() {
    local pyexe cmake_prefix abi
    pyexe=$(which python3)
    cmake_prefix=$(dirname "$(python3 -c "import torch; print(torch.__file__)")")
    abi=$(python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
    CMAKECOMMON=(
        "-DPython3_EXECUTABLE=${pyexe}"
        "-DCMAKE_PREFIX_PATH=${cmake_prefix}"
        "-D_GLIBCXX_USE_CXX11_ABI=${abi}"
        "-DCMAKE_C_FLAGS=-fstack-protector-strong"
        "-DCMAKE_CXX_FLAGS=-fstack-protector-strong"
        "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-z,relro -Wl,-z,now"
        "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,relro -Wl,-z,now"
        "-DCMAKE_SKIP_BUILD_RPATH=FALSE"
        "-DCMAKE_SKIP_INSTALL_RPATH=FALSE"
        "-DCMAKE_TOOLCHAIN_FILE=${SCRIPT_DIR}/cmake/toolchain_ascend.cmake"
    )
}

# ---------------------------------------------------------------------------
# 工具函数：同步 experimental/triton/ -> ops_rec/experimental/triton/
# 仅拷贝 .py 文件（不含 test/、README.md 等），清除非 .py 文件。
# ---------------------------------------------------------------------------
sync_triton() {
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
}

# ---------------------------------------------------------------------------
# 工具函数：并行构建单个 (SOC, vendor) 的 kernel/opapi 产物。
# 每个算子独立 configure（OPS_REC_TARGET_OP 过滤 + OPS_REC_SKIP_TORCH_PLUGIN），
# install 到独立 staging，避免并行冲突。内部 make 用 INNER_JOBS 线程。
# ---------------------------------------------------------------------------
build_one_kernel() {
    local soc="$1"
    local op="$2"
    local bdir="${SCRIPT_DIR}/build/${soc}/ops/${op}"
    local stage="${STAGE_ROOT}/${soc}/${op}"

    echo ""
    echo "-----------------------------------------------------------"
    echo "[INFO] Building ${soc}/${op}"
    echo "-----------------------------------------------------------"

    rm -rf "${bdir}" "${stage}"

    env OPS_REC_SOC_VARIANT="${soc}" OPS_REC_TARGET_OP="${op}" OPS_REC_SKIP_TORCH_PLUGIN=ON \
        cmake -S "${SCRIPT_DIR}" -B "${bdir}" \
            "${CMAKECOMMON[@]}" -DCMAKE_INSTALL_PREFIX="${stage}" || return 1

    cmake --build "${bdir}" -j"${INNER_JOBS}" || return 1
    cmake --install "${bdir}" || return 1

    echo "[INFO] Built ${soc}/${op} -> ${stage}"
}

# ---------------------------------------------------------------------------
# 工具函数：单次构建某 SOC 的 Torch 适配层 .so（OPS_REC_SKIP_OPS=ON，只编适配层）。
# ---------------------------------------------------------------------------
build_adapter() {
    local soc="$1"
    local bdir="${SCRIPT_DIR}/build/${soc}/adapter"
    local stage="${STAGE_ROOT}/${soc}/adapter"

    echo ""
    echo "-----------------------------------------------------------"
    echo "[INFO] Building adapter for ${soc}"
    echo "-----------------------------------------------------------"

    rm -rf "${bdir}" "${stage}"

    env OPS_REC_SOC_VARIANT="${soc}" OPS_REC_SKIP_OPS=ON \
        cmake -S "${SCRIPT_DIR}" -B "${bdir}" \
            "${CMAKECOMMON[@]}" -DCMAKE_INSTALL_PREFIX="${stage}"

    cmake --build "${bdir}" -j"${INNER_JOBS}"
    cmake --install "${bdir}"

    echo "[INFO] Built adapter for ${soc} -> ${stage}"
}

# ---------------------------------------------------------------------------
# 主流程执行
# ---------------------------------------------------------------------------
main() {
    # 清理历史残余目录
    rm -rf dist_merge/ dist/ build/
    mkdir -p "${MERGE_ROOT}"

    # 预计算公共 cmake 参数（每个子进程共享同一数组定义）
    init_common_cmake_args

    # 解析 --ops 参数
    local op_mode="all"
    local -a OP_ARRAY=()
    if [ -n "${OPS}" ]; then
        op_mode="specified"
        IFS=',' read -ra OP_ARRAY <<< "${OPS}"
        for i in "${!OP_ARRAY[@]}"; do
            OP_ARRAY[$i]="$(echo "${OP_ARRAY[$i]}" | xargs)"
        done
    fi

    # 同步 experimental/triton -> ops_rec（每次构建前做一次）
    sync_triton

    IFS=',' read -ra SOC_ARRAY <<< "${SOCS}"
    for soc in "${SOC_ARRAY[@]}"; do
        soc="$(echo "${soc}" | xargs)"
        [ -z "${soc}" ] && continue

        # 组装该 SOC 的算子清单。
        local -a op_list=()
        local op
        if [ "${op_mode}" = "all" ]; then
            while IFS= read -r raw_op; do
                op="$(echo "${raw_op}" | xargs)"
                [ -z "${op}" ] && continue
                # 匹配一条真实存在于 ascendc/ 下的 vendor 路径即可
                local found=""
                while IFS= read -r c; do
                    [ -z "${c}" ] && continue
                    [ -d "${SCRIPT_DIR}/ascendc/${c}/${op}" ] && found="1" && break
                done < <(ls "${SCRIPT_DIR}/ascendc/" 2>/dev/null)
                [ -n "${found}" ] && op_list+=("${op}")
            done < <(ops_for_soc "${soc}")
        else
            for op in "${OP_ARRAY[@]}"; do
                [ -z "${op}" ] && continue
                [ -d "${SCRIPT_DIR}/ascendc" ] || continue
                if ls -d "${SCRIPT_DIR}"/ascendc/*/"${op}" >/dev/null 2>&1; then
                    op_list+=("${op}")
                else
                    echo "[WARN] unknown op '${op}' (no ascendc/*/${op} dir); skipping"
                fi
            done
        fi
        if [ "${#op_list[@]}" -eq 0 ]; then
            echo "[WARN] No ops to build for SOC ${soc}; skipping"
            continue
        fi

        # 每个算子内部 make 的线程数 = 总核数 / 外层并行度（下限 1，上限 每个外层至少 2 核）
        local total_cpus outer inner
        total_cpus="$(nproc 2>/dev/null || echo 4)"
        outer="${#op_list[@]}"
        if [ "${outer}" -gt "${OUTER_PARALLEL}" ]; then
            outer="${OUTER_PARALLEL}"
        fi
        inner=$(( total_cpus / outer ))
        [ "${inner}" -lt 1 ] && inner=1
        INNER_JOBS="${inner}"
        echo "[INFO] ${soc}: ${#op_list[@]} ops, outer=${outer}, inner make -j${inner}"

        # 1) 并行逐算子编译（各自独立 configure + 独立 staging）
        local -a pids=()
        for op in "${op_list[@]}"; do
            build_one_kernel "${soc}" "${op}" &
            pids+=("$!")
            # 控制并发度：到达 OUTER_PARALLEL 个后台任务时等待全部完成
            if [ "${#pids[@]}" -ge "${OUTER_PARALLEL}" ]; then
                for pid in "${pids[@]}"; do
                    wait "${pid}"
                done
                pids=()
            fi
        done
        for pid in "${pids[@]}"; do
            wait "${pid}"
        done

        # 2) 串行收集各 op staging -> 共享 merge 目录（安全的纯 copy）
        echo ""
        echo "-----------------------------------------------------------"
        echo "[INFO] Collecting kernel artifacts for ${soc}"
        echo "-----------------------------------------------------------"
        for op in "${op_list[@]}"; do
            local sstage="${STAGE_ROOT}/${soc}/${op}"
            # 该 SOC 的 OPP 归档目录（variant 目录名由 CMake 确定，可能是 atlas_950 / atlasA2）
            local oppdir
            oppdir=$(find "${sstage}/opp" -maxdepth 2 -type d -name 'vendors' 2>/dev/null | head -1)
            if [ -z "${oppdir}" ]; then
                echo "[WARN] ${soc}/${op}: no opp/vendors staged; skipping collect"
                continue
            fi
            local variant
            variant=$(basename "$(dirname "${oppdir}")")
            local vend_pkg="${MERGE_ROOT}/opp/${variant}/vendors/${op}"
            mkdir -p "${vend_pkg}"
            cp -a "${oppdir}/${op}/." "${vend_pkg}/"

            # config.ini 取任一个 op 的（内容为该 SOC 全量 load_priority，各 op 一致）
            local cfg="${oppdir}/config.ini"
            if [ -f "${cfg}" ]; then
                mkdir -p "${MERGE_ROOT}/opp/${variant}/vendors"
                cp -f "${cfg}" "${MERGE_ROOT}/opp/${variant}/vendors/config.ini"
            fi
            echo "[INFO] Collected ${soc}/${op} -> ${variant}/vendors/${op}"
        done

        # 3) 单次构建适配层并收集
        build_adapter "${soc}"
        # 收集 ops_rec_atlas_<variant>.so（适配层 install 到 stage 根目录）
        local so
        so=$(find "${STAGE_ROOT}/${soc}/adapter" -maxdepth 1 -name 'ops_rec_atlas_*.so' 2>/dev/null | head -1)
        if [ -n "${so}" ]; then
            cp -f "${so}" "${MERGE_ROOT}/$(basename "${so}")"
            echo "[INFO] Collected $(basename "${so}")"
        else
            echo "[WARN] adapter .so not found for ${soc}"
        fi
    done

    # 4) 收集 Python 包树（含 triton、env_setup.sh）——排除工具链生成的 opp/
    echo ""
    echo "-----------------------------------------------------------"
    echo "[INFO] Copying python package tree"
    echo "-----------------------------------------------------------"
    for item in "${SCRIPT_DIR}"/ops_rec/*; do
        [ -e "${item}" ] || continue
        local name
        name="$(basename "${item}")"
        # 排除：工具链生成的 opp/（由算子/适配层 install 单独生成）、编译缓存、
        # 以及历史遗留的适配层 .so（由适配层 pass 重新生成）。
        if [ "${name}" = "opp" ] || [ "${name}" = "__pycache__" ] \
           || [[ "${name}" == ops_rec_atlas_*.so ]]; then
            continue
        fi
        cp -a "${item}" "${MERGE_ROOT}/${name}"
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
        --merge-dir "${MERGE_ROOT}" \
        --out-dir "${SCRIPT_DIR}/dist" \
        --version "${torch_ver}"

    # 清理中间过渡目录
    rm -rf dist_merge/

    # 检查最终 wheel 是否创建成功
    local final_whl
    final_whl=$(ls dist/ops_rec-*.whl 2>/dev/null | head -1)
    if [ -z "${final_whl}" ]; then
        echo "[ERROR] Build failed: no wheel file found in dist/"
        exit 1
    fi

    echo "[INFO] Build successful: ${final_whl}"

    # 5. 如果开启了 --install 模式，则自动安装
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

main "$@"

