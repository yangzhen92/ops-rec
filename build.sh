#!/usr/bin/env bash
# ============================================================================
# 用法：
#   bash build.sh
#   bash build.sh --socs atlas_950 --install
# ============================================================================

set -e

# ---------------------------------------------------------------------------
# 默认值
# ---------------------------------------------------------------------------
INSTALL_MODE=false
JOBS=""  # -j：并行编译的算子数（空 = 自动推导）
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
                echo "[ERROR] -j requires a thread number (e.g. -j6)" >&2
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
  -j[N]                     Number of operators built in parallel (default:
                            derived from CPU/memory). Each op also runs an inner
                            make -j = clamp(agg_target / N, 1, 64), so -j1 builds
                            one op at a time but still uses many parallel jobs.
  --socs <list>             SOC variants to build, comma-separated
                            eg: --socs atlas_950
                                --socs atlas_a2,atlas_a3,atlas_950 (default: all)
  --ops <list>              op vendors to build, comma-separated (default: all)
                            Values are the CANN vendor package names
                            eg: --ops ln_mul,disentangle_attention

Environment variables:
  ASCEND_HOME_PATH          CANN install path (default: /usr/local/Ascend/latest)
  OPS_REC_FORCE_VARIANT     Force runtime SOC variant (for testing)
  CMAKE_GENERATOR           CMake generator (default: Unix Makefiles)

Examples:
  bash run_examples.sh      Run operator examples
  bash build.sh
  bash build.sh --install
EOF
            exit 0
            ;;
        *)
            echo "[WARN] Unknown argument: $1 (ignored)"
            shift
            ;;
    esac
done

# 校验 -j 取值：必须为正整数，避免 -j0 / -jabc 在算术里除以 0
if [ -n "${JOBS}" ] && ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] -j requires a positive integer (got '${JOBS}')" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# 环境初始化与检查
# ---------------------------------------------------------------------------
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/latest}"

# 并发模型：滚动任务池 + 聚合核数封顶（详见 main()）。
#   - 所有 (soc, op) 任务（跨 SOC）投进同一个滚动池，最多同时跑 JOBS 个算子，
#     任一结束立即补下一个，不再整批等待，避免被最慢算子卡住以致其余核空转。
#   - JOBS = 并行编译的算子数（-jN 指定，缺省由 agg_target / per_op_base 推出）。
#   - 每算子内层 make -j = agg_target / JOBS，上限 64（防单算子抢占过多）。

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
echo "  - Op Vendors          : ${OPS:-<all>}"
echo "  - Ascend Home         : ${ASCEND_HOME_PATH}"
echo "  - Concurrent op builds: ${JOBS:-auto (derived in main)}"
echo "  - CMake Generator     : ${CMAKE_GENERATOR}"
echo "==========================================================="

# ---------------------------------------------------------------------------
# 工具函数：查询某 SOC/op 的"真实 vendor"清单（每行一个）。
# ---------------------------------------------------------------------------
op_leaves_for_soc() {
    local soc="$1"
    local op="$2"
    cmake -DSOC="${soc}" -DOPS_REC_QUERY_OP="${op}" \
        -P "${SCRIPT_DIR}/cmake/soc_ops_config.cmake" 2>&1
}

# ---------------------------------------------------------------------------
# 工具函数：某真实 vendor 是否存在于源码树（顶层 src/ascendc/<g>/<vendor>
# 或其下嵌套的聚合子目录 src/ascendc/<g>/<agg>/<vendor>）。
# ---------------------------------------------------------------------------
vendor_dir_exists() {
    local op="$1"
    local found=""
    local g
    for g in "${SCRIPT_DIR}"/src/ascendc/*; do
        [ -d "${g}/${op}" ] && found="1" && break
        local a
        for a in "${g}"/*; do
            [ -d "${a}/${op}" ] && found="1" && break
        done
        [ -n "${found}" ] && break
    done
    [ -n "${found}" ]
}

# ---------------------------------------------------------------------------
# 工具函数：生成公共 cmake 参数。
# 拆成两部分：
#   CMAKEOP_COMMON  —— 逐算子 pass 与适配层 pass 共用（kernel/opapi 编译所需，
#                      不含 torch/python 变量，避免 configure 时报"set but not used"）。
#   CMAKEOP_ADAPTER —— 适配层 pass 专属（需要 find_package(Python3)/torch/ABI 宏）。
# ---------------------------------------------------------------------------
CMAKEOP_COMMON=()
CMAKEOP_ADAPTER=()
init_common_cmake_args() {
    local pyexe cmake_prefix abi
    pyexe=$(which python3)
    cmake_prefix=$(dirname "$(python3 -c "import torch; print(torch.__file__)")")
    abi=$(python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
    CMAKEOP_COMMON=(
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "-DCMAKE_CXX_FLAGS=-fstack-protector-strong"
        "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-z,relro -Wl,-z,now -Wl,-s"
        "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,relro -Wl,-z,now -Wl,-s"
        "-DCMAKE_SKIP_BUILD_RPATH=TRUE"
        "-DCMAKE_SKIP_INSTALL_RPATH=TRUE"
    )
    CMAKEOP_ADAPTER=(
        "-DPython3_EXECUTABLE=${pyexe}"
        "-DCMAKE_PREFIX_PATH=${cmake_prefix}"
        "-D_GLIBCXX_USE_CXX11_ABI=${abi}"
    )
}

# ---------------------------------------------------------------------------
# 工具函数：同步各算子的 Python autograd 源码 -> 打包 merge 目录 ops_rec/autograd/
# 源码约定位于每个算子的 torch_extension/ 目录（与 *_plugin.cpp 同处），
# 打包阶段拷贝到 merge 树 ops_rec/autograd/ 子目录（源码树保留 __init__.py 作为
# 包入口，不写拷贝物，避免 build 后源码树被污染进 git status）。
# ---------------------------------------------------------------------------
sync_autograd() {
    local dst="${MERGE_ROOT}/autograd"
    # 服务端 __init__.py 是包加载入口，不拷贝（merge 树里由源码树 cp 提供）。
    # 同名冲突检测：多个算子提供同名 *_autograd.py 会在包内彼此覆盖，
    # 导致注册错误模块。先按 basename 去重，发现重复立即失败。
    local f name
    local -A seen=()
    while IFS= read -r -d '' f; do
        name="$(basename "${f}")"
        if [[ -n "${seen[${name}]}" ]]; then
            die "autograd filename collision: '${f#${SCRIPT_DIR}/}' vs '${seen[${name}]#${SCRIPT_DIR}/}' (same basename '${name}')"
        fi
        seen["${name}"]="${f}"
    done < <(find "${SCRIPT_DIR}/src/ascendc" -type f -name '*_autograd.py' -path '*/torch_extension/*' -print0)

    mkdir -p "${dst}"
    find "${dst}" -maxdepth 1 -name '*_autograd.py' -type f -delete

    for f in "${seen[@]}"; do
        cp -a "${f}" "${dst}/$(basename "${f}")"
        echo "[INFO] Synced ${f#${SCRIPT_DIR}/} -> ops_rec/autograd/$(basename "${f}")"
    done
}

# ---------------------------------------------------------------------------
# 工具函数：某真实 vendor 是否以 CANN RUN 模式打包（源码 CMakeLists 中
# npu_op_package(... TYPE RUN)）。RUN 与 SHARED 的 install 结构不同：
# RUN 装 packages/vendors/<v>/op_impl，SHARED 走 opp/vendors/<v>/op_impl。
# 返回 0=是 RUN；1=非 RUN（SHARED）。
# ---------------------------------------------------------------------------
op_is_run_type() {
    local op="$1"
    local f
    while IFS= read -r f; do
        if grep -q "TYPE[[:space:]]*RUN" "${f}" 2>/dev/null; then
            return 0
        fi
    done < <(find "${SCRIPT_DIR}/src/ascendc" -name CMakeLists.txt \
                -path "*/${op}/CMakeLists.txt" 2>/dev/null)
    return 1
}

# 打印错误并以非零退出（替代隐式依赖 set -e，确保失败被显式传播）。
die() {
    echo "[ERROR] $*" >&2
    exit 1
}

# 滚动任务池：同时最多跑 maxjobs 个后台任务，任一结束立即补充池中下一个任务
# （不再整批等待，避免被最慢算子卡住整条流水线）。参数为 (soc\top) 对。
# 任一任务失败立即终止，并回收其余残留后台任务，避免遗留孤儿 make / 半成品 staging。
run_rolling_pool() {
    local maxjobs="$1"
    shift
    [ "${maxjobs}" -lt 1 ] && maxjobs=1
    local task soc op
    for task in "$@"; do
        while [ "$(jobs -rp | wc -l)" -ge "${maxjobs}" ]; do
            if ! wait -n; then
                # shellcheck disable=SC2046
                kill $(jobs -rp) 2>/dev/null || true
                die "parallel op build failed (rolling pool)"
            fi
        done
        soc="${task%%$'\t'*}"
        op="${task#*$'\t'}"
        build_one_kernel "${soc}" "${op}" &
    done
    # 排空：剩余任务也逐个检查退出码
    while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
        if ! wait -n; then
            # shellcheck disable=SC2046
            kill $(jobs -rp) 2>/dev/null || true
            die "parallel op build failed (rolling pool drain)"
        fi
    done
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
    local _t0 _t1
    _t0="$(date +%s)"

    echo ""
    echo "-----------------------------------------------------------"
    echo "[INFO] Building ${soc}/${op} (start $(date '+%H:%M:%S'))"
    echo "-----------------------------------------------------------"

    rm -rf "${bdir}" "${stage}"

    env OPS_REC_SOC_VARIANT="${soc}" OPS_REC_TARGET_OP="${op}" OPS_REC_SKIP_TORCH_PLUGIN=ON \
        cmake -S "${SCRIPT_DIR}" -B "${bdir}" \
            "${CMAKEOP_COMMON[@]}" -DCMAKE_INSTALL_PREFIX="${stage}" || return 1

    cmake --build "${bdir}" -j"${INNER_JOBS}" || return 1

    if op_is_run_type "${op}"; then
        echo "[INFO] ${soc}/${op}: building 'binary' hub target (RUN kernel)"
        cmake --build "${bdir}" --target binary -j"${INNER_JOBS}" || return 1
    fi

    cmake --install "${bdir}" || return 1

    if op_is_run_type "${op}"; then
        local vendors_pkg="${stage}/packages/vendors"
        if [ ! -d "${vendors_pkg}" ] \
           || ! find "${vendors_pkg}" -type d \
               -path '*op_impl/ai_core/tbe/op_tiling' 2>/dev/null | grep -q .; then
            echo "[ERROR] ${soc}/${op}: RUN kernel op_impl missing after install " \
                 "('binary' target build failed or was not run)" >&2
            return 1
        fi
    fi

    _t1="$(date +%s)"
    echo "[INFO] Built ${soc}/${op} -> ${stage}"
    echo "[TIME] ${soc}/${op} done in $(( _t1 - _t0 ))s"
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
            "${CMAKEOP_COMMON[@]}" "${CMAKEOP_ADAPTER[@]}" -DCMAKE_INSTALL_PREFIX="${stage}"

    cmake --build "${bdir}" -j"${INNER_JOBS}"
    cmake --install "${bdir}"

    echo "[INFO] Built adapter for ${soc} -> ${stage}"
}

# ---------------------------------------------------------------------------
# 工具函数：打包前校验每个目标 (SOC, op) 的 vendor 产物确实已合并进
# dist_merge/ops_rec/opp/<variant>/vendors/ 目录树。
# 用 find 跨 variant 名（atlas_950 / atlas_a2 可能不同）逐 op 确认存在，
# 任一缺失即报错，避免"编译成功但打包缺失/空 whl 却提示成功"。
# ---------------------------------------------------------------------------
verify_merged_ops() {
    local missing=0
    local soc leaf
    while IFS=$'\t' read -r soc leaf; do
        [ -z "${soc}" ] && continue
        [ -z "${leaf}" ] && continue
        if ! find "${MERGE_ROOT}/opp" -type d -name "${leaf}" 2>/dev/null | grep -q .; then
            echo "[ERROR] ${soc}/${leaf}: vendor dir missing from merged opp/ tree" >&2
            missing=1
        fi
    done <<< "$1"
    if [ "${missing}" -eq 1 ]; then
        die "some vendor artifacts missing from merged package; aborting wheel build"
    fi
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

    # 累积"实际构建"的 (soc leaf) 对，供 verify_merged_ops 校验
    local -a BUILT_OPS=()

    # ------------------------------------------------------------------
    # 全局并发参数（滚动任务池跨 SOC 共用一套）：
    #   agg_target  —— 聚合 make job 上限 = min(3/4 总核数, 内存可支撑 job 数)，
    #                  内存按 ~0.5GiB/job 估算；不设固定硬上值，随机器自适应。
    #   per_op_base —— 推导并行算子数用的每算子基值 = clamp(agg_target/6, 8, 32)。
    #   JOBS        —— 并行编译的算子数：-jN 指定，缺省 = max(agg_target/per_op_base, 1)，
    #                  与算子总数无关（多了只排队）。
    #   inner       —— 每算子内层 make -j = clamp(agg_target/JOBS, 1, 64)。
    # ------------------------------------------------------------------
    local total_cpus cpu_agg mem_jobs agg_target per_op_base inner mem_kb
    total_cpus="$(nproc 2>/dev/null || echo 4)"
    # 可用内存(KB)：优先 MemAvailable(kernel>=3.14)，否则 MemFree+Buffers+Cached 近似；
    # 都取不到则告警，并按"内存未知"跳过内存封顶（只用 CPU 推导的预算）
    mem_kb="$(awk '/^MemAvailable:/{print $2; exit}' /proc/meminfo 2>/dev/null)"
    if [ -z "${mem_kb}" ]; then
        mem_kb="$(awk '/^MemFree:/{f=$2} /^Buffers:/{b=$2} /^Cached:/{c=$2} END{if(f!="")print f+b+c}' \
                     /proc/meminfo 2>/dev/null)"
        [ -n "${mem_kb}" ] && echo "[WARN] /proc/meminfo lacks MemAvailable; using MemFree+Buffers+Cached=${mem_kb}kB for job sizing" >&2
    fi
    mem_jobs=0                                    # 0 = 内存未知，不参与封顶
    if [ -n "${mem_kb}" ]; then
        mem_jobs=$(( mem_kb / 1024 / 1024 * 2 ))  # ~0.5 GiB / 并行 job 的保守估计
        [ "${mem_jobs}" -lt 1 ] && mem_jobs=1
    else
        echo "[WARN] cannot read memory from /proc/meminfo; skipping memory-based job cap" >&2
    fi
    cpu_agg=$(( total_cpus * 3 / 4 ))
    [ "${cpu_agg}" -lt 1 ] && cpu_agg=1
    agg_target="${cpu_agg}"
    if [ "${mem_jobs}" -gt 0 ] && [ "${mem_jobs}" -lt "${agg_target}" ]; then
        agg_target="${mem_jobs}"
    fi
    # 每算子基值：按"设计目标 ≈6 个并行算子"推导，clamp [8,32]。
    per_op_base=$(( agg_target / 6 ))
    [ "${per_op_base}" -lt 8 ] && per_op_base=8
    [ "${per_op_base}" -gt 32 ] && per_op_base=32
    if [ -z "${JOBS}" ]; then
        JOBS=$(( agg_target / per_op_base ))
        [ "${JOBS}" -lt 1 ] && JOBS=1
    fi
    inner=$(( agg_target / JOBS ))
    [ "${inner}" -lt 1 ] && inner=1
    [ "${inner}" -gt 64 ] && inner=64
    INNER_JOBS="${inner}"

    # ------------------------------------------------------------------
    # 第一遍：按 SOC 组装"真实 vendor"编译清单（每个 vendor 独立 configure），
    # 收集成全局 (soc, op) 任务列表。各 task 写入隔离的 build/<soc>/ops/<op> 与
    # staging 目录，无共享写冲突，因此可跨 SOC 投入同一个滚动池并行编译，
    # 让机器全程跑满。
    # ------------------------------------------------------------------
    local -A SOC_OP_LISTS=()
    local -a TASKS=()
    IFS=',' read -ra SOC_ARRAY <<< "${SOCS}"
    # 规整 SOC 列表：去空白、去空项、去重（保序），避免重复 SOC 造成任务/适配层重复构建
    local -A _soc_seen=()
    local -a _soc_uniq=()
    for soc in "${SOC_ARRAY[@]}"; do
        soc="$(echo "${soc}" | xargs)"
        [ -z "${soc}" ] && continue
        [ -n "${_soc_seen[${soc}]:-}" ] && continue
        _soc_seen["${soc}"]=1
        _soc_uniq+=("${soc}")
    done
    SOC_ARRAY=("${_soc_uniq[@]}")

    for soc in "${SOC_ARRAY[@]}"; do

        # 组装该 SOC 的"真实 vendor"编译清单（每个 vendor 独立 configure）。
        local -a op_list=()
        local op
        if [ "${op_mode}" = "all" ]; then
            while IFS= read -r raw_op; do
                op="$(echo "${raw_op}" | xargs)"
                [ -z "${op}" ] && continue
                vendor_dir_exists "${op}" && op_list+=("${op}")
            done < <(cmake -DSOC="${soc}" -DOPS_REC_LIST_MODE=leaf \
                        -P "${SCRIPT_DIR}/cmake/soc_ops_config.cmake" 2>&1)
        else
            # 预先获取该 SOC 的叶子白名单（cmake leaf 模式），用于过滤
            # 当 --ops 指定的 op 不在某 SOC 的白名单中时，跳过该 SOC
            local -a soc_leaves=()
            while IFS= read -r raw_leaf; do
                local leaf_trim
                leaf_trim="$(echo "${raw_leaf}" | xargs)"
                [ -n "${leaf_trim}" ] && soc_leaves+=("${leaf_trim}")
            done < <(cmake -DSOC="${soc}" -DOPS_REC_LIST_MODE=leaf \
                        -P "${SCRIPT_DIR}/cmake/soc_ops_config.cmake" 2>&1)

            for op in "${OP_ARRAY[@]}"; do
                [ -z "${op}" ] && continue
                if ! vendor_dir_exists "${op}"; then
                    die "unknown op/vendor '${op}' (not in src/ascendc/ tree); refusing to build"
                fi
                local matched=0
                local sl
                for sl in "${soc_leaves[@]}"; do
                    if [ "${sl}" = "${op}" ]; then
                        op_list+=("${sl}")
                        matched=1
                    fi
                done
                if [ "${matched}" -eq 0 ]; then
                    while IFS= read -r leaf; do
                        [ -n "${leaf}" ] || continue
                        for sl in "${soc_leaves[@]}"; do
                            if [ "${sl}" = "${leaf}" ]; then
                                op_list+=("${leaf}")
                                break
                            fi
                        done
                    done < <(op_leaves_for_soc "${soc}" "${op}")
                fi
            done
        fi
        if [ "${#op_list[@]}" -eq 0 ]; then
            if [ "${op_mode}" = "specified" ]; then
                echo "[INFO] ${soc}: no matching ops from --ops filter; skipping"
                continue
            fi
            die "no ops to build for SOC ${soc}; refusing to produce an empty wheel"
        fi
        # 去重（同一 leaf 可能被聚合名展开与直接指定重复加入）
        local -A _op_seen=()
        local -a _op_uniq=()
        for op in "${op_list[@]}"; do
            [ -n "${_op_seen[${op}]:-}" ] && continue
            _op_seen["${op}"]=1
            _op_uniq+=("${op}")
        done
        op_list=("${_op_uniq[@]}")
        SOC_OP_LISTS["${soc}"]="$(IFS=$'\n'; echo "${op_list[*]}")"
        for op in "${op_list[@]}"; do
            TASKS+=("${soc}"$'\t'"${op}")
        done
    done

    if [ "${#TASKS[@]}" -eq 0 ]; then
        die "no ops to build for any SOC; refusing to produce an empty wheel"
    fi
    echo "[INFO] ${#TASKS[@]} op builds total, ${JOBS} parallel ops, inner make -j${inner} (agg_target ${agg_target})"

    # 1) 跨 SOC 全局滚动池编译全部 (soc, op) kernel（任一完成立即补下一任务）
    run_rolling_pool "${JOBS}" "${TASKS[@]}"

    # 2) 逐 SOC 串行收集各 op staging -> 共享 merge 目录
    for soc in "${SOC_ARRAY[@]}"; do
        [ -z "${SOC_OP_LISTS[${soc}]:-}" ] && continue

        local -a op_list=()
        mapfile -t op_list <<< "${SOC_OP_LISTS[${soc}]}"

        echo ""
        echo "-----------------------------------------------------------"
        echo "[INFO] Collecting kernel artifacts for ${soc}"
        echo "-----------------------------------------------------------"
        for op in "${op_list[@]}"; do
            BUILT_OPS+=("${soc}	${op}")
            local sstage="${STAGE_ROOT}/${soc}/${op}"
            # 该 SOC 的 OPP 归档目录（variant 目录名由 CMake 确定，可能是 atlas_950 / atlas_a2 / atlas_a3）
            local oppdir
            oppdir=$(find "${sstage}/opp" -maxdepth 2 -type d -name 'vendors' 2>/dev/null | head -1)
            if [ -z "${oppdir}" ]; then
                die "${soc}/${op}: build succeeded but no opp/vendors staged; refusing to produce a wheel missing this op"
            fi
            local variant=$(basename "$(dirname "${oppdir}")")
            local collected=0
            for vend in "${oppdir}"/*/; do
                [ -d "${vend}" ] || continue
                local vname
                vname="$(basename "${vend}")"
                local vend_pkg="${MERGE_ROOT}/opp/${variant}/vendors/${vname}"
                mkdir -p "${vend_pkg}"
                cp -a "${vend}/." "${vend_pkg}/"
                echo "[INFO] Collected ${soc}/${op} -> ${variant}/vendors/${vname}"
                collected=1

                local vdir="${sstage}/packages/vendors/${vname}"
                if [ -d "${vdir}" ]; then
                    local sub
                    for sub in op_api op_proto op_impl/ai_core/tbe/op_tiling; do
                        if [ -d "${vdir}/${sub}" ]; then
                            cp -a "${vdir}/${sub}/." "${vend_pkg}/${sub}/" 2>/dev/null
                            echo "[INFO] Collected ${soc}/${op} ${sub} -> ${variant}/vendors/${vname}/${sub}"
                        fi
                    done
                    # RUN 模式（产出的 packages/vendors 目录）下，op_api（aclnn 头 +
                    # libcust_opapi.so）由框架只装到 packages/vendors/<vendor>/op_api，
                    # 是 aclnn/torch 适配层 dlopen 的必需产物；缺失说明 install 阶段
                    # 没把 vendor 的 op_api 装上，直接失败，避免 wheel 打包成功却缺
                    # op_api、运行时才报错且难以定位。
                    if [ ! -d "${vdir}/op_api/lib" ] \
                       || [ ! -f "${vdir}/op_api/lib/libcust_opapi.so" ]; then
                        die "${soc}/${op}: RUN package missing op_api/lib/libcust_opapi.so " \
                            "under packages/vendors/${vname}; refusing to package a wheel missing op_api"
                    fi
                fi
            done
            if [ "${collected}" -eq 0 ]; then
                die "${soc}/${op}: no vendor dirs under ${oppdir}; refusing to produce a wheel missing this op"
            fi

            # config.ini 取任一个 op 的（内容为该 SOC 全量 load_priority，各 op 一致）
            local cfg="${oppdir}/config.ini"
            if [ -f "${cfg}" ]; then
                mkdir -p "${MERGE_ROOT}/opp/${variant}/vendors"
                cp -f "${cfg}" "${MERGE_ROOT}/opp/${variant}/vendors/config.ini"
            fi
        done

        # 3) 单次构建适配层并收集 ops_rec_atlas_<variant>.so（适配层 install 到 stage 根目录）
        build_adapter "${soc}"
        local so=$(find "${STAGE_ROOT}/${soc}/adapter" -maxdepth 1 -name 'ops_rec_atlas_*.so' 2>/dev/null | head -1)
        if [ -n "${so}" ]; then
            cp -f "${so}" "${MERGE_ROOT}/$(basename "${so}")"
            echo "[INFO] Collected $(basename "${so}")"
        else
            die "adapter .so not produced for ${soc}; refusing to build"
        fi
    done

    verify_merged_ops "$(IFS=$'\n'; echo "${BUILT_OPS[*]}")"

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

    # 5) 同步各算子 torch_extension/*_autograd.py -> merge 树 ops_rec/autograd
    sync_autograd

    # 6) 清理打包树中的 Python 缓存（__pycache__ / *.pyc）。
    find "${MERGE_ROOT}" -type d -name '__pycache__' -prune -exec rm -rf {} +
    find "${MERGE_ROOT}" -type f -name '*.pyc' -delete

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
