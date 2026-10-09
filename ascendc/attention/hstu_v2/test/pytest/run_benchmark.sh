#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR=${HSTU_BENCHMARK_OUTPUT_DIR:-/data/llh/hstu_v2_plan_user_case_profiles}
DEVICE_ID=${ASCEND_DEVICE_ID:-0}
MSPROF=${MSPROF_BIN:-$(command -v msprof)}
WARMUP=10
REPEATS=10
CURRENT_ROOT=/usr/local/lib/python3.11/dist-packages/ops_rec/opp/atlas_950
CURRENT_OPP="${CURRENT_ROOT}/vendors/arbitrary_func:${CURRENT_ROOT}/vendors/packed_block_sparse:${CURRENT_ROOT}/vendors/atten_metadata:${CURRENT_ROOT}/vendors/hstu_v2:${CURRENT_ROOT}"
BETA_OPP=${HSTU_BETA_OPP:-"/tmp/agent_task1_env/beta_legacy/hstu_v2_b73fe363:/tmp/agent_task1_env/beta_official/hstu_attn_metadata_transformer:${CURRENT_ROOT}"}
mkdir -p "${OUTPUT_DIR}/profiles" "${OUTPUT_DIR}/records" "${OUTPUT_DIR}/logs"
ONLY_MODES=${HSTU_BENCHMARK_MODES:-"beta current_deterministic current_nondeterministic"}
ONLY_DIRECTIONS=${HSTU_BENCHMARK_DIRECTIONS:-"forward backward"}

run_profile() {
    local direction=$1
    local mode=$2
    local test_file="${SCRIPT_DIR}/test_${direction}_user_case_benchmark.py"
    local profile_dir="${OUTPUT_DIR}/profiles/${direction}_${mode}"
    local record="${OUTPUT_DIR}/records/${direction}_${mode}.jsonl"
    local log="${OUTPUT_DIR}/logs/${direction}_${mode}.log"
    local opp_path=${CURRENT_OPP}
    if [[ ${mode} == beta ]]; then opp_path=${BETA_OPP}; fi
    local app
    app=$(printf '/usr/bin/env HSTU_BENCHMARK_MODE=%q HSTU_BENCHMARK_RECORD=%q ASCEND_DEVICE_ID=0 python3 -m pytest %q -v' \
        "${mode}" "${record}" "${test_file}")
    echo "[hstu-plan] ${direction} ${mode}: ${WARMUP}+${REPEATS}, min Task Duration(us)"
    rm -rf "${profile_dir}"
    ASCEND_RT_VISIBLE_DEVICES=${DEVICE_ID} ASCEND_DEVICE_ID=0 ASCEND_CUSTOM_OPP_PATH=${opp_path} \
        "${MSPROF}" --application="${app}" --output="${profile_dir}" --task-time=on --aicpu=on \
        > >(tee "${log}") 2>&1
}

for direction in ${ONLY_DIRECTIONS}; do
    for mode in ${ONLY_MODES}; do
        run_profile "${direction}" "${mode}"
    done
done

if [[ ${HSTU_BENCHMARK_SKIP_STATS:-0} != 1 ]]; then
    python3 "${SCRIPT_DIR}/utils/user_case_perf_stats.py" \
        --input-dir "${OUTPUT_DIR}" --warmup "${WARMUP}" --repeats "${REPEATS}" \
        --csv "${OUTPUT_DIR}/user_case_results.csv" \
        --xlsx "${OUTPUT_DIR}/user_case_results.xlsx"
fi

echo "[hstu-plan] results: ${OUTPUT_DIR}/user_case_results.csv"
