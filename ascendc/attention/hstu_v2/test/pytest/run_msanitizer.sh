#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=${HSTU_MSAN_LOG_DIR:-/data/llh/guide_new9_msan_logs}
DEVICE_ID=${ASCEND_DEVICE_ID:-0}
VISIBLE_DEVICE=${ASCEND_RT_VISIBLE_DEVICES:-${DEVICE_ID}}
MSAN=${MSSANITIZER_BIN:-$(command -v mssanitizer)}
PYTEST=(python3 -m pytest "${SCRIPT_DIR}/test_msanitizer.py" -v)
CONTROL=(python3 "${SCRIPT_DIR}/../../../../../benchmarks/hstu_v2_agent_optim6/guide9_msan_control.py")
mkdir -p "${LOG_DIR}"
chmod 0750 "${LOG_DIR}"
: > "${LOG_DIR}/summary.tsv"
OVERALL_STATUS=0

run_round() {
    local name=$1
    shift
    echo "[guide_new9] ${name}"
    set +e
    ASCEND_DEVICE_ID=${DEVICE_ID} ASCEND_RT_VISIBLE_DEVICES=${VISIBLE_DEVICE} "$@" \
        > >(tee "${LOG_DIR}/${name}.log") 2>&1
    local status=$?
    set -e
    printf '%s\t%s\n' "${name}" "${status}" >> "${LOG_DIR}/summary.tsv"
    if [[ ${status} -ne 0 ]]; then OVERALL_STATUS=1; fi
}

set -e
run_round 00_plain "${PYTEST[@]}"
run_round 01_device_heap "${MSAN}" -t memcheck --leak-check=yes --check-device-heap=yes \
    --log-file="${LOG_DIR}/01_device_heap.msan" -- "${PYTEST[@]}"
run_round 01_device_heap_control "${MSAN}" -t memcheck --leak-check=yes --check-device-heap=yes \
    --log-file="${LOG_DIR}/01_device_heap_control.msan" -- "${CONTROL[@]}"
run_round 02_cann_heap "${MSAN}" -t memcheck --leak-check=yes --check-cann-heap=yes \
    --log-file="${LOG_DIR}/02_cann_heap.msan" -- "${PYTEST[@]}"
run_round 02_cann_heap_control "${MSAN}" -t memcheck --leak-check=yes --check-cann-heap=yes \
    --log-file="${LOG_DIR}/02_cann_heap_control.msan" -- "${CONTROL[@]}"
run_round 03_unused_memory "${MSAN}" -t memcheck --check-unused-memory=yes \
    --log-file="${LOG_DIR}/03_unused_memory.msan" -- "${PYTEST[@]}"
run_round 03_unused_memory_control "${MSAN}" -t memcheck --check-unused-memory=yes \
    --log-file="${LOG_DIR}/03_unused_memory_control.msan" -- "${CONTROL[@]}"
run_round 04_initcheck "${MSAN}" -t initcheck --log-file="${LOG_DIR}/04_initcheck.msan" -- "${PYTEST[@]}"
run_round 04_initcheck_control "${MSAN}" -t initcheck \
    --log-file="${LOG_DIR}/04_initcheck_control.msan" -- "${CONTROL[@]}"
run_round 05_racecheck "${MSAN}" -t racecheck --log-file="${LOG_DIR}/05_racecheck.msan" -- "${PYTEST[@]}"
run_round 05_racecheck_control "${MSAN}" -t racecheck \
    --log-file="${LOG_DIR}/05_racecheck_control.msan" -- "${CONTROL[@]}"
run_round 06_synccheck "${MSAN}" -t synccheck --log-file="${LOG_DIR}/06_synccheck.msan" -- "${PYTEST[@]}"
run_round 06_synccheck_control "${MSAN}" -t synccheck \
    --log-file="${LOG_DIR}/06_synccheck_control.msan" -- "${CONTROL[@]}"

echo "[guide_new9] reports: ${LOG_DIR}"
cat "${LOG_DIR}/summary.tsv"
exit "${OVERALL_STATUS}"
