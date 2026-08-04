# ============================================================================
# cmake/toolchain_ascend.cmake — Ascend 编译器 toolchain
# ----------------------------------------------------------------------------
# 此文件仅管理 AscendC 编译器和 CANN SDK 路径查找，
# 不包含 SOC 映射、算子列表、CANN 构建框架等工程相关配置。
# ============================================================================

# ---------------------------------------------------------------------------
# ASCEND_HOME_PATH
# ---------------------------------------------------------------------------
if(NOT DEFINED ENV{ASCEND_HOME_PATH} OR "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    message(FATAL_ERROR "ASCEND_HOME_PATH not set. Please set it to CANN install path (e.g. /usr/local/Ascend/latest)")
endif()

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN install root")

# ---------------------------------------------------------------------------
# AscendC 编译器
# ---------------------------------------------------------------------------
find_program(ASCENDC_COMPILER ascendc
    PATHS "${ASCEND_HOME_PATH}/tools/ascendc/bin"
    NO_DEFAULT_PATH
)
if(NOT ASCENDC_COMPILER)
    message(WARNING "ascendc compiler not found at ${ASCEND_HOME_PATH}/tools/ascendc/bin")
else()
    message(STATUS "ASCENDC_COMPILER=${ASCENDC_COMPILER}")
endif()

# ---------------------------------------------------------------------------
# op_build 工具
# ---------------------------------------------------------------------------
set(OP_BUILD_TOOL "${ASCEND_HOME_PATH}/tools/opbuild/op_build" CACHE FILEPATH "op_build tool")
if(NOT EXISTS "${OP_BUILD_TOOL}")
    message(WARNING "op_build tool not found at ${OP_BUILD_TOOL}")
else()
    message(STATUS "OP_BUILD_TOOL=${OP_BUILD_TOOL}")
endif()
