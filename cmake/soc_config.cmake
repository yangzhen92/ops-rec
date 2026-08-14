# ============================================================================
# cmake/soc_config.cmake — SOC 变体 -> 芯片宏映射 + CANN 构建框架初始化
# ----------------------------------------------------------------------------
# 本文件包含工程相关的 SOC 配置和 CANN 构建框架的加载。
# ASCEND_HOME_PATH 由环境变量或 -D 提供（见下方定义）。
#
# 提供以下变量供其他 CMakeLists.txt 使用：
#   NPU_CHIP_ATLAS_950 / NPU_CHIP_ATLAS_A2 / NPU_CHIP_ATLAS_A3
#   ASCEND_COMPUTE_UNIT
#   CANN_INCLUDE_DIR, CANN_LIB_DIR
# ============================================================================

# ---------------------------------------------------------------------------
# ASCEND_HOME_PATH — 从环境变量或 -D 读取，未设置则报错
# ---------------------------------------------------------------------------
if(NOT DEFINED ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
    else()
        message(FATAL_ERROR "ASCEND_HOME_PATH not set. Please set it to CANN install path (e.g. /usr/local/Ascend/latest)")
    endif()
endif()

message(STATUS "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")

# ---------------------------------------------------------------------------
# 加载 CANN 构建框架（提供 npu_op_* 系列函数）
# 不放在 toolchain 中，因为这是工程构建框架而非编译器配置。
# ---------------------------------------------------------------------------
set(_FMWK_DIR "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules")
if(EXISTS "${_FMWK_DIR}/config.cmake")
    include("${_FMWK_DIR}/config.cmake")
    include("${_FMWK_DIR}/func.cmake")
    include("${_FMWK_DIR}/intf.cmake")
    message(STATUS "CANN fwk_modules loaded from ${_FMWK_DIR}")
else()
    message(FATAL_ERROR "fwk_modules not found at ${_FMWK_DIR}")
endif()

set(_ASC_MODULES_DIR "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/asc_modules")
if(EXISTS "${_ASC_MODULES_DIR}/FindASC.cmake")
    list(APPEND CMAKE_MODULE_PATH "${_ASC_MODULES_DIR}")
    include("${_ASC_MODULES_DIR}/FindASC.cmake")
    message(STATUS "CANN asc_modules loaded from ${_ASC_MODULES_DIR}")
endif()

# ---------------------------------------------------------------------------
# CANN include/lib 路径
# ---------------------------------------------------------------------------
set(CANN_ARCH_INCLUDE_DIR "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
if(IS_DIRECTORY "${CANN_ARCH_INCLUDE_DIR}")
    set(CANN_INCLUDE_DIR "${CANN_ARCH_INCLUDE_DIR}" CACHE PATH "CANN include path")
else()
    set(CANN_INCLUDE_DIR "${ASCEND_HOME_PATH}/include" CACHE PATH "CANN include path")
endif()
set(CANN_INCLUDE_EXTERNAL_DIR "${ASCEND_HOME_PATH}/include/external" CACHE PATH "CANN external include path")
set(CANN_LIB_DIR "${ASCEND_HOME_PATH}/lib64" CACHE PATH "CANN lib path")

message(STATUS "CANN_INCLUDE_DIR=${CANN_INCLUDE_DIR}")
message(STATUS "CANN_LIB_DIR=${CANN_LIB_DIR}")

# ---------------------------------------------------------------------------
# OPS_REC_SOC_VARIANT — 由 cmake -D 或环境变量传入
# 支持的值：atlas_950 / atlas_a2 / atlas_a3
# ---------------------------------------------------------------------------
if(DEFINED OPS_REC_SOC_VARIANT AND NOT OPS_REC_SOC_VARIANT STREQUAL "")
    set(OPS_REC_SOC_VARIANT "${OPS_REC_SOC_VARIANT}" CACHE STRING "ops-rec SOC variant" FORCE)
elseif(DEFINED ENV{OPS_REC_SOC_VARIANT})
    set(OPS_REC_SOC_VARIANT "$ENV{OPS_REC_SOC_VARIANT}" CACHE STRING "ops-rec SOC variant" FORCE)
else()
    set(OPS_REC_SOC_VARIANT "atlas_950" CACHE STRING "ops-rec SOC variant (default)")
endif()

# 推导 ASCEND_COMPUTE_UNIT 和 chip 宏
string(TOLOWER "${OPS_REC_SOC_VARIANT}" _var_lower)
string(REGEX REPLACE "^atlas_?" "" _soc_short "${_var_lower}")

# 先统一初始化为 0
set(NPU_CHIP_ATLAS_950 0)
set(NPU_CHIP_ATLAS_A2 0)
set(NPU_CHIP_ATLAS_A3 0)

if(_soc_short STREQUAL "950")
    set(ASCEND_COMPUTE_UNIT "ascend950" CACHE STRING "Ascend compute unit" FORCE)
    set(NPU_CHIP_ATLAS_950 1)
    set(OPS_REC_ARCH_DIR "atlas950")
elseif(_soc_short STREQUAL "a2")
    set(ASCEND_COMPUTE_UNIT "ascend910b" CACHE STRING "Ascend compute unit" FORCE)
    set(NPU_CHIP_ATLAS_A2 1)
    set(OPS_REC_ARCH_DIR "atlasA2")
elseif(_soc_short STREQUAL "a3")
    set(ASCEND_COMPUTE_UNIT "ascend910_93" CACHE STRING "Ascend compute unit" FORCE)
    set(NPU_CHIP_ATLAS_A3 1)
    set(OPS_REC_ARCH_DIR "atlasA2")
else()
    message(FATAL_ERROR "Unknown OPS_REC_SOC_VARIANT=${OPS_REC_SOC_VARIANT}, expected atlas_950 / atlas_a2 / atlas_a3")
endif()

message(STATUS "OPS_REC_SOC_VARIANT=${OPS_REC_SOC_VARIANT}, ASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT}")
message(STATUS "NPU_CHIP_ATLAS_950=${NPU_CHIP_ATLAS_950}, NPU_CHIP_ATLAS_A2=${NPU_CHIP_ATLAS_A2}, NPU_CHIP_ATLAS_A3=${NPU_CHIP_ATLAS_A3}")
