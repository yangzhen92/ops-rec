# ============================================================================
# disentangle_attention/op_kernel：AscendC kernel 编译
# ----------------------------------------------------------------------------
# 使用 CANN 的 npu_op_kernel_* 函数编译 kernel。
# npu_op_package 已在上层 CMakeLists.txt 调用。
# OPS_REC_SOC_CXX_DEFS 由根 CMakeLists.txt 统一设置（SUPPORT_950 / SUPPORT_A2 / SUPPORT_A3）
set(_KERNEL_TARGET "${VENDOR_NAME}_kernels")

# Atlas 950 特定编译选项
set(_EXTRA_COMPILE_OPTIONS "")
if(NPU_CHIP_ATLAS_950)
    list(APPEND _EXTRA_COMPILE_OPTIONS "-DENABLE_CV_COMM_VIA_SSBUF=true")
endif()

npu_op_kernel_options(${_KERNEL_TARGET} ALL
    OPTIONS "--cce-long-call=true" ${OPS_REC_SOC_CXX_DEFS} ${_EXTRA_COMPILE_OPTIONS}
)

npu_op_kernel_sources(${_KERNEL_TARGET}
    KERNEL_DIR ./
)

npu_op_kernel_library(${_KERNEL_TARGET}
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${VENDOR_NAME}
    LIBRARY ${_KERNEL_TARGET}
)