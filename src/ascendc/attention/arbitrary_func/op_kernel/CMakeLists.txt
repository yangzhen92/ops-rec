# ============================================================================
# build_arbitrary_func/op_kernel：AscendC kernel 编译
# ----------------------------------------------------------------------------
# 使用 CANN 的 npu_op_kernel_* 函数编译 kernel。
# npu_op_package 已在上层 CMakeLists.txt 调用。
# OPS_REC_SOC_CXX_DEFS 由根 CMakeLists.txt 统一设置（SUPPORT_950 / SUPPORT_A2 / SUPPORT_A3）
set(_KERNEL_TARGET "${VENDOR_NAME}_kernels")

npu_op_kernel_options(${_KERNEL_TARGET} ALL
    OPTIONS "--cce-long-call=true" ${OPS_REC_SOC_CXX_DEFS}
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
