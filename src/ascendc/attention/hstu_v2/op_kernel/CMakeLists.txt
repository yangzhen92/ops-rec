# ============================================================================
# hstu_backward_v2/op_kernel：AscendC kernel 编译
# ----------------------------------------------------------------------------
# 使用 CANN 的 npu_op_kernel_* 函数编译 kernel。
# OPS_REC_SOC_CXX_DEFS 由根 CMakeLists.txt 统一设置（SUPPORT_950 / SUPPORT_A2 / SUPPORT_A3）
set(_KERNEL_TARGET "${VENDOR_NAME}_kernels")

# 依赖 third_party/catlass（git submodule）——挂载 include 路径并注入 a5(Ascend950) 特化宏
set(CATLASS_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../third_party/catlass/include")

npu_op_kernel_options(${_KERNEL_TARGET} ALL
        OPTIONS "--cce-long-call=true"
        -DCATLASS_ARCH=3510
        -DARCH_CODE=Ascend950
        -DUSE_TLA=1
        -DCATLASS_BISHENG_ARCH=a5
        -DCATLASS_ARCH_A5_ENABLED=true
        -DENABLE_CV_COMM_VIA_SSBUF=true
        -DCATLASS_HOME=${CATLASS_INCLUDE_DIR}/..
        -I${CATLASS_INCLUDE_DIR}
        ${OPS_REC_SOC_CXX_DEFS}
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
