# ============================================================================
# hstu_dense_forward/op_kernel：AscendC kernel 编译
# ----------------------------------------------------------------------------
# 使用 CANN 的 npu_op_kernel_* 函数编译 kernel。
# npu_op_package 已在上层 CMakeLists.txt 调用。
# OPS_REC_SOC_CXX_DEFS 由根 CMakeLists.txt 统一设置（SUPPORT_950 / SUPPORT_A2 / SUPPORT_A3）
#
# 该算子按 SOC 使用两套互斥的 kernel 实现（同名头文件，不能同时编译）：
#   - A2/A3：atlasA2/（v220 实现，含共享头）
#   - 950：  atlas950/（c310 实现，含共享头）
# 因此 KERNEL_DIR 须按 SOC 切换，避免 tiling key 重复声明冲突。
set(_KERNEL_TARGET "${VENDOR_NAME}_kernels")

# Atlas 950 特定编译选项
set(_EXTRA_COMPILE_OPTIONS "")
if(NPU_CHIP_ATLAS_950)
    set(_KERNEL_SRC_DIR "atlas950")
    list(APPEND _EXTRA_COMPILE_OPTIONS "-DENABLE_CV_COMM_VIA_SSBUF=true")
else()
    set(_KERNEL_SRC_DIR "atlasA2")
endif()

npu_op_kernel_options(${_KERNEL_TARGET} ALL
    OPTIONS "--cce-long-call=true" ${OPS_REC_SOC_CXX_DEFS} ${_EXTRA_COMPILE_OPTIONS}
)

# op 实现源文件按 SOC 切目录（atlas950/ 或 atlasA2/）。
# 显式 SOURCES + 绝对路径列出：KERNEL_DIR 子目录模式下，opbuild 的 adapt 脚本
# 在 build 目录执行时相对路径失效，找不到 op 源文件。
npu_op_kernel_sources(${_KERNEL_TARGET}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/${_KERNEL_SRC_DIR}/hstu_dense_forward.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/${_KERNEL_SRC_DIR}/hstu_jagged_forward.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/${_KERNEL_SRC_DIR}/hstu_paged_forward.cpp
)

# SRC_BASE 必须与 op 实现文件所在目录一致：opbuild 按 SRC_BASE + <op名>.cpp 定位实现文件，
# 本算子实现位于架构子目录（atlas950/ 或 atlasA2/），不能指向 op_kernel 根目录。
npu_op_kernel_library(${_KERNEL_TARGET}
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/${_KERNEL_SRC_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${VENDOR_NAME}
    LIBRARY ${_KERNEL_TARGET}
)
