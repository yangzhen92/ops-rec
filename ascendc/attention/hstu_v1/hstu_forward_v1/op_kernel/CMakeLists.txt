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
# 官方机制：npu_op_kernel_sources 仅支持 OP_TYPE/KERNEL_DIR/COMPUTE_UNIT/KERNEL_FILE 配对，
# KERNEL_DIR 相对 SRC_BASE 定位子目录，KERNEL_FILE 指定实现文件名；每个 op 一个调用。
# （SOURCES 参数不存在，KERNEL_DIR 不带 KERNEL_FILE 时框架按 SRC_BASE + <op名>.cpp 定位，均找不到子目录文件）
npu_op_kernel_sources(${_KERNEL_TARGET}
    OP_TYPE HstuDenseForward
    KERNEL_DIR ./${_KERNEL_SRC_DIR}
    KERNEL_FILE hstu_dense_forward.cpp
)
npu_op_kernel_sources(${_KERNEL_TARGET}
    OP_TYPE HstuJaggedForward
    KERNEL_DIR ./${_KERNEL_SRC_DIR}
    KERNEL_FILE hstu_jagged_forward.cpp
)
npu_op_kernel_sources(${_KERNEL_TARGET}
    OP_TYPE HstuDenseForward
    KERNEL_DIR ./${_KERNEL_SRC_DIR}
    KERNEL_FILE hstu_paged_forward.cpp
)

# SRC_BASE 为 KERNEL_DIR 的相对基准（实际目录 = SRC_BASE + KERNEL_DIR）
npu_op_kernel_library(${_KERNEL_TARGET}
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${VENDOR_NAME}
    LIBRARY ${_KERNEL_TARGET}
)
