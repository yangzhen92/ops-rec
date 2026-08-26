# ============================================================================
# in_linear_silu/op_kernel：AscendC kernel 编译
# ----------------------------------------------------------------------------
# 使用 CANN 的 npu_op_kernel_* 函数编译 kernel。
# npu_op_package 已在上层 CMakeLists.txt 调用。
# OPS_REC_SOC_CXX_DEFS 由根 CMakeLists.txt 统一设置（SUPPORT_950 / SUPPORT_A2 / SUPPORT_A3）
set(_KERNEL_TARGET "${VENDOR_NAME}_kernels")

# ---------------------------------------------------------------------------
# CATLASS 编译选项 — 按 SOC 变体选择对应宏
# CATLASS_HOME 采用相对路径，指向仓库根下 third_party/catlass，不依赖环境变量。
# 可用 -DCATLASS_HOME=xxx 覆盖。
# ---------------------------------------------------------------------------

if(NOT DEFINED CATLASS_HOME)
    # 当前文件位于 <repo>/ascendc/activation/in_linear_silu/in_linear_silu/op_kernel
    set(CATLASS_HOME "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../third_party/catlass")
endif()

if(IS_DIRECTORY "${CATLASS_HOME}")
    set(catlass_include_dir "${CATLASS_HOME}/include")
    if(NPU_CHIP_ATLAS_950)
        # c310 / Ascend950 / a5
        npu_op_kernel_options(${_KERNEL_TARGET} ALL
            OPTIONS
                "--cce-long-call=true"
                ${OPS_REC_SOC_CXX_DEFS}
                -DCATLASS_ARCH=3510
                -DCATLASS_BISHENG_ARCH=a5
                -DUSE_TLA=1
                -DARCH_CODE=Ascend950
                -DCATLASS_ARCH_A5_ENABLED=1
                -DENABLE_CV_COMM_VIA_SSBUF=true
                -DCATLASS_HOME=${CATLASS_HOME}
                -I${CATLASS_HOME}/include
        )
    elseif(NPU_CHIP_ATLAS_A2 OR NPU_CHIP_ATLAS_A3)
        # AtlasA2 / a2
        npu_op_kernel_options(${_KERNEL_TARGET} ALL
            OPTIONS
                "--cce-long-call=true"
                ${OPS_REC_SOC_CXX_DEFS}
                -DCATLASS_ARCH=2201
                -DCATLASS_BISHENG_ARCH=a2
                -DUSE_TLA=0
                -DARCH_CODE=AtlasA2
                -DCATLASS_ARCH_A2_ENABLED=1
                -DENABLE_CV_COMM_VIA_SSBUF=true
                -DCATLASS_HOME=${CATLASS_HOME}
                -I${CATLASS_HOME}/include
        )
    endif()
endif()

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
