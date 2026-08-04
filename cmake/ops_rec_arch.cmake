# ============================================================================
# cmake/ops_rec_arch.cmake — SOC 架构特化源码收集
# ----------------------------------------------------------------------------
# 提供函数：
#   ops_rec_collect_arch_sources(<var>)
#
# 用法示例（在算子 CMakeLists.txt 中）：
#   ops_rec_collect_arch_sources(ops_srcs)
#
# 功能：根据当前 OPS_REC_ARCH_DIR（由 soc_config.cmake 设置），
# 从当前源码目录下的 <OPS_REC_ARCH_DIR>/ 收集所有 .cpp/.h 文件，
# 追加到 <var> 中。如果该目录不存在，静默跳过。
#
# 目录结构约定：
#   <op>/
#     op_host/                  ← 基础代码（所有 SOC 共享）
#       ln_mul.cpp
#       atlas950/               ← 950 架构特化代码
#         *_tiling*.cpp
#       atlasA2/                ← A2/A3 共享架构特化代码
#         *_tiling*.cpp
#     op_kernel/                ← 基础 kernel
#       ln_mul_kernel.h
#       atlas950/               ← 950 特化 kernel
#       atlasA2/                ← A2/A3 特化 kernel
# ============================================================================

function(ops_rec_collect_arch_sources _out_var)
    if(NOT DEFINED OPS_REC_ARCH_DIR)
        return()
    endif()

    set(_arch_dir "${CMAKE_CURRENT_SOURCE_DIR}/${OPS_REC_ARCH_DIR}")
    if(NOT IS_DIRECTORY "${_arch_dir}")
        return()
    endif()

    file(GLOB _arch_srcs
        "${_arch_dir}/*.cpp"
        "${_arch_dir}/*.h"
    )
    if(_arch_srcs)
        list(APPEND ${_out_var} ${_arch_srcs})
        set(${_out_var} ${${_out_var}} PARENT_SCOPE)
        message(STATUS "ops-rec: collected ${OPS_REC_ARCH_DIR} sources in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endfunction()
