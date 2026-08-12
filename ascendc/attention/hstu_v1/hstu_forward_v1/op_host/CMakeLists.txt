# ============================================================================
# hstu_forward_v1/op_host：算子 host 侧实现（OpDef + tiling + infershape）
# ----------------------------------------------------------------------------
# 使用 CANN 的 npu_op_code_gen + npu_op_library 编译。
# npu_op_package 已在上层 CMakeLists.txt 调用。
# 注意：本 vendor 位于 hstu_v1/hstu_forward_v1 两级嵌套子组下，
# 相对 common 需上溯 4 级（../../../..），与 ln_mul 等一层子目录的 vendor 不同。
set(MY_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common)

# OPS_REC_SOC_CXX_DEFS 由根 CMakeLists.txt 统一设置（SUPPORT_950 / SUPPORT_A2 / SUPPORT_A3）
# 该算子按 SOC 使用不同布局的 hstu_kernel_tiling_key.h（host 侧 tiling 生成依赖其字段布局）：
#   - A2/A3：op_host/atlasA2/（v220 实现，blockN 9bit 固定 256/256/512）
#   - 950：  op_host/atlas950/（c310 实现，blockN 13bit 多 block 组合）
# OPS_REC_ARCH_DIR 由 soc_config.cmake 按 variant 设置（atlas950 / atlasA2），
# 因此 kernel key include 路径随 ARCH_DIR 切换，tiling_policy.cpp 无需改动。
set(HSTU_KERNEL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${OPS_REC_ARCH_DIR}")

file(GLOB ops_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB ops_cc_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
list(APPEND ops_srcs ${ops_cc_srcs})

# 收集 SOC 架构特化源码（op_host/atlas950/ 或 op_host/atlasA2/ 目录下的额外实现）。
# 本算子的 arch 目录仅含 hstu_kernel_tiling_key.h（供 -I include 使用，非编译单元），
# 过滤掉 .h 避免作为源文件传入 npu_op_code_gen。
ops_rec_collect_arch_sources(ops_srcs)
list(FILTER ops_srcs EXCLUDE REGEX "\\.h$")

npu_op_code_gen(
    SRC ${ops_srcs}
    PACKAGE ${VENDOR_NAME}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        ${OPS_REC_SOC_CXX_DEFS}
        -I${MY_COMMON_DIR}/log
        -I${MY_COMMON_DIR}/utils
        -I${HSTU_KERNEL_INCLUDE_DIR}
)

file(GLOB autogen_aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
set_source_files_properties(${autogen_aclnn_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN
    ${autogen_aclnn_src}
)

target_include_directories(cust_opapi PRIVATE
    ${MY_COMMON_DIR}/log
    ${MY_COMMON_DIR}/utils
)

target_compile_options(cust_opapi PRIVATE
    -fvisibility=hidden
)

file(GLOB group_proto_src ${ASCEND_AUTOGEN_PATH}/group_op_proto/*.cc)
file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${group_proto_src} PROPERTIES GENERATED TRUE)
set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_op_proto GRAPH
    ${ops_srcs}
    ${group_proto_src}
    ${proto_src}
)

target_include_directories(cust_op_proto PRIVATE
    ${MY_COMMON_DIR}/log
    ${MY_COMMON_DIR}/utils
    ${HSTU_KERNEL_INCLUDE_DIR}
)

target_compile_options(cust_op_proto PRIVATE
    -fvisibility=hidden
)

file(GLOB fallback_src ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
set_source_files_properties(${fallback_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_optiling TILING
    ${ops_srcs}
    ${fallback_src}
)

target_include_directories(cust_optiling PRIVATE
    ${MY_COMMON_DIR}/log
    ${MY_COMMON_DIR}/utils
    ${HSTU_KERNEL_INCLUDE_DIR}
)

npu_op_package_add(${VENDOR_NAME}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)

# CANN 自动生成的 tiling target 名为 <PACKAGE>_ascendc_cust_optiling
# 一个 vendor 下多个算子共享同一个 tiling target
set(TILING_TARGET ${VENDOR_NAME}_ascendc_cust_optiling)

# 要求 tiling 库独立安装到 op_tiling/ 路径
# nnopbase individual op 流程通过 dlopen 该目录下的 .so 定位 tiling 函数。
install(FILES $<TARGET_FILE:${TILING_TARGET}>
    DESTINATION ${CMAKE_INSTALL_PREFIX}/op_impl/ai_core/tbe/op_tiling
    RENAME liboptiling.so
    PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
install(FILES $<TARGET_FILE:${TILING_TARGET}>
    DESTINATION ${CMAKE_INSTALL_PREFIX}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
    RENAME libcust_opmaster_rt2.0.so
)
