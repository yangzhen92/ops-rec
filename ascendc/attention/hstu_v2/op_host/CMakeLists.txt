# ============================================================================
# hstu_backward_v2/op_host：算子 host 侧实现（OpDef + tiling + infershape）
# ----------------------------------------------------------------------------
# 使用 CANN 的 npu_op_code_gen + npu_op_library 编译。
# npu_op_package 已在上层 CMakeLists.txt 调用。
set(MY_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../common)

# OPS_REC_SOC_CXX_DEFS 由根 CMakeLists.txt 统一设置（SUPPORT_950 / SUPPORT_A2 / SUPPORT_A3）

file(GLOB ops_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB ops_cc_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
list(APPEND ops_srcs ${ops_cc_srcs})

# 收集 SOC 架构特化源码（op_host/atlas950/ 或 op_host/atlasA2/ 目录下的额外实现）
ops_rec_collect_arch_sources(ops_srcs)

npu_op_code_gen(
        SRC ${ops_srcs}
        PACKAGE ${VENDOR_NAME}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
        COMPILE_OPTIONS
        ${OPS_REC_SOC_CXX_DEFS}
        -I${MY_COMMON_DIR}/log
        -I${MY_COMMON_DIR}/utils
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
)

npu_op_package_add(${VENDOR_NAME}
        LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)

# CANN 自动生成的 tiling target 名为 <PACKAGE>_ascendc_cust_optiling
set(TILING_TARGET ${VENDOR_NAME}_ascendc_cust_optiling)

# 要求 tiling 库独立安装到 op_tiling/ 路径
install(FILES $<TARGET_FILE:${TILING_TARGET}>
        DESTINATION ${CMAKE_INSTALL_PREFIX}/op_impl/ai_core/tbe/op_tiling
        RENAME liboptiling.so
        PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
install(FILES $<TARGET_FILE:${TILING_TARGET}>
        DESTINATION ${CMAKE_INSTALL_PREFIX}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
        RENAME libcust_opmaster_rt2.0.so
)
