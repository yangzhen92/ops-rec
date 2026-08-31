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
# 本算子的 arch 目录含编译单元 tiling_policy_arch.cpp（TilingKeySetImpl/GetArchWorkspaceExtraSize），
# 以及头文件 op_def_arch.h / hstu_kernel_tiling_key.h（供 -I include 使用，非编译单元），
# 过滤掉 .h 避免作为源文件传入 npu_op_code_gen。
ops_rec_collect_arch_sources(ops_srcs)
list(FILTER ops_srcs EXCLUDE REGEX "\\.h$")

# ── 纯 tiling 实现编译单元拆分（修复 libcust_opapi.so multiple definition）──
# tiling_policy*.cpp（根目录）与 <arch>/tiling_policy_arch.cpp 是纯 TilingPolicy
# 实现编译单元，不含 OP_ADD/REG_OP 注册宏（不参与 op def 注册与 codegen 扫描）。
# 若同时编入 cust_op_proto（GRAPH）与 cust_optiling（TILING），最终链接
# libcust_opapi.so 时同一批符号出现两份定义 → /usr/bin/ld multiple definition。
# 因此从 GRAPH 源列表剔除，仅编译进 TILING 库；hstu_*_forward.cpp.o 中对
# TilingPolicy 符号的未定义引用由 cust_optiling 在最终 .so 链接时解析。
set(tiling_impl_srcs ${ops_srcs})
list(FILTER tiling_impl_srcs INCLUDE REGEX "[/\\]tiling_policy[^/\\]*\\.cpp$")

npu_op_code_gen(
    SRC ${ops_srcs}
    PACKAGE ${VENDOR_NAME}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        ${OPS_REC_SOC_CXX_DEFS}
        -I${MY_COMMON_DIR}/log
        -I${MY_COMMON_DIR}/utils
        -I${HSTU_KERNEL_INCLUDE_DIR}
        # arch 子目录下的 tiling_policy_arch.cpp 需 include op_host 根目录的公共头
        -I${CMAKE_CURRENT_SOURCE_DIR}
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
# RUN 模式下 GRAPH（libcust_opsproto_rt2.0.so）、TILING（libcust_opmaster_rt2.0.so）、
# ACLNN（libcust_opapi.so）是三个独立 dlopen 的 .so。hstu_*_forward.cpp（op def/
# infershape/tiling 三套注册）直接实例化 TilingPolicy* 子类并调用其方法，因此
# tiling_policy*.cpp 实现必须编进 GRAPH 与 TILING 两个库（各自 .so 内自含符号，
# 跨 .so 不冲突）。SHARED 模式下所有 host 库打进单一 libcust_opapi.so 才会
# 出现 multiple definition，RUN 模式无此问题，故不再从 GRAPH 源列表剔除。
npu_op_library(cust_op_proto GRAPH
    ${ops_srcs}
    ${tiling_impl_srcs}
    ${group_proto_src}
    ${proto_src}
)

target_include_directories(cust_op_proto PRIVATE
    ${MY_COMMON_DIR}/log
    ${MY_COMMON_DIR}/utils
    ${HSTU_KERNEL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
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
    ${CMAKE_CURRENT_SOURCE_DIR}
)

npu_op_package_add(${VENDOR_NAME}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)
