# ============================================================================
# index_select_for_rank1_backward 算子聚合 CMakeLists
# ----------------------------------------------------------------------------
# 设置 vendor 名，使用 npu_op_package + INSTALL_PATH 将产物
# 定向到 opp/<variant>/vendors/<op_name>/ 目录。
# ============================================================================

# vendor 名，同时也是 CANN package 名
set(VENDOR_NAME "index_select_for_rank1_backward")

# 独立 autogen 路径，避免多个算子共享冲突
set(ASCEND_AUTOGEN_PATH "${CMAKE_CURRENT_BINARY_DIR}/autogen")
file(MAKE_DIRECTORY "${ASCEND_AUTOGEN_PATH}")

# 安装路径
set(OPP_VENDOR_INSTALL_PREFIX
    "${CUSTOM_OPP_INSTALL_PREFIX}/vendors/${VENDOR_NAME}")

npu_op_package(${VENDOR_NAME}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${OPP_VENDOR_INSTALL_PREFIX}
        VENDOR_NAME ${VENDOR_NAME}
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
