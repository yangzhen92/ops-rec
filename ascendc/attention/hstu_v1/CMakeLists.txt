# ============================================================================
# hstu_backward_v1 算子聚合 CMakeLists
# ----------------------------------------------------------------------------
# 设置 vendor 名，使用 npu_op_package + INSTALL_PATH 将产物
# 定向到 opp/<variant>/vendors/<op_name>/ 目录。
# ============================================================================

add_subdirectory(hstu_backward_v1)
#add_subdirectory(hstu_forward_v1)
