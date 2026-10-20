# ============================================================================
# gather_for_rank1 算子聚合 CMakeLists
# ----------------------------------------------------------------------------
# gather_for_rank1 是 1D 嵌入权重表的 index_select 前向 + scatter-add 反向
# 组成的 autograd 算子对，前向/反向分拆为两个独立 vendor 编译：
#   - gather_for_rank1_forward        （前向，迁移完成后加入）
#   - index_select_for_rank1_backward （反向）
# ============================================================================

add_subdirectory(index_select_for_rank1_backward)
# add_subdirectory(gather_for_rank1_forward)
