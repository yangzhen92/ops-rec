# ===========================================================================
# soc_ops_config.cmake — ops-rec AscendC 算子列表（按 SOC 芯片管理）
# 控制每个 SOC variant 编译哪些算子，同时用于生成 config.ini 的 load_priority。
# 新增算子时在对应芯片列表追加算子名，无需改动 CMakeLists.txt。
# ================================ A5 ops ================================
set(OPS_REC_OPS_ATLAS_950
    ln_mul
    hstu_v1
    disentangle_attention
    disentangle_attention
    block_sparse_func
)

# ================================ A3 ops ================================
set(OPS_REC_OPS_ATLAS_A3
    ln_mul
    disentangle_attention
    hstu_v1
)

# ================================ A2 ops ================================
set(OPS_REC_OPS_ATLAS_A2
    ln_mul
    disentangle_attention
    hstu_v1
)


# ===========================================================================
# 算子清单（脚本模式）
# 用法（bash）：
#   cmake -DSOC=atlas_950 -P cmake/soc_ops_config.cmake。
# ---------------------------------------------------------------------------
string(TOUPPER "${SOC}" _ops_scm_soc_upper)
string(REPLACE "." "_" _ops_scm_var_name "OPS_REC_OPS_${_ops_scm_soc_upper}")

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    if(NOT DEFINED ${_ops_scm_var_name})
        message(FATAL_ERROR
            "soc_ops_config: unknown SOC '${SOC}' (no variable ${_ops_scm_var_name})")
    endif()
    foreach(_ops_scm_op IN LISTS ${_ops_scm_var_name})
        message("${_ops_scm_op}")
    endforeach()
endif()
unset(_ops_scm_soc_upper)
unset(_ops_scm_var_name)
