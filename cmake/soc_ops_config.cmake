# ===========================================================================
# soc_ops_config.cmake — ops-rec AscendC 算子列表（按 SOC 芯片管理）
# 控制每个 SOC variant 编译哪些算子，同时用于生成 config.ini 的 load_priority。
# 新增算子时在对应芯片列表追加算子名，无需改动 CMakeLists.txt。
# ================================ A5 ops ===================================
set(OPS_REC_OPS_ATLAS_950
    ln_mul
    disentangle_attention
    block_sparse_func
    arbitrary_func
    hstu_v1
    hstu_v2
    reverse_sequence
    in_linear_silu
    fa_infer_with_arbitrary_func
)

# ================================ A3 ops ===================================
set(OPS_REC_OPS_ATLAS_A3
    ln_mul
    disentangle_attention
    hstu_v1
    reverse_sequence
    in_linear_silu
)

# ================================ A2 ops ===================================
set(OPS_REC_OPS_ATLAS_A2
    ln_mul
    disentangle_attention
    hstu_v1
    reverse_sequence
    in_linear_silu
)


# ===========================================================================
# 算子路径定位与聚合展开辅助函数
#   - ops_rec_find_op_dir / ops_rec_vendor_dir：跨层级定位算子及其子 vendor
#     的物理源码路径（含嵌套聚合目录）。
#   - ops_rec_op_leaves：单算子返回自身，聚合算子按子目录 CMakeLists.txt
#     展开为叶子 vendor 列表。
#   - 自动将白名单算子列表转成叶子级展开列表，供构建及 load_priority 收集。
# ===========================================================================
get_filename_component(OPS_REC_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

function(ops_rec_find_op_dir op_name out_var)
    file(GLOB _grp_dirs LIST_DIRECTORIES true "${OPS_REC_REPO_ROOT}/ascendc/*")
    set(_found_dir)
    foreach(_gd ${_grp_dirs})
        if(EXISTS "${_gd}/${op_name}")
            set(_found_dir "${_gd}/${op_name}")
            break()
        endif()
    endforeach()
    set(${out_var} "${_found_dir}" PARENT_SCOPE)
endfunction()

function(ops_rec_vendor_dir vendor out_var)
    file(GLOB _grp_dirs LIST_DIRECTORIES true "${OPS_REC_REPO_ROOT}/ascendc/*")
    set(_found_dir)
    foreach(_gd ${_grp_dirs})
        if(EXISTS "${_gd}/${vendor}" AND IS_DIRECTORY "${_gd}/${vendor}")
            set(_found_dir "${_gd}/${vendor}")
            break()
        endif()
        file(GLOB _agg_dirs LIST_DIRECTORIES true "${_gd}/*")
        foreach(_ad ${_agg_dirs})
            if(EXISTS "${_ad}/${vendor}" AND IS_DIRECTORY "${_ad}/${vendor}")
                set(_found_dir "${_ad}/${vendor}")
                break()
            endif()
        endforeach()
        if(_found_dir)
            break()
        endif()
    endforeach()
    set(${out_var} "${_found_dir}" PARENT_SCOPE)
endfunction()

function(ops_rec_op_leaves op out_var)
    set(_leaves)
    ops_rec_find_op_dir("${op}" _opdir)
    if(NOT _opdir)
        set(${out_var} "${op}" PARENT_SCOPE)  # 目录缺失（如未显式建目录）时回退
        return()
    endif()

    set(_self_name "")
    if(EXISTS "${_opdir}/CMakeLists.txt")
        file(READ "${_opdir}/CMakeLists.txt" _self_content)
        string(REGEX MATCH "set\\(VENDOR_NAME[ \t]+\"([^\"]+)\"" _self_m "${_self_content}")
        if(CMAKE_MATCH_1)
            set(_self_name "${CMAKE_MATCH_1}")
        endif()
    endif()
    if(_self_name)
        set(${out_var} "${_self_name}" PARENT_SCOPE)
        return()
    endif()

    file(GLOB _subdirs LIST_DIRECTORIES true "${_opdir}/*")
    set(_has_leaf FALSE)
    foreach(_sd ${_subdirs})
        if(NOT EXISTS "${_sd}/CMakeLists.txt")
            continue()
        endif()
        file(READ "${_sd}/CMakeLists.txt" _content)
        string(REGEX MATCH "set\\(VENDOR_NAME[ \t]+\"([^\"]+)\"" _m "${_content}")
        if(CMAKE_MATCH_1)
            list(APPEND _leaves "${CMAKE_MATCH_1}")
            set(_has_leaf TRUE)
        endif()
    endforeach()
    if(NOT _has_leaf)
        set(_leaves "${op}")
    endif()
    set(${out_var} ${_leaves} PARENT_SCOPE)
endfunction()

foreach(_lf_soc_var IN ITEMS
        OPS_REC_OPS_ATLAS_950
        OPS_REC_OPS_ATLAS_A3
        OPS_REC_OPS_ATLAS_A2)
    if(NOT DEFINED ${_lf_soc_var})
        continue()
    endif()
    string(REPLACE "OPS_REC_OPS_" "OPS_REC_LEAF_" _lf_leaf_var "${_lf_soc_var}")
    set(${_lf_leaf_var})
    foreach(_lf_op IN LISTS ${_lf_soc_var})
        ops_rec_op_leaves("${_lf_op}" _lf_op_leaves)
        list(APPEND ${_lf_leaf_var} ${_lf_op_leaves})
    endforeach()
endforeach()

unset(_lf_soc_var)
unset(_lf_leaf_var)
unset(_lf_op)
unset(_lf_op_leaves)


# =========================================================================================
# 算子清单
# 用法（bash）：
#   cmake -DSOC=atlas_950 -P cmake/soc_ops_config.cmake     → 构建清单（含聚合名）
#   cmake -DSOC=atlas_950 -DOPS_REC_LIST_MODE=leaf -P ...   → 真实 vendor 清单（聚合名已展开）
# =========================================================================================
string(TOUPPER "${SOC}" _ops_scm_soc_upper)
if("${OPS_REC_LIST_MODE}" STREQUAL "leaf")
    string(REPLACE "OPS_REC_OPS_" "OPS_REC_LEAF_" _ops_scm_var_name "OPS_REC_OPS_${_ops_scm_soc_upper}")
else()
    set(_ops_scm_var_name "OPS_REC_OPS_${_ops_scm_soc_upper}")
endif()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    if(OPS_REC_QUERY_OP)
        ops_rec_op_leaves("${OPS_REC_QUERY_OP}" _ops_scm_query_leaves)
        foreach(_ops_scm_op IN LISTS _ops_scm_query_leaves)
            message("${_ops_scm_op}")
        endforeach()
    else()
        if(NOT DEFINED ${_ops_scm_var_name})
            message(FATAL_ERROR
                "soc_ops_config: unknown SOC '${SOC}' (no variable ${_ops_scm_var_name})")
        endif()
        foreach(_ops_scm_op IN LISTS ${_ops_scm_var_name})
            message("${_ops_scm_op}")
        endforeach()
    endif()
endif()
unset(_ops_scm_soc_upper)
unset(_ops_scm_var_name)
