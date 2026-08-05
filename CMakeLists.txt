# ============================================================================
# ops-rec 根 CMakeLists
# ----------------------------------------------------------------------------
# 支持多 SOC 编译：通过 cmake -DOPS_REC_SOC_VARIANT=atlas_950 指定变体。
# build.sh --socs 循环调用 pip wheel，每轮编译一种 variant。
#
# OPP 产物安装结构：
#   opp/<variant>/
#     vendors/
#       config.ini          <-- load_priority=ln_mul,...
#       ln_mul/              <--  vendor目录名
#         op_api/include/aclnnop/aclnn_*.h
#         op_api/lib/libcust_opapi.so
#         op_impl/ai_core/tbe/...
#       <next_op>/
#         ...
cmake_minimum_required(VERSION 3.18)
project(ops_rec LANGUAGES CXX)

# ---------------------------------------------------------------------------
# SOC 配置 — 从 OPS_REC_SOC_VARIANT 推导芯片宏
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/soc_config.cmake)

# ---------------------------------------------------------------------------
# 统一的 SOC 编译宏 — 供所有算子 CMakeLists.txt 使用
# 算子 C++ 代码中用 #ifdef SUPPORT_950 等方式区分 SOC 特性。
# ---------------------------------------------------------------------------
set(OPS_REC_SOC_CXX_DEFS "")
if(NPU_CHIP_ATLAS_950)
    set(OPS_REC_SOC_CXX_DEFS "-DSUPPORT_950=1")
elseif(NPU_CHIP_ATLAS_A2)
    set(OPS_REC_SOC_CXX_DEFS "-DSUPPORT_A2=1")
elseif(NPU_CHIP_ATLAS_A3)
    set(OPS_REC_SOC_CXX_DEFS "-DSUPPORT_A3=1")
endif()

# ---------------------------------------------------------------------------
# SOC 架构特化源码收集工具
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ops_rec_arch.cmake)

# ---------------------------------------------------------------------------
# OPP 安装前缀 — 每个 variant 独立目录
# ---------------------------------------------------------------------------
string(TOLOWER "${OPS_REC_SOC_VARIANT}" _variant_lower)
set(CUSTOM_OPP_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/opp/${_variant_lower}")
message(STATUS "CUSTOM_OPP_INSTALL_PREFIX=${CUSTOM_OPP_INSTALL_PREFIX}")

# ---------------------------------------------------------------------------
# 算子列表 — 从 soc_ops_config.cmake 加载（按 SOC 芯片管理）
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/soc_ops_config.cmake)

string(TOUPPER "${OPS_REC_SOC_VARIANT}" _variant_upper)
string(REPLACE "." "_" _variant_upper "${_variant_upper}")
set(OPS_REC_VENDORS ${OPS_REC_OPS_${_variant_upper}})
message(STATUS "ops-rec: vendors for ${OPS_REC_SOC_VARIANT} = ${OPS_REC_VENDORS}")

# ---------------------------------------------------------------------------
# 算子编译 — 仅在 OPS_REC_VENDORS 中存在的算子才被编译
# ---------------------------------------------------------------------------
file(GLOB _op_group_dirs LIST_DIRECTORIES true
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ascendc
    ${CMAKE_CURRENT_SOURCE_DIR}/ascendc/*)

foreach(_gd ${_op_group_dirs})
    if(_gd STREQUAL "common" OR NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ascendc/${_gd}")
        continue()
    endif()

    file(GLOB _op_dirs LIST_DIRECTORIES true
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ascendc/${_gd}
        ${CMAKE_CURRENT_SOURCE_DIR}/ascendc/${_gd}/*)

    foreach(_od ${_op_dirs})
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ascendc/${_gd}/${_od}/CMakeLists.txt")
            continue()
        endif()
        # 检查算子名是否在当前 variant 的编译列表中
        list(FIND OPS_REC_VENDORS ${_od} _vendor_idx)
        if(_vendor_idx EQUAL -1)
            message(STATUS "ops-rec: skipping operator '${_gd}/${_od}' (not in ${OPS_REC_SOC_VARIANT} list)")
            continue()
        endif()
        add_subdirectory(ascendc/${_gd}/${_od})
        message(STATUS "ops-rec: added operator '${_gd}/${_od}'")
    endforeach()
endforeach()

# ---------------------------------------------------------------------------
# Torch 适配层编译
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(CMAKE_PREFIX_PATH)
    set(PYTORCH_INSTALL_PATH "${CMAKE_PREFIX_PATH}")
else()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(PYTORCH_INSTALL_PATH "${PYTHON_SITE_PACKAGES}/torch")
endif()

if(NOT EXISTS "${PYTORCH_INSTALL_PATH}/include/torch/library.h")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch, os; print(os.path.dirname(torch.__file__))"
        OUTPUT_VARIABLE PYTORCH_INSTALL_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

get_filename_component(PYTHON_SITE_PACKAGES "${PYTORCH_INSTALL_PATH}/.." ABSOLUTE)
set(TORCH_NPU_INSTALL_PATH "${PYTHON_SITE_PACKAGES}/torch_npu")

# CXX11 ABI
if(NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; v=getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', 0); print(1 if v else 0)"
        OUTPUT_VARIABLE _GLIBCXX_USE_CXX11_ABI
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

# 从 OPS_REC_SOC_VARIANT 生成 .so 名
if(NOT DEFINED OPS_REC_SOC_VARIANT)
    set(OPS_REC_SOC_VARIANT "atlas_950")
endif()

string(TOLOWER "${OPS_REC_SOC_VARIANT}" _so_suffix)
string(REGEX REPLACE "^atlas_" "" _so_suffix "${_so_suffix}")
set(OPS_REC_TORCH_SO_NAME "ops_rec_atlas_${_so_suffix}")
message(STATUS "OPS_REC_SOC_VARIANT=${OPS_REC_SOC_VARIANT}, torch .so=${OPS_REC_TORCH_SO_NAME}")

# 设置 SOC 宏 — torch 适配层用
set(_OPS_REC_SOC_DEFS
    -D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}
    ${OPS_REC_SOC_CXX_DEFS}
)

# Torch 适配层源码 — 从每个算子的 torch_extension/ 目录收集
file(GLOB_RECURSE OPS_REC_TORCH_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/ascendc/*/*/torch_extension/*.cpp
)

if(OPS_REC_TORCH_PLUGIN_SRCS)
    add_library(${OPS_REC_TORCH_SO_NAME} SHARED ${OPS_REC_TORCH_PLUGIN_SRCS})
    target_compile_features(${OPS_REC_TORCH_SO_NAME} PRIVATE cxx_std_17)
    target_compile_options(${OPS_REC_TORCH_SO_NAME} PRIVATE ${_OPS_REC_SOC_DEFS})

    target_include_directories(${OPS_REC_TORCH_SO_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ascendc/common/torch_plugin
        ${TORCH_NPU_INSTALL_PATH}/include
        ${TORCH_NPU_INSTALL_PATH}/include/third_party/acl/inc
        ${PYTORCH_INSTALL_PATH}/include
        ${PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
        ${Python3_INCLUDE_DIRS}
        ${CANN_INCLUDE_DIR}
    )

    target_link_directories(${OPS_REC_TORCH_SO_NAME} PRIVATE
        ${PYTORCH_INSTALL_PATH}/lib
        ${TORCH_NPU_INSTALL_PATH}/lib
    )

    set(OPS_REC_TORCH_LIBS c10 torch torch_cpu torch_npu)
    find_library(TORCH_PYTHON_LIB torch_python HINTS "${PYTORCH_INSTALL_PATH}/lib" NO_DEFAULT_PATH)
    if(TORCH_PYTHON_LIB)
        list(APPEND OPS_REC_TORCH_LIBS torch_python)
    endif()
    target_link_libraries(${OPS_REC_TORCH_SO_NAME} PRIVATE ${OPS_REC_TORCH_LIBS} Python3::Python)

    set_target_properties(${OPS_REC_TORCH_SO_NAME} PROPERTIES
        PREFIX ""
        OUTPUT_NAME ${OPS_REC_TORCH_SO_NAME}
        BUILD_RPATH "${PYTORCH_INSTALL_PATH}/lib;${TORCH_NPU_INSTALL_PATH}/lib"
        INSTALL_RPATH "\$ORIGIN;${PYTORCH_INSTALL_PATH}/lib;${TORCH_NPU_INSTALL_PATH}/lib"
    )

    # .so 安装到包根目录
    install(TARGETS ${OPS_REC_TORCH_SO_NAME}
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
    )
endif()

# ===========================================================================
# kernel 产物安装（按算子拆分的 install-time 动态拷贝）
# ===========================================================================
get_property(_kernel_targets GLOBAL PROPERTY ASCENDC_KERNEL_TARGET)
string(TOLOWER "${ASCEND_COMPUTE_UNIT}" _soc_lower)

foreach(_kt ${_kernel_targets})
    get_property(_kernel_bin_dir GLOBAL PROPERTY _ASC_TGT_KERNEL_${_kt}_BINARY_DIR)

    if(NOT _kernel_bin_dir)
        continue()
    endif()

    # 从 kernel target 名提取 vendor/op 名
    string(REGEX REPLACE "_kernels$" "" _vendor_name "${_kt}")

    set(_kernel_src  "${_kernel_bin_dir}/binary/${_soc_lower}")
    set(_config_src  "${_kernel_bin_dir}/binary/config")
    set(_opsinfo_src "${_kernel_bin_dir}/tbe/op_info_cfg/ai_core/${_soc_lower}/aic-${_soc_lower}-ops-info.json")
    set(_vendor_prefix "opp/${_variant_lower}/vendors/${_vendor_name}/op_impl/ai_core/tbe")

    install(CODE "
        set(_kernel_src \"${_kernel_src}\")
        set(_config_src \"${_config_src}\")
        set(_opsinfo_src \"${_opsinfo_src}\")
        set(_prefix \"${_vendor_prefix}\")
        set(_soc \"${_soc_lower}\")

        if(EXISTS \"\${_kernel_src}\")
            file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/\${_prefix}/kernel/\${_soc}\"
                TYPE DIRECTORY FILES \"\${_kernel_src}/\")
        endif()

        if(EXISTS \"\${_config_src}\")
            file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/\${_prefix}/kernel/config\"
                TYPE DIRECTORY FILES \"\${_config_src}/\")
        endif()

        if(EXISTS \"\${_opsinfo_src}\")
            file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/\${_prefix}/config/\${_soc}\"
                TYPE FILE FILES \"\${_opsinfo_src}\")
        endif()
    ")

    message(STATUS "ops-rec: registered kernel artifact install rules for ${_kt} -> ${_vendor_prefix}")
endforeach()

# ===========================================================================
# config.ini 生成
# ===========================================================================
string(REPLACE ";" "," _vendor_csv "${OPS_REC_VENDORS}")
set(_VENDOR_CONFIG "${CMAKE_BINARY_DIR}/opp/vendors/config.ini")
file(WRITE "${_VENDOR_CONFIG}" "[vendor]\nload_priority=${_vendor_csv}\n")
install(FILES "${_VENDOR_CONFIG}"
    DESTINATION "opp/${_variant_lower}/vendors"
)

# ===========================================================================
# op_api 产物分发 + 清理顶层目录
# ===========================================================================
install(CODE "
    # 分发 libcust_opapi.so — 从 cmake config 目录推断 vendor 名
    file(GLOB _cmake_dirs
        \"\${CMAKE_INSTALL_PREFIX}/lib/cmake/*\")
    foreach(_cd \${_cmake_dirs})
        if(NOT IS_DIRECTORY \"\${_cd}\")
            continue()
        endif()
        get_filename_component(_vname \"\${_cd}\" NAME)
        set(_lib \"\${CMAKE_INSTALL_PREFIX}/lib/libcust_opapi.so\")
        if(EXISTS \"\${_lib}\")
            file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/opp/${_variant_lower}/vendors/\${_vname}/op_api/lib\"
                TYPE SHARED_LIBRARY
                FILES \"\${_lib}\")
        endif()
    endforeach()

    # 分发 aclnn 头文件
    file(GLOB _aclnn_headers
        \"\${CMAKE_INSTALL_PREFIX}/include/aclnn_*.h\")
    foreach(_h \${_aclnn_headers})
        foreach(_cd \${_cmake_dirs})
            if(NOT IS_DIRECTORY \"\${_cd}\")
                continue()
            endif()
            get_filename_component(_vname \"\${_cd}\" NAME)
            file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/opp/${_variant_lower}/vendors/\${_vname}/op_api/include/aclnnop\"
                TYPE FILE FILES \"\${_h}\")
        endforeach()
    endforeach()

    # 清理顶层 lib/ include/ — 已分发到 vendor 目录
    file(REMOVE_RECURSE
        \"\${CMAKE_INSTALL_PREFIX}/lib\"
        \"\${CMAKE_INSTALL_PREFIX}/include\")

")
