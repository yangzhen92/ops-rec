cmake_minimum_required(VERSION 3.18)

set(VENDOR_NAME "atten_metadata")
set(OP_NAME create_attn_metadata)
set(KERNEL_SO libtransformer_aicpu_kernels.so)
set(VENDOR_STAGE "${CMAKE_CURRENT_BINARY_DIR}/vendor/${VENDOR_NAME}")

if(NOT DEFINED ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()
set(ARCH_DIR "${ASCEND_HOME_PATH}/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux")
if(NOT IS_DIRECTORY "${ARCH_DIR}")
    set(ARCH_DIR "${ASCEND_HOME_PATH}")
endif()
set(LIB64 "${ARCH_DIR}/lib64")
find_program(AICPU_CXX NAMES aarch64-target-linux-gnu-g++
    PATHS "${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/bin" NO_DEFAULT_PATH)
if(NOT AICPU_CXX)
    message(FATAL_ERROR "AICPU cross compiler not found")
endif()

set(KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu/${OP_NAME}_aicpu.cpp")
set(KERNEL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${OP_NAME}_aicpu.cpp.o")
set(KERNEL_OUT "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_SO}")
file(GLOB_RECURSE KERNEL_DEP_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB ATTN_PLAN_COMMON_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../attn_plan/common/*.h")
add_custom_command(OUTPUT "${KERNEL_OUT}"
    COMMAND "${AICPU_CXX}" -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=1
        -Dgoogle=ascend_private -DDISABLE_COMPILE_V1 -fPIC -O2 -std=c++17 -fvisibility=hidden
        -I${ARCH_DIR}/pkg_inc/aicpu_common/context -I${ARCH_DIR}/pkg_inc/aicpu_common/context/common
        -I${ARCH_DIR}/pkg_inc/aicpu_common/context/cpu_proto -I${ARCH_DIR}/pkg_inc/aicpu_common/context/utils
        -I${ARCH_DIR}/pkg_inc/aicpu -I${ARCH_DIR}/include -I${ASCEND_HOME_PATH}/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu
        -I${CMAKE_CURRENT_SOURCE_DIR}/deps/attention_common/op_kernel
        -I${CMAKE_CURRENT_SOURCE_DIR}/deps/hstu_attn -I${CMAKE_CURRENT_SOURCE_DIR}/../attn_plan/common
        -isystem${ASCEND_HOME_PATH}/pkg_inc/base
        -c "${KERNEL_SRC}" -o "${KERNEL_OBJ}"
    COMMAND "${AICPU_CXX}" -shared "${KERNEL_OBJ}" -Wl,--whole-archive
        "${LIB64}/libaicpu_context.a" "${LIB64}/libbase_ascend_protobuf.a"
        -Wl,--no-whole-archive -Wl,-Bsymbolic -Wl,--exclude-libs=libbase_ascend_protobuf.a -s
        -o "${KERNEL_OUT}"
    DEPENDS "${KERNEL_SRC}" ${KERNEL_DEP_HEADERS} ${ATTN_PLAN_COMMON_HEADERS})
add_custom_target(aicpu_kernel ALL DEPENDS "${KERNEL_OUT}")

add_library(create_attn_metadata_opapi SHARED op_host/op_api/aclnn_${OP_NAME}.cpp op_host/op_api/l0_${OP_NAME}.cpp)
set_target_properties(create_attn_metadata_opapi PROPERTIES OUTPUT_NAME cust_opapi PREFIX "lib")
target_compile_definitions(create_attn_metadata_opapi PRIVATE _GLIBCXX_USE_CXX11_ABI=0 google=ascend_private
    ACLNN_LOG_FMT_CHECK OP_LOG_LIBOPAPI_ONLY BUILD_OPEN_PROJECT BUILD_OPEN_PROJECT_API=1)
target_compile_options(create_attn_metadata_opapi PRIVATE -fPIC -O2 -fvisibility=hidden)
target_include_directories(create_attn_metadata_opapi PRIVATE ${ASCEND_HOME_PATH}/include ${ARCH_DIR}/include
    ${ARCH_DIR}/include/aclnn ${ASCEND_HOME_PATH}/include/aclnn ${ARCH_DIR}/pkg_inc/aicpu
    ${ARCH_DIR}/pkg_inc/aicpu_common/context ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api)
target_include_directories(create_attn_metadata_opapi SYSTEM PRIVATE ${ASCEND_HOME_PATH}/pkg_inc/base)
target_link_directories(create_attn_metadata_opapi PRIVATE ${LIB64})
target_link_options(create_attn_metadata_opapi PRIVATE -s -Wl,-z,relro -Wl,-z,now -Wl,-Bsymbolic)
target_link_libraries(create_attn_metadata_opapi PRIVATE ${LIB64}/libnnopbase.so profapi ge_common_base ascend_dump ascendalog dl ${LIB64}/libopapi_math.so)

set(VENDOR_STAMP "${CMAKE_CURRENT_BINARY_DIR}/vendor/.stamp")
add_custom_command(OUTPUT "${VENDOR_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${VENDOR_STAGE}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${VENDOR_STAGE}/op_api/lib"
        "${VENDOR_STAGE}/op_api/include/aclnnop" "${VENDOR_STAGE}/op_impl/cpu/aicpu_kernel/impl"
        "${VENDOR_STAGE}/op_impl/cpu/config"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:create_attn_metadata_opapi>" "${VENDOR_STAGE}/op_api/lib/"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api/aclnn_${OP_NAME}.h" "${VENDOR_STAGE}/op_api/include/aclnnop/"
    COMMAND ${CMAKE_COMMAND} -E copy "${KERNEL_OUT}" "${VENDOR_STAGE}/op_impl/cpu/aicpu_kernel/impl/"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu/${OP_NAME}_aicpu.json" "${VENDOR_STAGE}/op_impl/cpu/config/cust_aicpu_kernel.json"
    COMMAND ${CMAKE_COMMAND} -E echo "Version=1.0" > "${VENDOR_STAGE}/version.info"
    COMMAND ${CMAKE_COMMAND} -E touch "${VENDOR_STAMP}"
    DEPENDS create_attn_metadata_opapi aicpu_kernel)
add_custom_target(create_attn_metadata_vendor ALL DEPENDS "${VENDOR_STAMP}")

install(DIRECTORY "${VENDOR_STAGE}/" DESTINATION "${CUSTOM_OPP_INSTALL_PREFIX}/vendors/${VENDOR_NAME}")
