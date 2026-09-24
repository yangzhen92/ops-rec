set(VENDOR_NAME "packed_block_sparse")
set(ASCEND_AUTOGEN_PATH "${CMAKE_CURRENT_BINARY_DIR}/autogen")
file(MAKE_DIRECTORY "${ASCEND_AUTOGEN_PATH}")
set(OPP_VENDOR_INSTALL_PREFIX "${CUSTOM_OPP_INSTALL_PREFIX}/vendors/${VENDOR_NAME}")

npu_op_package(${VENDOR_NAME}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${OPP_VENDOR_INSTALL_PREFIX}
        VENDOR_NAME ${VENDOR_NAME}
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
