# =============================================================================
# Compile AscendC Ops
# =============================================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# OP_HOST_PATH, OP_KERNEL_PATH, and OP_COMPILER_SCRIPT should be set by the calling CMakeLists.txt
if(NOT DEFINED OP_HOST_PATH)
    message(FATAL_ERROR "OP_HOST_PATH must be set before including this file")
endif()

if(NOT DEFINED OP_KERNEL_PATH)
    message(FATAL_ERROR "OP_KERNEL_PATH must be set before including this file")
endif()

if(NOT DEFINED OP_COMPILER_SCRIPT)
    message(FATAL_ERROR "OP_COMPILER_SCRIPT must be set before including this file")
endif()
set(SOC_VERSION "Ascend910,Ascend910B,Ascend310P" CACHE STRING "SOC version")
set(VENDOR_NAME "customize" CACHE STRING "Vendor name")
set(ASCENDC_INSTALL_PATH "" CACHE PATH "Install path")
if(NOT ASCENDC_INSTALL_PATH)
    message(FATAL_ERROR "ASCENDC_INSTALL_PATH must be set. Use -DASCENDC_INSTALL_PATH=<path>")
endif()

set(CLEAR ON CACHE BOOL "Clear build output")
set(INSTALL_OP OFF CACHE BOOL "Install custom op")
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    message(STATUS "Using ASCEND_HOME_PATH environment variable: ${ASCEND_HOME_PATH}")
else()
    set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()

add_custom_target(
    build_custom_op ALL
    COMMAND ${Python3_EXECUTABLE} ${OP_COMPILER_SCRIPT}
        -o=${OP_HOST_PATH}
        -k=${OP_KERNEL_PATH}
        --soc_version=${SOC_VERSION}
        --ascend_cann_package_path=${ASCEND_CANN_PACKAGE_PATH}
        --vendor_name=${VENDOR_NAME}
        --install_path=${ASCENDC_INSTALL_PATH}
        $<$<BOOL:${CLEAR}>:-c>
        $<$<BOOL:${INSTALL_OP}>:-i>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building custom operator using setup.py"
)
