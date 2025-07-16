cmake_minimum_required(VERSION 3.16)
project(CustomOpBuild)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(OP_HOST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/kernel/op_host" CACHE PATH "Path to op_host")
set(OP_KERNEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/kernel/op_kernel" CACHE PATH "Path to op_kernel")
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

set(INCLUDE_PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENT_SOURCE_DIR}/pyboost"
    "${CMAKE_CURRENT_SOURCE_DIR}/graphmode"
)
list(JOIN INCLUDE_PATHS ";" ASCENDC_INC_PATH)
set(ASCENDC_INC_PATH "${ASCENDC_INC_PATH}" PARENT_SCOPE)

file(GLOB_RECURSE ASCENDC_GRAPH_MODE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/graphmode/*.cc)
file(GLOB_RECURSE ASCENDC_BOOST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/pyboost/*.cc)
set(ASCENDC_SRC_FILES 
    ${ASCENDC_GRAPH_MODE_SRC_FILES} 
    ${ASCENDC_BOOST_SRC_FILES} 
    PARENT_SCOPE
)
if(NOT ASCENDC_SRC_FILES)
    message(WARNING "No .cc files found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()


add_custom_target(
    build_custom_op ALL
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/op_compiler.py
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
