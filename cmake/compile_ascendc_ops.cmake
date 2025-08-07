# =============================================================================
# Compile AscendC Ops
# =============================================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(NOT DEFINED ASCENDC_OP_DIRS)
    message(FATAL_ERROR "ASCENDC_OP_DIRS must be set before including this file")
endif()

if(NOT DEFINED OP_COMPILER_SCRIPT)
    message(FATAL_ERROR "OP_COMPILER_SCRIPT must be set before including this file")
endif()
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Cmake build type")
set(CMAKE_BUILD_PATH "" CACHE STRING "Cmake build path")

if(DEFINED ENV{SOC_VERSION})
    set(SOC_VERSION $ENV{SOC_VERSION})
else()
    set(SOC_VERSION "Ascend910B,Ascend310P" CACHE STRING "SOC version")
endif()
set(VENDOR_NAME "customize" CACHE STRING "Vendor name")
set(ASCENDC_INSTALL_PATH "" CACHE PATH "Install path")
if(NOT ASCENDC_INSTALL_PATH)
    message(FATAL_ERROR "ASCENDC_INSTALL_PATH must be set. Use -DASCENDC_INSTALL_PATH=<path>")
endif()

set(CLEAR OFF CACHE BOOL "Clear build output")
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
        --op_dirs="${ASCENDC_OP_DIRS}"
        --build_path=${CMAKE_BUILD_PATH}
        --build_type=${CMAKE_BUILD_TYPE}
        --soc_version="${SOC_VERSION}"
        --ascend_cann_package_path=${ASCEND_CANN_PACKAGE_PATH}
        --vendor_name=${VENDOR_NAME}
        --install_path=${ASCENDC_INSTALL_PATH}
        $<$<BOOL:${CLEAR}>:-c>
        $<$<BOOL:${INSTALL_OP}>:-i>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building custom operator using setup.py"
)
