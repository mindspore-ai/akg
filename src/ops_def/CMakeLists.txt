cmake_minimum_required(VERSION 3.16)
project(Ops)

set(MS_EXTENSION_NAME "" CACHE STRING "Extension Name")
set(BUILD_EXTENSION_DIR "" CACHE STRING "Extension directory")
if (MS_EXTENSION_NAME STREQUAL "")
    message(FATAL_ERROR "MS_EXTENSION_NAME must be set. Use -DMS_EXTENSION_NAME=<name>")
endif()
if (BUILD_EXTENSION_DIR STREQUAL "")
    message(FATAL_ERROR "BUILD_EXTENSION_DIR must be set. Use -DBUILD_EXTENSION_DIR=<path>")
endif()

# Build ascendc kernels
add_subdirectory(ms_kernels_internal)

# Collect source files
# file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/module/*.cpp)
# 确保变量已经设置
if(NOT DEFINED INTERNAL_KERNEL_LIB_PATH)
    message(FATAL_ERROR "INTERNAL_KERNEL_LIB_PATH is not defined. Check if ms_kernels_internal subdirectory was processed correctly.")
endif()

if(NOT DEFINED MINDSPORE_INTERNAL_KERNELS_LIB)
    message(FATAL_ERROR "MINDSPORE_INTERNAL_KERNELS_LIB is not defined. Check if ms_kernels_internal subdirectory was processed correctly.")
endif()

set(LIB_DIR ${INTERNAL_KERNEL_LIB_PATH})
set(LIBS ${MINDSPORE_INTERNAL_KERNELS_LIB})

# 定义多个include目录
set(INCLUDE_DIRS 
    ${INTERNAL_KERNEL_INC_PATH}
    # 在这里添加更多的include目录，例如：
    # ${CMAKE_CURRENT_SOURCE_DIR}/include
    # ${CMAKE_CURRENT_SOURCE_DIR}/third_party/include
    # /path/to/another/include
)

set(SRC_FILES ${MS_KERNELS_INTERNAL_SRC_FILES})

message(STATUS "LIB_DIR: ${LIB_DIR}")
message(STATUS "LIBS: ${LIBS}")
message(STATUS "SRC_FILES: ${SRC_FILES}")
message(STATUS "INCLUDE_DIRS: ${INCLUDE_DIRS}")

# 将include目录列表转换为cflags格式
set(CFLAGS_INCLUDES "")
foreach(INC_DIR_ITEM ${INCLUDE_DIRS})
    if(CFLAGS_INCLUDES STREQUAL "")
        set(CFLAGS_INCLUDES "-I${INC_DIR_ITEM}")
    else()
        set(CFLAGS_INCLUDES "${CFLAGS_INCLUDES} -I${INC_DIR_ITEM}")
    endif()
endforeach()

message(STATUS "CFLAGS_INCLUDES: ${CFLAGS_INCLUDES}")

# Generate a temporary python script file to build custom ops with MindSpore's CustomOpBuilder
set(PYTHON_SCRIPT_PATH "${CMAKE_BINARY_DIR}/build_custom_with_ms.py")
file(WRITE ${PYTHON_SCRIPT_PATH} "
import mindspore as ms
src_files = '${SRC_FILES}'.split(';')
ms.ops.CustomOpBuilder(
    name='${MS_EXTENSION_NAME}',
    sources=src_files,
    backend='Ascend',
    cflags='${CFLAGS_INCLUDES}',
    ldflags='-L${INTERNAL_KERNEL_LIB_PATH} -l${LIBS}',
    build_dir='${BUILD_EXTENSION_DIR}'
).build()
")

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_target(
    BuildCustomOp ALL
    COMMAND cd ${CMAKE_BINARY_DIR} && ${Python3_EXECUTABLE} ${PYTHON_SCRIPT_PATH}
    DEPENDS ${ASCENDC_TARGET_NAME}
    COMMENT "Building custom operator with MindSpore"
)
