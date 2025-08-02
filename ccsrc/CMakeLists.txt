cmake_minimum_required(VERSION 3.16)
project(ms_custom_ops)

# =============================================================================
# Configuration and Validation
# =============================================================================

set(MS_EXTENSION_NAME "" CACHE STRING "Extension Name")
set(BUILD_EXTENSION_DIR "" CACHE STRING "Extension directory")
if (MS_EXTENSION_NAME STREQUAL "")
    message(FATAL_ERROR "MS_EXTENSION_NAME must be set. Use -DMS_EXTENSION_NAME=<name>")
endif()
if (BUILD_EXTENSION_DIR STREQUAL "")
    message(FATAL_ERROR "BUILD_EXTENSION_DIR must be set. Use -DBUILD_EXTENSION_DIR=<path>")
endif()

# =============================================================================
# Build Dependencies
# =============================================================================

# Include find_lib.cmake to set up MindSpore paths
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/find_ms_internal_kernels_lib.cmake)

add_subdirectory(base)
add_subdirectory(ops)

# Set library and source variables
set(LIB_DIR ${INTERNAL_KERNEL_LIB_PATH})
set(LIBS ${MINDSPORE_INTERNAL_KERNELS_LIB})
set(SRC_FILES ${BASE_SRC_FILES} ${OPS_SRC_FILES})
set(INCLUDE_DIRS ${BASE_INCLUDE_DIRS} ${INTERNAL_KERNEL_INC_PATH})

# =============================================================================
# Debug Output and Validation
# =============================================================================

message(STATUS "LIB_DIR: ${LIB_DIR}")
message(STATUS "LIBS: ${LIBS}")
message(STATUS "SRC_FILES: ${SRC_FILES}")
message(STATUS "INCLUDE_DIRS: ${INCLUDE_DIRS}")

# =============================================================================
# Build Configuration
# =============================================================================

# Convert include directories to CFLAGS format
set(CFLAGS_INCLUDES "")
foreach(INC_DIR_ITEM ${INCLUDE_DIRS})
    if(CFLAGS_INCLUDES STREQUAL "")
        set(CFLAGS_INCLUDES "-I${INC_DIR_ITEM}")
    else()
        set(CFLAGS_INCLUDES "${CFLAGS_INCLUDES} -I${INC_DIR_ITEM}")
    endif()
endforeach()

message(STATUS "CFLAGS_INCLUDES: ${CFLAGS_INCLUDES}")

# =============================================================================
# Get YAML files
# =============================================================================
function(get_yaml_files YAML_FILES OUTPUT_VAR)
    set(YAML_STRING "[")
    set(FIRST_ITEM TRUE)
    foreach(YAML_FILE ${YAML_FILES})
        if(NOT FIRST_ITEM)
            set(YAML_STRING "${YAML_STRING}, ")
        endif()
        set(YAML_STRING "${YAML_STRING}'${YAML_FILE}'")
        set(FIRST_ITEM FALSE)
    endforeach()
    set(YAML_STRING "${YAML_STRING}]")
    set(${OUTPUT_VAR} "${YAML_STRING}" PARENT_SCOPE)
endfunction()

file(GLOB_RECURSE OPS_YAML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../yaml/*_op.yaml")
message(STATUS "OPS_YAML_FILES: ${OPS_YAML_FILES}")
get_yaml_files("${OPS_YAML_FILES}" DEF_YAML_STRING)

file(GLOB_RECURSE DOC_YAML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../yaml/*_doc.yaml")
message(STATUS "DOC_YAML_FILES: ${DOC_YAML_FILES}")
get_yaml_files("${DOC_YAML_FILES}" DOC_YAML_STRING)

# =============================================================================
# Custom Op Builder
# =============================================================================

# Generate Python script for building custom ops with MindSpore's CustomOpBuilder
set(ENABLE_DEBUG False)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(ENABLE_DEBUG True)
endif()
set(PYTHON_SCRIPT_PATH "${CMAKE_BINARY_DIR}/build_custom_with_ms.py")
file(WRITE ${PYTHON_SCRIPT_PATH} "
import mindspore as ms
src_files = '${SRC_FILES}'.split(';')
ms.ops.CustomOpBuilder(
    name='${MS_EXTENSION_NAME}',
    sources=src_files,
    op_def=${DEF_YAML_STRING},
    op_doc=${DOC_YAML_STRING},
    backend='Ascend',
    cflags='${CFLAGS_INCLUDES}',
    ldflags='-L${INTERNAL_KERNEL_LIB_PATH} -l${LIBS}',
    build_dir='${BUILD_EXTENSION_DIR}',
    debug_mode=${ENABLE_DEBUG}
).build()
")

# Find Python and create custom target
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_target(
    BuildCustomOp ALL
    COMMAND cd ${CMAKE_BINARY_DIR} && ${Python3_EXECUTABLE} ${PYTHON_SCRIPT_PATH}
    DEPENDS ${ASCENDC_TARGET_NAME}
    COMMENT "Building custom operator with MindSpore"
)
