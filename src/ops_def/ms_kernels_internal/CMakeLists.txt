cmake_minimum_required(VERSION 3.16)
project(ms_kernels_internal)

# Find Python to get MindSpore installation path
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Allow user to override MindSpore path
if(DEFINED ENV{MINDSPORE_PATH})
    set(MS_PATH $ENV{MINDSPORE_PATH})
    message(STATUS "Using MINDSPORE_PATH environment variable: ${MS_PATH}")
else()
    # Get MindSpore installation path using Python
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import mindspore as ms; print(ms.__file__)"
        OUTPUT_VARIABLE MS_MODULE_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYTHON_RESULT
    )
    
    if(NOT PYTHON_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to find MindSpore installation. Please ensure MindSpore is installed or set MINDSPORE_PATH environment variable.")
    endif()
    
    # Extract directory from MindSpore module path
    get_filename_component(MS_PATH ${MS_MODULE_PATH} DIRECTORY)
endif()

# Set MindSpore internal kernel paths (matching Python logic)
set(INCLUDE_PATHS
    "${MS_PATH}/lib/plugin/ascend/ms_kernels_internal/internal_kernel"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/pyboost"
    "${CMAKE_CURRENT_SOURCE_DIR}/graphmode"
)
list(JOIN INCLUDE_PATHS ";" INTERNAL_KERNEL_INC_PATH)
set(INTERNAL_KERNEL_INC_PATH "${INTERNAL_KERNEL_INC_PATH}" PARENT_SCOPE)
set(INTERNAL_KERNEL_LIB_PATH "${MS_PATH}/lib/plugin/ascend" PARENT_SCOPE)

# Debug: Print the exact value being set
message(STATUS "Setting INTERNAL_KERNEL_LIB_PATH to: '${MS_PATH}/lib/plugin/ascend'")
message(STATUS "MS_PATH value: '${MS_PATH}'")

# Print paths for debugging
message(STATUS "MindSpore path: ${MS_PATH}")
message(STATUS "Internal kernel include path: ${INTERNAL_KERNEL_INC_PATH}")
message(STATUS "Internal kernel library path: ${INTERNAL_KERNEL_LIB_PATH}")

# Check if paths exist
foreach(INCLUDE_PATH ${INCLUDE_PATHS})
    if(NOT EXISTS ${INCLUDE_PATH})
        message(WARNING "Include path does not exist: ${INCLUDE_PATH}")
        message(WARNING "This may cause compilation errors if headers are needed")
    endif()
endforeach()

if(NOT EXISTS ${INTERNAL_KERNEL_LIB_PATH})
    message(WARNING "Internal kernel library path does not exist: ${INTERNAL_KERNEL_LIB_PATH}")
    message(WARNING "This may cause linking errors")
endif()

# Check for mindspore_internal_kernels library
find_library(MINDSPORE_INTERNAL_KERNELS_LIB
    NAMES mindspore_internal_kernels
    PATHS ${INTERNAL_KERNEL_LIB_PATH}
    NO_DEFAULT_PATH
)

if(MINDSPORE_INTERNAL_KERNELS_LIB)
    message(STATUS "Found mindspore_internal_kernels library: ${MINDSPORE_INTERNAL_KERNELS_LIB}")
    set(MINDSPORE_INTERNAL_KERNELS_LIB "${MINDSPORE_INTERNAL_KERNELS_LIB}" PARENT_SCOPE)
else()
    message(WARNING "mindspore_internal_kernels library not found in ${INTERNAL_KERNEL_LIB_PATH}")
    message(WARNING "Will try to link with -lmindspore_internal_kernels")
    set(MINDSPORE_INTERNAL_KERNELS_LIB "mindspore_internal_kernels" PARENT_SCOPE)
endif()

# Collect source files
file(GLOB_RECURSE GRAPH_MODE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/graphmode/*.cc)
file(GLOB_RECURSE PYTHON_BOOST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/pyboost/*.cc)
file(GLOB COMMON_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

set(MS_KERNELS_INTERNAL_SRC_FILES 
    ${GRAPH_MODE_SRC_FILES} 
    ${PYTHON_BOOST_SRC_FILES} 
    ${COMMON_SRC_FILES}
    PARENT_SCOPE
)

if(NOT MS_KERNELS_INTERNAL_SRC_FILES)
    message(WARNING "No .cc files found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()
