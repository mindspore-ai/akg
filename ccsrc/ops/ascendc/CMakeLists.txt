# =============================================================================
# Collect Source Files from Ops Directories
# =============================================================================

# AscendC kernel files
if(DEFINED ENV{OP_DIRS})
    set(ASCENDC_OP_DIRS $ENV{OP_DIRS})
else()
    set(ASCENDC_OP_DIRS "")
    file(GLOB ITEMS "${CMAKE_CURRENT_SOURCE_DIR}/*")
    foreach(ITEM ${ITEMS})
        if(IS_DIRECTORY "${ITEM}" AND EXISTS "${ITEM}/op_host" AND EXISTS "${ITEM}/op_kernel")
            list(APPEND ASCENDC_OP_DIRS ${ITEM})
        endif()
    endforeach()
endif()

# AscendC src files
file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
list(FILTER SRC_FILES EXCLUDE REGEX ".*op_host.*")
list(FILTER SRC_FILES EXCLUDE REGEX ".*op_kernel.*")
set(ASCENDC_SRC_FILES ${SRC_FILES} PARENT_SCOPE)

set(OP_COMPILER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../../../scripts/op_compiler.py")

if(ASCENDC_OP_DIRS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/compile_ascendc_ops.cmake)
endif()

