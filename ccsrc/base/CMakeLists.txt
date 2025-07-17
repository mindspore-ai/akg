# =============================================================================
# Base Source Files Collection
# =============================================================================

# Collect all .cc files recursively from the base directory
file(GLOB_RECURSE BASE_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

# Debug output to verify collected files
message(STATUS "BASE_SRC_FILES files found:")
foreach(SRC_FILE ${BASE_SRC_FILES})
    message(STATUS "  ${SRC_FILE}")
endforeach()

# Make BASE_SRC_FILES available to parent scope
set(BASE_SRC_FILES ${BASE_SRC_FILES} PARENT_SCOPE)

# =============================================================================
# Include Directories
# =============================================================================

# Set include directories for base module
set(BASE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ms_kernels_internal
    ${CMAKE_CURRENT_SOURCE_DIR}/ms_kernels_internal/pyboost
    ${CMAKE_CURRENT_SOURCE_DIR}/ms_kernels_internal/graphmode
    ${CMAKE_CURRENT_SOURCE_DIR}/ascendc
    ${CMAKE_CURRENT_SOURCE_DIR}/ascendc/pyboost
    ${CMAKE_CURRENT_SOURCE_DIR}/ascendc/graphmode
)

# Make include directories available to parent scope
set(BASE_INCLUDE_DIRS ${BASE_INCLUDE_DIRS} PARENT_SCOPE)

message(STATUS "BASE_INCLUDE_DIRS: ${BASE_INCLUDE_DIRS}")
