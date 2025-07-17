# =============================================================================
# Collect Source Files from Ops Directories
# =============================================================================

add_subdirectory(ascendc)

file(GLOB_RECURSE MS_KERNELS_INTERNAL_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/ms_kernels_internal/*.cc")

# Collect all .cc files from ops/ascendc directory (excluding kernel_impl)
file(GLOB_RECURSE ASCENDC_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/ascendc/*.cc")
list(FILTER ASCENDC_SRC_FILES EXCLUDE REGEX ".*kernel_impl.*")

set(OPS_SRC_FILES ${MS_KERNELS_INTERNAL_SRC_FILES} ${ASCENDC_SRC_FILES} PARENT_SCOPE)
