# =============================================================================
# Collect Source Files from Ops Directories
# =============================================================================

add_subdirectory(ascendc)
add_subdirectory(ms_kernels_internal)

set(OPS_SRC_FILES ${MS_KERNELS_INTERNAL_SRC_FILES} ${ASCENDC_SRC_FILES} PARENT_SCOPE)
