# =============================================================================
# Collect Source Files from Ops Directories
# =============================================================================

# Set paths for AscendC ops compilation
set(OP_HOST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/kernel_impl/op_host")
set(OP_KERNEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/kernel_impl/op_kernel")
set(OP_COMPILER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../../../scripts/op_compiler.py")

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/compile_ascendc_ops.cmake)
