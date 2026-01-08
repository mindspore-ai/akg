# ===========================================================================
# Torch-MLIR Source Download (no build)
# ===========================================================================
# Downloads Torch-MLIR source code and applies patches. The code will be built
# in the next stage.
# ===========================================================================

if(NOT COMMAND akg_add_pkg)
    include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)
endif()

set(TORCHMLIR_VERSION "2025.8.13" CACHE INTERNAL "Torch-MLIR 2025.8.13 daily version")
set(TORCHMLIR_COMMIT "155680c08e08bff6d2e6883415e3f5a1b474d96e" CACHE INTERNAL "Torch-MLIR commit hash")
set(TORCHMLIR_SHA256 "f39644fbaae44400da69862251d888bcee5c1fec54b62c77a1a8596958620705")
set(TORCHMLIR_URL "https://gitee.com/mirrors_llvm/torch-mlir/repository/archive/${TORCHMLIR_COMMIT}.zip")

akg_add_pkg(torch_mlir
    VER ${TORCHMLIR_VERSION}
    URL ${TORCHMLIR_URL}
    SHA256 ${TORCHMLIR_SHA256}
    SOURCEMODULES .
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/001-build-isolate-symbols.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/002-build-embedded.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/003-build-remove-tests.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/004-disable-torch-to-linalg.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/005-fix-compilation-errors.patch
)

set(TORCHMLIR_SOURCE_DIR ${torch_mlir_DIRPATH})

message(STATUS "Torch-MLIR source directory: ${TORCHMLIR_SOURCE_DIR}")
