# =========================================================================
# LLVM Source Build
# =========================================================================
# This file builds and installs LLVM/MLIR with akg_add_pkg and
# records the relevant paths for later stages.
# =========================================================================

message(STATUS "========================================")
message(STATUS "Preparing LLVM build")
message(STATUS "========================================")

set(TOP_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

if(NOT COMMAND akg_add_pkg)
    include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)
endif()

set(LLVM_VERSION "19.1.7" CACHE INTERNAL "llvmorg-19.1.7")
set(LLVM_COMMIT "cd708029e0b2869e80abe31ddb175f7c35361f90" CACHE INTERNAL "LLVM commit hash")
set(LLVM_SHA256 "a7e9c2b2f3c9774ff422693ab8b76e932615a5ef4a8a7ae04b968544bc47ca54")
set(LLVM_URL "https://gitee.com/mirrors/LLVM/repository/archive/${LLVM_COMMIT}.zip")

set(LLVM_CMAKE_OPTIONS
    -DCMAKE_BUILD_TYPE=Release
    -DPython3_FIND_VIRTUALENV=FIRST
    -DPython_FIND_VIRTUALENV=FIRST
    -DLLVM_ENABLE_PROJECTS=mlir
    -DLLVM_TARGETS_TO_BUILD=host
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DLLVM_INSTALL_UTILS=ON
)

akg_add_pkg(llvm_project
    VER ${LLVM_VERSION}
    URL ${LLVM_URL}
    SHA256 ${LLVM_SHA256}
    CMAKE_PATH llvm
    CMAKE_OPTION ${LLVM_CMAKE_OPTIONS}
    CUSTOM_CMAKE_GENERATOR Ninja
)

set(LLVM_BUILD_DIR "${llvm_project_DIRPATH}")
set(LLVM_DIR "${llvm_project_DIRPATH}/lib/cmake/llvm")
set(MLIR_DIR "${llvm_project_DIRPATH}/lib/cmake/mlir")

message(STATUS "LLVM build directory: ${LLVM_BUILD_DIR}")
message(STATUS "LLVM directory: ${LLVM_DIR}")
message(STATUS "MLIR directory: ${MLIR_DIR}")
