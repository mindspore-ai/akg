# =========================================================================
# MLIR Dependency Management
# =========================================================================
# This file manages MLIR dependency with the following priority:
# 1. User-specified LLVM_DIR and MLIR_DIR
# 2. System-installed LLVM/MLIR (via find_package)
# 3. Automatic download and build from source
# =========================================================================

find_package(MLIR 19.1 CONFIG)

if(MLIR_FOUND)
    message(STATUS "Found system-installed MLIR:")
    message(STATUS "  MLIR version: ${LLVM_PACKAGE_VERSION}")
    message(STATUS "  MLIR_DIR: ${MLIR_DIR}")
else()
    if(AKG_BUILD_MLIR)
        message(STATUS "MLIR not found, will build from source (AKG_BUILD_MLIR=ON)")
        message(STATUS "Warning This will take 30 mins, required disk space: ~20GB")

        # LLVM/MLIR version and source configuration
        set(MLIR_VERSION "19.1.7" CACHE INTERNAL "LLVM version")
        if(ENABLE_GITEE OR ENABLE_GITEE_EULER)
            set(REQ_URL "https://gitee.com/mirrors/LLVM/repository/archive/llvmorg-${MLIR_VERSION}.zip")
            set(SHA256 "367bb984b3b2600d6f51e6b43b140f8753d7dc782d1b9c1d8aa7df0d4315a07b")
        else()
            set(REQ_URL "https://github.com/llvm/llvm-project/archive/llvmorg-${MLIR_VERSION}.zip")
            set(SHA256 "61277aef807cb7307cb0838895e649609bbf0d7da5e158a523f6b05634432ef8")
        endif()

        set(LLVM_CMAKE_OPTIONS
            -DLLVM_BUILD_EXAMPLES=OFF
            -DLLVM_ENABLE_PROJECTS=mlir;clang;openmp
            -DLLVM_ENABLE_RUNTIMES=compiler-rt
            -DLLVM_OPTIMIZED_TABLEGEN=ON
            -DLLVM_ENABLE_OCAMLDOC=OFF
            -DLLVM_ENABLE_BINDINGS=OFF
            -DLLVM_INSTALL_UTILS=OFF
            -DCMAKE_BUILD_TYPE=Release
            -DLLVM_ENABLE_ASSERTIONS=ON
            -DLLVM_ENABLE_RTTI=ON
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON
            -DLLVM_ENABLE_EH=ON
        )

        akg_add_pkg(MLIR
            VER ${MLIR_VERSION}
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CMAKE_PATH llvm
            CMAKE_OPTION ${LLVM_CMAKE_OPTIONS}
            CUSTOM_CMAKE_GENERATOR Ninja)
    else()
        message(FATAL_ERROR
            "=======================================\n"
            "MLIR NOT FOUND\n"
            "========================================\n"
            "MLIR 19.1.7 is required but not found on your system.\n"
            "Options to resolve:\n"
            "  1. Install LLVM/MLIR 19.1.7 and set environment variables:\n"
            "     export LLVM_DIR=/path/to/llvm/lib/cmake/llvm\n"
            "     export MLIR_DIR=/path/to/llvm/lib/cmake/mlir\n"
            "  2. Or specify via CMake variables:\n"
            "     cmake -DLLVM_DIR=/path/to/llvm -DMLIR_DIR=/path/to/mlir ..\n"
            "  3. Or enable automatic build:\n"
            "     cmake -DAKG_BUILD_MLIR=ON ..\n"
            "========================================\n"
        )
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

################################################################################
# Setup python.
################################################################################
if(AKG_ENABLE_BINDINGS_PYTHON)
    include(AddMLIRPython)
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(BUILD_SHARED_LIBS ${LLVM_ENABLE_SHARED_LIBS} CACHE BOOL "" FORCE)
message(STATUS "BUILD_SHARED_LIBS: " ${BUILD_SHARED_LIBS})

# Declare the library associated with a translation.
function(add_akg_translation_library name)
    set_property(GLOBAL APPEND PROPERTY AKG_TRANSLATION_LIBS ${name})
    set_property(GLOBAL APPEND PROPERTY MINDSPORE_TRANSLATION_LIBS ${name})
    add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction()

get_property(MLIR_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)
