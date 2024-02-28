# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(CMAKE_CXX_STANDARD 17)

include(CheckCXXCompilerFlag)
include(CheckLanguage)
include(FetchContent)
# We may use projects that rely on PkgConfig
find_package(PkgConfig)
include(cmake/utils.cmake)
include(cmake/external_libs/symengine.cmake)
include(cmake/external_libs/llvm.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -Wno-unknown-pragmas")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set Path
get_filename_component(AKG_MLIR_SOURCE_DIR "${AKG_MLIR_SOURCE_DIR}" ABSOLUTE)
message("AKG_MLIR_SOURCE_DIR = ${AKG_MLIR_SOURCE_DIR}")

set(AKG_MLIR_BINARY_DIR ${CMAKE_BINARY_DIR}/akg-mlir)
get_filename_component(AKG_MLIR_BINARY_DIR "${AKG_MLIR_BINARY_DIR}" ABSOLUTE)
message("AKG_MLIR_BINARY_DIR = ${AKG_MLIR_BINARY_DIR}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${AKG_MLIR_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${AKG_MLIR_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${AKG_MLIR_BINARY_DIR}/bin)

set(AKG_MLIR_SRC_INCLUDE_DIR ${AKG_MLIR_SOURCE_DIR}/include)
set(AKG_MLIR_BIN_INCLUDE_DIR ${AKG_MLIR_BINARY_DIR}/include)
set(AKG_MLIR_BIN_LIB_DIR ${AKG_MLIR_BINARY_DIR}/lib)


#-------------------------------------------------------------------------------
# CUDA Backend Configuration
#-------------------------------------------------------------------------------
string(TOUPPER ${USE_CUDA} USE_CUDA)

if(USE_CUDA STREQUAL "ON")
  find_package(CUDA  REQUIRED)
  add_definitions(-DUSE_CUDA)
  message(STATUS "CUDA backend enabled")
else()
  message(STATUS "CUDA backend disabled")
endif()

#-------------------------------------------------------------------------------
# LLVM Configuration
#-------------------------------------------------------------------------------

# Temporarily support legacy variable
if (DEFINED LLVM_INSTALL_PATH)
  set(LLVM_BUILD_PATH ${LLVM_INSTALL_PATH})
  message(WARNING
    "Deprecated variable LLVM_INSTALL_PATH was set. "
    "It is recommended to either set cmake variable $LLVM_BUILD_PATH "
    "or environment variable $LLVM_DIR."
  )
endif ()

set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)

message("LLVM_BUILD_PATH: ${LLVM_BUILD_PATH}")
set(LLVM_INSTALL_DIR "${LLVM_BUILD_PATH}")
set(LLVM_TOOLS_DIR "${LLVM_BUILD_PATH}/bin")


if(NOT DEFINED LLVM_EXTERNAL_LIT)
  set(LLVM_EXTERNAL_LIT "${LLVM_TOOLS_DIR}/llvm-lit.py" CACHE PATH "PATH to llvm-lit")
endif()

# ----------------------------- Dependencies
set(MLIR_DIR "${LLVM_INSTALL_DIR}/lib/cmake/mlir")
include(cmake/MLIR.cmake)
set(JSON_TARBALL_URL "${AKG_MLIR_SOURCE_DIR}/../third-party/JSON-for-Modern-CPP-v3.10.1.zip")


#-------------------------------------------------------------------------------
# PolyTOPS
#-------------------------------------------------------------------------------

# Configure and build PolyTOPS
set(POLYTOPS_USE_OSL OFF)
set(POLYTOPS_USE_ISL ON)
set(POLYTOPS_USE_FPL ON)
set(POLYTOPS_USE_PICOJSON OFF)
set(POLYTOPS_USE_QIUQI_IP OFF)
set(POLYTOPS_ENABLE_SANITIZERS OFF)
set(POLYTOPS_BUILD_MAIN_TOOL OFF)
set(POLYTOPS_ENABLE_MLIR_DIALECT ON)
set(POLYTOPS_BUILD_MLIR_TOOL OFF)

set(POLYTOPS_FPL_FOUND ON)
set(POLYTOPS_MLIR_FOUND ON)
set(POLYTOPS_LLVM_FOUND ON)

if (AKG_MLIR_BUILD_ISL)
  # Make PolyTOPS build isl from the bundled third-party tarball
  set(ISL_FORCE_DOWNLOAD ON)
  set(ISL_DOWNLOAD_METHOD tarball)
  set(ISL_TARBALL_URL "${AKG_MLIR_SOURCE_DIR}/../third-party/isl-0.26.tar.gz" CACHE STRING "osl: tarball url or path" FORCE)
endif ()

include(cmake/AddPolytops.cmake)

if (POLYTOPS_BUILD_MAIN_TOOL)
  target_link_libraries(polytops PUBLIC ${MLIR_LIBS})
  target_include_directories(polytops PUBLIC "${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}")
endif()
if (POLYTOPS_BUILD_SHARED_LIBRARY)
  target_link_libraries(libpolytops-shared PUBLIC ${MLIR_LIBS})
  target_include_directories(libpolytops-shared PUBLIC "${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}")
endif()
if (POLYTOPS_BUILD_STATIC_LIBRARY)
  target_link_libraries(libpolytops-static PUBLIC ${MLIR_LIBS})
  target_include_directories(libpolytops-static PUBLIC "${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}")
endif()


#-------------------------------------------------------------------------------
# AKG-MLIR
#-------------------------------------------------------------------------------

include_directories(${AKG_MLIR_SRC_INCLUDE_DIR})
include_directories(${AKG_MLIR_BIN_INCLUDE_DIR})

add_subdirectory(${AKG_MLIR_SOURCE_DIR}/lib ${AKG_MLIR_BINARY_DIR}/lib)
add_subdirectory(${AKG_MLIR_SOURCE_DIR}/include ${AKG_MLIR_BINARY_DIR}/include)

# Build tools
add_subdirectory(${AKG_MLIR_SOURCE_DIR}/tools ${AKG_MLIR_BINARY_DIR}/tools)

set(AKG_MLIR_TOOLS_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

install(DIRECTORY ${AKG_MLIR_SRC_INCLUDE_DIR}/akg
  DESTINATION include
  COMPONENT akg-mlir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  )

install(DIRECTORY ${AKG_MLIR_BIN_INCLUDE_DIR}/akg
  DESTINATION include
  COMPONENT akg-mlir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
  )

option(AKG_INCLUDE_TESTS "Generate build targets for the akg-mlir tests." ON)
if (AKG_INCLUDE_TESTS)
  add_definitions(-DAKG_INCLUDE_TESTS)
endif()

# Build tests
if (AKG_INCLUDE_TESTS)
  set(AKG_TEST_BUILD_DIR "${AKG_MLIR_BINARY_DIR}/test")
  set(AKG_TEST_SOURCE_DIR "${AKG_MLIR_SOURCE_DIR}/../test")
  add_subdirectory(${AKG_TEST_SOURCE_DIR} ${AKG_TEST_BUILD_DIR})
endif()
