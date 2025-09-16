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

cmake_minimum_required(VERSION 3.14)

if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()
if(POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)
endif()

project(akg C CXX)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.5.0)
        message(FATAL_ERROR "GCC version must be 7.5.0 and above, but found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(AKG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(AKG_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/compiler")

set(CMAKE_CXX_STANDARD 17)

include(CheckCXXCompilerFlag)
include(CheckLanguage)
include(FetchContent)
# We may use projects that rely on PkgConfig
find_package(PkgConfig)
include(cmake/utils.cmake)
include(cmake/external_libs/symengine.cmake)
if(ENABLE_AKG)
  include(cmake/external_libs/llvm.cmake)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -Wno-unknown-pragmas")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set Path
get_filename_component(AKG_MLIR_SOURCE_DIR "${AKG_MLIR_SOURCE_DIR}" ABSOLUTE)
set(AKG_MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
get_filename_component(AKG_MLIR_BINARY_DIR "${AKG_MLIR_BINARY_DIR}" ABSOLUTE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${AKG_MLIR_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${AKG_MLIR_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${AKG_MLIR_BINARY_DIR}/bin)

set(AKG_MLIR_SRC_INCLUDE_DIR ${AKG_MLIR_SOURCE_DIR}/include)
set(AKG_MLIR_BIN_INCLUDE_DIR ${AKG_MLIR_BINARY_DIR}/include)
set(AKG_MLIR_BIN_LIB_DIR ${AKG_MLIR_BINARY_DIR}/lib)


#-------------------------------------------------------------------------------
# CUDA Backend Configuration
#-------------------------------------------------------------------------------
string(TOUPPER "${USE_CUDA}" USE_CUDA)

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
set(LLVM_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third-party/llvm-project")


if(NOT DEFINED LLVM_EXTERNAL_LIT)
  set(LLVM_EXTERNAL_LIT "${LLVM_TOOLS_DIR}/llvm-lit.py" CACHE PATH "PATH to llvm-lit")
endif()

# ----------------------------- Dependencies
set(MLIR_DIR "${LLVM_INSTALL_DIR}/lib/cmake/mlir")
include(cmake/MLIR.cmake)
set(JSON_TARBALL_URL "${AKG_MLIR_SOURCE_DIR}/../third-party/JSON-for-Modern-CPP-v3.10.1.zip")

#-------------------------------------------------------------------------------
# BishengIR
#-------------------------------------------------------------------------------

set(BISHENGIR_INCLUDE_DIR ${LLVM_SOURCE_PATH}/third-party/bishengir/bishengir/include)
include_directories(${BISHENGIR_INCLUDE_DIR})

set(BISHENGIR_BUILD_INCLUDE_DIR ${LLVM_BUILD_PATH}/tools/bishengir/bishengir/include)
include_directories(${BISHENGIR_BUILD_INCLUDE_DIR})

#-------------------------------------------------------------------------------
# AKG-MLIR
#-------------------------------------------------------------------------------

include(${CMAKE_SOURCE_DIR}/cmake/external_libs/glog.cmake)
include_directories(${AKG_MLIR_SRC_INCLUDE_DIR})
include_directories(${AKG_MLIR_BIN_INCLUDE_DIR})

add_subdirectory(${AKG_MLIR_SOURCE_DIR}/lib ${AKG_MLIR_BINARY_DIR}/lib)
add_subdirectory(${AKG_MLIR_SOURCE_DIR}/include ${AKG_MLIR_BINARY_DIR}/include)

# Build tools
add_subdirectory(${AKG_MLIR_SOURCE_DIR}/tools ${AKG_MLIR_BINARY_DIR}/tools)

set(AKG_MLIR_TOOLS_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

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

# Installation rules
if(ENABLE_AKG)
  install(TARGETS akg-opt DESTINATION lib${LIB_SUFFIX})
  install(TARGETS mindspore-translate DESTINATION lib${LIB_SUFFIX})
  install(TARGETS mlir_akgParallelLaunch_runtime DESTINATION lib${LIB_SUFFIX})
  if(USE_CUDA)
    install(TARGETS akg-ptx-replace DESTINATION lib${LIB_SUFFIX})
    install(TARGETS akg-translate DESTINATION lib${LIB_SUFFIX})
  endif()
else()
  install(TARGETS akg-opt DESTINATION ${AKG_MLIR_BINARY_DIR}/akg/bin)
  install(TARGETS mindspore-translate DESTINATION  ${AKG_MLIR_BINARY_DIR}/akg/bin)
  if(USE_CUDA)
    install(TARGETS akg-ptx-replace DESTINATION ${AKG_MLIR_BINARY_DIR}/akg/bin)
    install(TARGETS akg-translate DESTINATION ${AKG_MLIR_BINARY_DIR}/akg/bin)
  endif()
endif()

install(
  DIRECTORY
  ${AKG_MLIR_SOURCE_DIR}/../python/akg_mlir/
  DESTINATION ${AKG_MLIR_BINARY_DIR}/akg)
