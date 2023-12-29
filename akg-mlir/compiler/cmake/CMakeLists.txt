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

project(akg-mlir C CXX)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.5.0)
        message(FATAL_ERROR "GCC version must be 7.5.0 and above, but found ${CMAKE_CXX_COMPILER_VERSION}")
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 11.1.0)
        message(WARNING "GCC version ${CMAKE_CXX_COMPILER_VERSION} is greater than 11.1.0, may cause unknown problems.")
    endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)

include(CheckCXXCompilerFlag)
include(CheckLanguage)
include(FetchContent)
include(ExternalProject)
# We may use projects that rely on PkgConfig
find_package(PkgConfig)

option(AKG_MLIR_BUILD_SYMENGINE "Build SymEngine along AKG-MLIR" OFF)
option(AKG_MLIR_BUILD_ISL "Build isl along AKG-MLIR" ON)
option(AKG_MLIR_BUILD_OSL "Build osl along AKG-MLIR" ON)

# Set Path
set(AKG_MLIR_SOURCE_DIR ${PROJECT_SOURCE_DIR}/..)
get_filename_component(AKG_MLIR_SOURCE_DIR "${AKG_MLIR_SOURCE_DIR}" ABSOLUTE)
message("AKG_MLIR_SOURCE_DIR = ${AKG_MLIR_SOURCE_DIR}")

set(AKG_MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
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
set(USE_CUDA AUTO CACHE STRING "Enable CUDA backend if available (ON, OFF, or AUTO)")

string(TOUPPER ${USE_CUDA} USE_CUDA)

if(USE_CUDA STREQUAL "AUTO")
  find_package(CUDA  QUIET)
  if(CUDA_FOUND)
    message(STATUS "CUDA found")
    set(USE_CUDA ON)
  else()
    set(USE_CUDA OFF)
  endif()
endif()

if(USE_CUDA STREQUAL "ON")
  find_package(CUDA  REQUIRED)
  add_definitions(-DUSE_CUDA)
  message(STATUS "CUDA backend enabled")
else()
  message(STATUS "CUDA backend disabled")
endif()

#-------------------------------------------------------------------------------
# SymEngine Configuration
#-------------------------------------------------------------------------------

# Temporarily support legacy variable
if (DEFINED SYMENGINE_INSTALL_PATH)
  set(SYMENGINE_BUILD_PATH ${SYMENGINE_INSTALL_PATH})
  message(WARNING
    "Deprecated variable SYMENGINE_INSTALL_PATH was set. "
    "It is recommended to either set cmake variable $SYMENGINE_BUILD_PATH "
    "or environment variable $SYMENGINE_DIR."
  )
endif ()

# First check if SymEngine was installed in system default paths or user environment
# (Users may set environment variables `$SymEngine_DIR` or `$CMAKE_MODULE_PATH`)
if (NOT AKG_MLIR_BUILD_SYMENGINE AND NOT DEFINED SYMENGINE_BUILD_PATH)
  find_package(SymEngine)
  if (SYMENGINE_FOUND)
    message(STATUS "SymEngine found via find_package()")
  endif ()
endif ()

# Otherwise we try to build it or use an already built (but not installed) SymEngine
if (DEFINED SYMENGINE_BUILD_PATH)
  # Legacy method: custom build directory is provided
  get_filename_component(SYMENGINE_BUILD_PATH ${SYMENGINE_BUILD_PATH} ABSOLUTE)
  set(SYMENGINE_BUILD_DIR "${SYMENGINE_BUILD_PATH}")
  find_package(SymEngine PATHS ${SYMENGINE_BUILD_DIR})
  if(NOT SYMENGINE_FOUND)
    MESSAGE(FATAL_ERROR "SymEngine is not found")
  endif()
  include(${SYMENGINE_BUILD_DIR}/SymEngineConfig.cmake)
elseif (AKG_MLIR_BUILD_SYMENGINE OR NOT SYMENGINE_FOUND)
  # Build SymEngine from this cmake instance
  add_subdirectory(symengine)
endif ()

if (AKG_MLIR_BUILD_SYMENGINE AND NOT DEFINED SYMENGINE_BUILD_PATH)
  FetchContent_MakeAvailable(SymEngine)

  get_target_property(SYMENGINE_INCLUDE_DIRS symengine INCLUDE_DIRECTORIES)
  get_target_property(SYMENGINE_LIBRARIES symengine LINK_LIBRARIES)
endif ()

include_directories(${SYMENGINE_INCLUDE_DIRS})
set(SYMENGINE_LIBRARIES symengine ${SYMENGINE_LIBRARIES})

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

get_filename_component(LLVM_BUILD_PATH ${LLVM_BUILD_PATH} ABSOLUTE)

set(LLVM_INSTALL_DIR "${LLVM_BUILD_PATH}")
set(LLVM_TOOLS_DIR "${LLVM_BUILD_PATH}/bin")


if(NOT DEFINED LLVM_EXTERNAL_LIT)
  set(LLVM_EXTERNAL_LIT "${LLVM_TOOLS_DIR}/llvm-lit.py" CACHE PATH "PATH to llvm-lit")
endif()

# ----------------------------- Dependencies
set(MLIR_DIR "${LLVM_INSTALL_DIR}/lib/cmake/mlir")
include(MLIR.cmake)
set(JSON_TARBALL_URL "${AKG_MLIR_SOURCE_DIR}/../third-party/JSON-for-Modern-CPP-v3.10.1.zip")

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
