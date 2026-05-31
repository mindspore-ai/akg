# Copyright 2023-2026 Huawei Technologies Co., Ltd
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
project(akg C CXX)

option(AKG_ENABLE_BINDINGS_PYTHON "Enables MLIR Python Bindings" OFF)
option(AKG_INCLUDE_TESTS "Generate build targets for the akg-mlir tests." ON)
option(AKG_BUILD_MLIR "Automatic download and build MLIR from source " OFF)

include(CheckCXXCompilerFlag)
include(CheckLanguage)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.5.0)
        message(FATAL_ERROR "GCC version must be 7.5.0 and above, but found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(AKG_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -Wno-unknown-pragmas")

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
set(AKG_MLIR_TOOLS_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})


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
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)

include(${CMAKE_SOURCE_DIR}/cmake/utils.cmake)
find_package(pybind11 REQUIRED)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/symengine.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/MLIR.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/glog.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/json.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/ascendnpu_ir.cmake)

#-------------------------------------------------------------------------------
# AKG-MLIR
#-------------------------------------------------------------------------------
include_directories(${AKG_MLIR_SRC_INCLUDE_DIR})
include_directories(${AKG_MLIR_BIN_INCLUDE_DIR})

add_subdirectory(${AKG_MLIR_SOURCE_DIR}/lib ${AKG_MLIR_BINARY_DIR}/lib)
add_subdirectory(${AKG_MLIR_SOURCE_DIR}/include ${AKG_MLIR_BINARY_DIR}/include)

# Build tools
add_subdirectory(${AKG_MLIR_SOURCE_DIR}/tools ${AKG_MLIR_BINARY_DIR}/tools)


if(AKG_INCLUDE_TESTS)
  add_definitions(-DAKG_INCLUDE_TESTS)
endif()

# Build tests
if(AKG_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()

if(AKG_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
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
  ${AKG_MLIR_SOURCE_DIR}/python/akg_mlir/
  DESTINATION ${AKG_MLIR_BINARY_DIR}/akg)
