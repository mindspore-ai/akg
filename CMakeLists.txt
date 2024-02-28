# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 11.1.0)
        message(WARNING "GCC version ${CMAKE_CXX_COMPILER_VERSION} is greater than 11.1.0, may cause unknown problems.")
    endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(AKG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(AKG_TVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/akg-tvm")
set(TVM_DIR "${AKG_TVM_SOURCE_DIR}/third_party/incubator-tvm")
add_subdirectory(akg-tvm)

if(ENABLE_AKG_MLIR)
  set(AKG_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/akg-mlir/compiler")
  add_subdirectory(akg-mlir)
endif()
