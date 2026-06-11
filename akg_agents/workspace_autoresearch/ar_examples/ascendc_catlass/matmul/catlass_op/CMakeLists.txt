# Copyright (c) 2025 Huawei Technologies Co., Ltd. This program is free
# software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the
# "License"). Please refer to the License for details. You may not use this file
# except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS IS"
# BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
# full text of the License.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR "ASCEND_HOME_PATH not set")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

if(NOT DEFINED ENV{CATLASS_ROOT})
  message(FATAL_ERROR "CATLASS_ROOT not set")
endif()
set(CATLASS_ROOT $ENV{CATLASS_ROOT})

find_package(ASC REQUIRED)
project(catlass LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(NPU_ARCH "2201" CACHE STRING "NPU architecture")
add_compile_definitions(CATLASS_ARCH=${NPU_ARCH})

add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-${NPU_ARCH}>)
add_compile_options(
  $<$<COMPILE_LANGUAGE:ASC>:-Wno-macro-redefined>
  $<$<COMPILE_LANGUAGE:ASC>:-Wno-ignored-attributes>
  $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-value>)
add_link_options($<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-command-line-argument>)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}/torch/share/cmake")
find_package(Torch REQUIRED)

set(CATLASS_INCLUDE_DIR ${CATLASS_ROOT}/include)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CATLASS_INCLUDE_DIR}
  ${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS}
  ${Python3_SITELIB}/torch_npu/include ${ASCEND_HOME_PATH}/include)

link_directories(${ASCEND_HOME_PATH}/lib64 ${Python3_SITELIB}/torch_npu/lib)

set_source_files_properties(kernel/catlass_kernel.asc PROPERTIES LANGUAGE ASC)

add_library(catlass SHARED src/catlass_torch.cpp kernel/catlass_kernel.asc)

target_link_libraries(
  catlass
  ${TORCH_LIBRARIES}
  torch_npu
  ascendc_runtime
  runtime
  ascendcl
  tiling_api
  platform)
