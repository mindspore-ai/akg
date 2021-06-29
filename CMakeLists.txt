# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

set(AKG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TVM_DIR "${AKG_SOURCE_DIR}/third_party/incubator-tvm")

if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows" AND NOT USE_KC_AIR AND NOT USE_CCE_PROFILING)
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

# Find gmp
find_path(GMP_INCLUDE_DIR NAME gmp.h HINTS /usr /usr/local PATH_SUFFIXES include include/x86_64-linux-gnu)
find_library(GMP_LIBRARY NAMES gmp HINTS /usr /usr/local PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
if(NOT GMP_INCLUDE_DIR)
  message(FATAL_ERROR "Please export CMAKE_INCLUDE_PATH to directory where gmp.h locates at.")
endif()
if(NOT GMP_LIBRARY)
  message(FATAL_ERROR "Please export CMAKE_LIBRARY_PATH to directory where libgmp.so locates at.")
endif()

include(cmake/RT.cmake)
include(cmake/utils.cmake)
include(cmake/external_libs/isl.cmake)

if(USE_CCE_RT)
  set(AKG_RPATH ${AKG_RPATH}:/usr/local/Ascend/nnae/latest/fwkacllib/lib64)
  set(AKG_RPATH ${AKG_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64)
  set(AKG_RPATH ${AKG_RPATH}:/usr/local/Ascend/fwkacllib/lib64)
  set(AKG_RPATH ${AKG_RPATH}:/usr/local/Ascend/nnae/latest/fwkacllib/lib64/plugin/opskernel)
  set(AKG_RPATH ${AKG_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel)
  set(AKG_RPATH ${AKG_RPATH}:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel)
  set(AKG_RPATH ${AKG_RPATH}:/usr/local/Ascend/add-ons)
  set(AKG_RPATH ${AKG_RPATH}:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/op_tiling)
  set(AKG_RPATH ${AKG_RPATH}:/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)
  set(AKG_RPATH ${AKG_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)
elseif(USE_CUDA)
  set(AKG_RPATH ${AKG_RPATH}:/usr/local/cuda/lib64)
endif()
link_directories(${AKG_RPATH})

# Search AKG_EXTEND by order
set(AKG_EXTEND )
if(NOT USE_CUDA)
  set(AKG_EXTEND_FILE )
  set(LIB_PATH1 ${AKG_SOURCE_DIR}/libakg_ext.a)
  set(LIB_PATH2 ${AKG_SOURCE_DIR}/build/libakg_ext.a)

  if(EXISTS ${LIB_PATH1})  # Search libakg_ext.a in akg/
    set(AKG_EXTEND_FILE ${LIB_PATH1})
  else()
    if(EXISTS ${LIB_PATH2})  # Search libakg_ext.a in akg/build/
      set(AKG_EXTEND_FILE ${LIB_PATH2})
    elseif(NOT USE_KC_AIR AND NOT USE_CCE_PROFILING)  # Download libakg_ext.a to akg/build/
      execute_process(COMMAND bash ${AKG_SOURCE_DIR}/build.sh -a
              WORKING_DIRECTORY ${AKG_SOURCE_DIR}
              OUTPUT_VARIABLE OUTPUT_URL
              RESULT_VARIABLE RESULT)
      if(RESULT EQUAL 0 AND OUTPUT_URL MATCHES "libakg_ext.a")
        # Download library
        string(STRIP ${OUTPUT_URL} LIB_URL)
        message("-- Downloading ${LIB_URL} --> ${LIB_PATH2}")
        file(DOWNLOAD ${LIB_URL} ${LIB_PATH2} STATUS DOWNLOAD_STATUS)
        message("-- Download status: ${DOWNLOAD_STATUS}")
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_CODE)
        if(DOWNLOAD_CODE EQUAL 0)
          set(AKG_EXTEND_FILE ${LIB_PATH2})
        endif()
      endif()
    endif()
  endif()

  message("-- AKG_EXTEND_FILE: ${AKG_EXTEND_FILE}")
  if(EXISTS ${AKG_EXTEND_FILE})
    file(COPY ${AKG_EXTEND_FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/akg_extend)
    execute_process(COMMAND ar -x libakg_ext.a
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/akg_extend)
    file(GLOB AKG_EXTEND ${CMAKE_CURRENT_BINARY_DIR}/akg_extend/*.o)
  endif()
endif()
if(NOT AKG_EXTEND)
  message("-- Build without AKG_EXTEND support")
else()
  message("-- Build with AKG_EXTEND support")
endif()

file(COPY ${AKG_SOURCE_DIR}/python/akg DESTINATION
    ${CMAKE_CURRENT_BINARY_DIR})

# Utility functions
include(${TVM_DIR}/cmake/util/Util.cmake)

tvm_option(USE_ASAN "Build with AddressSanitizer" OFF)
tvm_option(USE_CUDA "Build with CUDA" OFF)
tvm_option(USE_CUDNN "Build with cuDNN" OFF)
tvm_option(USE_LLVM "Build with LLVM" OFF)


tvm_option(
  USE_DEFAULT_LOG
  "Use Customize log to eliminate useless log. If you want to enable defalut log, set this option to ON"
  OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# include directories
include_directories(AFTER "${TVM_DIR}/include")
include_directories(AFTER "${TVM_DIR}")
include_directories(AFTER "${TVM_DIR}/src")
include_directories(AFTER "${TVM_DIR}/src/schedule")
include_directories(AFTER "${AKG_SOURCE_DIR}/src")
include_directories(AFTER "${AKG_SOURCE_DIR}/src/include")
include_directories(AFTER "${TVM_DIR}/3rdparty/dmlc-core/include")
include_directories(AFTER "${TVM_DIR}/3rdparty/dlpack/include")
include_directories(AFTER "${TVM_DIR}/3rdparty/compiler-rt")
include_directories(AFTER "${TVM_DIR}/3rdparty/rang/include")
include_directories(AFTER "${TVM_DIR}/3rdparty/picojson")
# cce rt
include_directories(AFTER "${AKG_SOURCE_DIR}/third_party/fwkacllib/inc")
include_directories(AFTER "${AKG_SOURCE_DIR}/third_party/fwkacllib/inc/toolchain")

# initial variables
set(TVM_LINKER_LIBS )

add_definitions(-DPICOJSON_USE_INT64=1)
add_definitions(-DDMLC_LOG_CUSTOMIZE=1)
if(USE_AKG_LOG)
  add_definitions(-DUSE_AKG_LOG=1)
endif()
if(NOT AKG_EXTEND)
  add_definitions(-DUSE_AKG_COMPILE_STUB=1)
endif()
if(USE_CCE_PROFILING)
  add_definitions(-DUSE_CCE_PROFILING=1)
endif()
if(USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()

# Generic compilation options
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" SUPPORT_CXX11)
if(NOT SUPPORT_CXX11)
  message(
    FATAL_ERROR "-- please choose a compiler which support C++ 11 standard")
endif()

check_cxx_compiler_flag("-march=native" NATIVE_BUILD)
if(NATIVE_BUILD AND ENABLE_NATIVE_BUILD)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -Wall -fPIC -fstack-protector-all")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pipe -Wall -fPIC -fstack-protector-all")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack")

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("-- Build in Debug mode")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g -rdynamic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -rdynamic")
else()
  message("-- Build in Release mode")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -D_FORTIFY_SOURCE=2 -Werror")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -D_FORTIFY_SOURCE=2 -Werror")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION
                                           VERSION_GREATER 7.0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
endif()
if(USE_ASAN)
  set(CMAKE_C_FLAGS
      "-fsanitize=address -fno-omit-frame-pointer  ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS
      "-fsanitize=address -fno-omit-frame-pointer  ${CMAKE_CXX_FLAGS}")
endif()

# add source group
file(GLOB_RECURSE GROUP_SOURCE "${TVM_DIR}/src/*.cc" "src/*.cc")
file(GLOB_RECURSE GROUP_INCLUDE "${TVM_DIR}/src/*.h"
     "${TVM_DIR}/include/*.h" "src/*.h" "include/*.h")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
file(
  GLOB
  COMPILER_SRCS
  ${TVM_DIR}/3rdparty/dmlc-core/src/io/filesys.cc
  ${TVM_DIR}/3rdparty/dmlc-core/src/io/indexed_recordio_split.cc
  ${TVM_DIR}/3rdparty/dmlc-core/src/io/input_split_base.cc
  ${TVM_DIR}/3rdparty/dmlc-core/src/io/line_split.cc
  ${TVM_DIR}/3rdparty/dmlc-core/src/io/recordio_split.cc
  ${TVM_DIR}/3rdparty/dmlc-core/src/io/local_filesys.cc
  ${TVM_DIR}/3rdparty/dmlc-core/src/io.cc
  ${TVM_DIR}/src/api/*.cc
  ${TVM_DIR}/src/arithmetic/*.cc
  ${TVM_DIR}/src/codegen/*.cc
  ${TVM_DIR}/src/lang/*.cc
  ${TVM_DIR}/src/pass/*.cc
  ${TVM_DIR}/src/op/*.cc
  ${TVM_DIR}/src/node/*.cc
  ${TVM_DIR}/src/schedule/*.cc
  ${TVM_DIR}/src/runtime/*.cc
  ${TVM_DIR}/src/runtime/vm/*.cc
  ${TVM_DIR}/src/runtime/vm/profiler/*.cc
  ${TVM_DIR}/src/codegen/stackvm/*.cc
  ${TVM_DIR}/src/runtime/cce/*.cc
  ${AKG_SOURCE_DIR}/src/poly/*.cc
  ${AKG_SOURCE_DIR}/src/poly/schedule_pass/*.cc
  ${AKG_SOURCE_DIR}/src/poly/schedule_pass_gpu/*.cc
  ${AKG_SOURCE_DIR}/src/poly/tiling/*.cc
  ${AKG_SOURCE_DIR}/src/poly/gpu_emit/*.cc
  ${AKG_SOURCE_DIR}/src/api/*.cc
  ${AKG_SOURCE_DIR}/src/pass/*.cc
  ${AKG_SOURCE_DIR}/src/schedule/*.cc
  ${AKG_SOURCE_DIR}/src/codegen/*.cc
  ${AKG_SOURCE_DIR}/src/composite/*.cc
  ${AKG_SOURCE_DIR}/src/composite/optimize/*.cc
  ${AKG_SOURCE_DIR}/src/common/*.cc
  ${AKG_SOURCE_DIR}/src/runtime/stub/*.cc)

file(GLOB RUNTIME_STUB_SRC ${AKG_SOURCE_DIR}/src/runtime/stub/*.cc)
if(USE_CCE_RT OR USE_KC_AIR)
  list(REMOVE_ITEM COMPILER_SRCS ${RUNTIME_STUB_SRC})
endif()

if (USE_CCE_RT)
  file(GLOB ASCEND_SRCS  ${AKG_SOURCE_DIR}/src/profiler/ascend/*.cc)
  list(APPEND COMPILER_SRCS ${ASCEND_SRCS})
  include_directories(AFTER "${AKG_SOURCE_DIR}/src/profiler/ascend")
endif()


file(GLOB DATATYPE_SRCS ${TVM_DIR}/src/codegen/datatype/*.cc)
list(APPEND COMPILER_SRCS ${DATATYPE_SRCS})

file(GLOB TOPI_SRCS ${TVM_DIR}/topi/src/*.cc)

file(GLOB_RECURSE RELAY_SRCS ${TVM_DIR}/src/relay/*.cc)
list(APPEND COMPILER_SRCS ${RELAY_SRCS})

file(GLOB COMPILER_VERILOG_SRCS ${TVM_DIR}/src/codegen/veriog/*.cc)
list(APPEND COMPILER_SRCS ${COMPILER_VERILOG_SRCS})

file(
  GLOB
  RUNTIME_SRCS
  ${TVM_DIR}/src/runtime/*.cc
  ${TVM_DIR}/src/runtime/vm/*.cc
  ${TVM_DIR}/src/runtime/stackvm/*.cc)

if(USE_CUDA)
  include(${TVM_DIR}/cmake/util/FindCUDA.cmake)
  include(${TVM_DIR}/cmake/modules/CUDA.cmake)
  file(GLOB PROFILER_SRCS ${AKG_SOURCE_DIR}/src/profiler/gpu/*.cc)
  list(APPEND COMPILER_SRCS ${PROFILER_SRCS})
endif()

if(USE_LLVM)
  include(${TVM_DIR}/cmake/util/FindLLVM.cmake)
  file(GLOB LLVM_SRCS ${TVM_DIR}/src/codegen/llvm/*.cc)
  list(APPEND COMPILER_SRCS ${LLVM_SRCS})
  include(${TVM_DIR}/cmake/modules/LLVM.cmake)
endif()

if(USE_RPC)
  message(STATUS "Build with RPC support...")
  file(GLOB RUNTIME_RPC_SRCS ${TVM_DIR}/src/runtime/rpc/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_RPC_SRCS})
endif(USE_RPC)

file(GLOB COMPILER_OFF_SRC ${TVM_DIR}/src/codegen/opt/build_*_off.cc)
list(APPEND COMPILER_SRCS ${COMPILER_OFF_SRC})

if(USE_CUDA)
  list(REMOVE_ITEM COMPILER_SRCS ${TVM_DIR}/src/codegen/opt/build_cuda_off.cc)
endif(USE_CUDA)

add_library(akg SHARED ${COMPILER_SRCS} ${RUNTIME_SRCS} ${TOPI_SRCS} ${AKG_EXTEND})

add_dependencies(akg akg::isl_fixed)
target_link_libraries(akg ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS} akg::isl_fixed ${GMP_LIBRARY} pthread)

if(USE_CCE_RT)
  find_library(profiler_acl msprofiler_fwkacl /usr/local/Ascend/fwkacllib/lib64)
  find_library(profiler msprofiler_fwk /usr/local/Ascend/fwkacllib/lib64)
  find_library(profiler_protobuf ascend_protobuf /usr/local/Ascend/fwkacllib/lib64) 
  find_library(ge_runner ge_runner /usr/local/Ascend/fwkacllib/lib64)
  target_link_libraries(akg ${ge_runner} ${profiler_acl} ${profiler} ${profiler_protobuf})
endif()

# Related headers
target_include_directories(akg PRIVATE "${TVM_DIR}/topi/include")

# Installation rules
if(ENABLE_AKG)
  install(TARGETS akg DESTINATION lib${LIB_SUFFIX})
else()
  install(TARGETS akg DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/akg/lib)
endif()

install(
  DIRECTORY
  ${TVM_DIR}/python/tvm
  ${TVM_DIR}/topi/python/topi
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/akg)
install(
  DIRECTORY
  ${AKG_SOURCE_DIR}/src/akg_reduce
  ${AKG_SOURCE_DIR}/src/paris_reduce
  ${AKG_SOURCE_DIR}/src/akg_mma_lib
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/akg/include)
file(GLOB REPOSITORY_FILE_LIST ${AKG_SOURCE_DIR}/python/akg/composite/*.json)
install(FILES ${REPOSITORY_FILE_LIST} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/akg/config)

