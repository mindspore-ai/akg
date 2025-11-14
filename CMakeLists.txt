# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
        message(FATAL_ERROR "GCC version must be 7.3.0 and above, but found ${CMAKE_CXX_COMPILER_VERSION}")
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 11.1.0)
        message(WARNING "GCC version ${CMAKE_CXX_COMPILER_VERSION} is greater than 11.1.0, may cause unknown problems.")
    endif()
endif()

set(AKG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TVM_DIR "${AKG_SOURCE_DIR}/third_party/incubator-tvm")

get_property(COMPILE_DEFS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY COMPILE_DEFINITIONS)
list(REMOVE_ITEM COMPILE_DEFS "_GLIBCXX_USE_CXX11_ABI=0")
list(REMOVE_ITEM COMPILE_DEFS "_GLIBCXX_USE_CXX11_ABI=1")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY COMPILE_DEFINITIONS ${COMPILE_DEFS})

include(cmake/RT.cmake)
include(cmake/utils.cmake)
include(cmake/external_libs/isl.cmake)

# Search AKG_EXTEND by order
set(AKG_EXTEND )
if(ENABLE_D)
  set(AKG_EXTEND_FILE )
  set(LIB_PATH1 ${AKG_SOURCE_DIR}/libakg_ext.a)
  set(LIB_PATH2 ${CMAKE_CURRENT_BINARY_DIR}/libakg_ext.a)

  if(EXISTS ${LIB_PATH1})  # Search libakg_ext.a in akg_source_dir/
    set(AKG_EXTEND_FILE ${LIB_PATH1})
    message("-- Find ${LIB_PATH1}")
  else()
    if(EXISTS ${LIB_PATH2})  # If .a not found, search .a in akg_build_dir/
      set(AKG_EXTEND_FILE ${LIB_PATH2})
      message("-- Find ${LIB_PATH2}")
    elseif(NOT USE_KC_AIR)  # If .a not found, search .o in akg_source_dir/prebuild
      execute_process(COMMAND bash ${AKG_SOURCE_DIR}/build.sh -o
              WORKING_DIRECTORY ${AKG_SOURCE_DIR}
              OUTPUT_VARIABLE EXEC_OUTPUT
              RESULT_VARIABLE RESULT)
      message("${EXEC_OUTPUT}")
      if(RESULT EQUAL 0)
        string(STRIP ${EXEC_OUTPUT} AKG_EXTEND_DIR)
        file(GLOB AKG_EXTEND ${AKG_EXTEND_DIR}/*.o)
        message("-- Find .o in ${AKG_EXTEND_DIR}")
      endif()
    endif()
  endif()

  if(EXISTS ${AKG_EXTEND_FILE})
    file(COPY ${AKG_EXTEND_FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/akg_extend)
    execute_process(COMMAND ar -x libakg_ext.a
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/akg_extend)
    file(GLOB AKG_EXTEND ${CMAKE_CURRENT_BINARY_DIR}/akg_extend/*.o)
  endif()

  if(NOT AKG_EXTEND)
    message(WARNING "The .o files under ${AKG_SOURCE_DIR}/prebuild are not downloaded, AKG Ascend backend will be not support!")
  else()
    message(STATUS "Build AKG with Ascend backend support")
  endif()
endif()

if (BUILD_AUTO_TUNE)
  set(PY_AUTOTUNE ${AKG_SOURCE_DIR}/python/akg/auto_tune)
  if (EXISTS ${PY_AUTOTUNE})
    file(COPY ${PY_AUTOTUNE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  else()
    message(FATAL_ERROR "Files ${PY_AUTOTUNE} does not exist. Please build auto tune in tuning-service branch or remove -DBUILD_AUTO_TUNE option.")
  endif()
else()
  file(COPY ${AKG_SOURCE_DIR}/python/akg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Utility functions
include(${TVM_DIR}/cmake/util/Util.cmake)

tvm_option(USE_ASAN "Build with AddressSanitizer" OFF)
tvm_option(USE_CUDA "Build with CUDA" OFF)
tvm_option(USE_CUDNN "Build with cuDNN" OFF)
tvm_option(USE_LLVM "Build with LLVM" OFF)
tvm_option(USE_OPENMP "Build with OpenMP" ON)

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
include_directories(AFTER "${AKG_SOURCE_DIR}/third_party/fwkacllib/910/inc")
include_directories(AFTER "${AKG_SOURCE_DIR}/third_party/fwkacllib/910/inc/toolchain")


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
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
if(NOT SUPPORT_CXX17)
  message(
    FATAL_ERROR "-- please choose a compiler which support C++ 17 standard")
endif()

check_cxx_compiler_flag("-march=native" NATIVE_BUILD)
if(NATIVE_BUILD AND ENABLE_NATIVE_BUILD)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -Wall -fPIC -fstack-protector-all -Wno-uninitialized -Wno-parentheses")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pipe -Wall -fPIC -fstack-protector-all -Wno-uninitialized -Wno-parentheses")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack")

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("-- Build in Debug mode")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g -rdynamic -ftrapv")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -rdynamic -ftrapv")
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

if (BUILD_AUTO_TUNE)
  # Build auto tune lib only
  file(GLOB COMPILER_SRCS ${AKG_SOURCE_DIR}/src/auto_tune/*.cc)
  add_library(tune SHARED ${COMPILER_SRCS})
  install(TARGETS tune DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/auto_tune)
  return()
endif()
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
  ${AKG_SOURCE_DIR}/src/light_cp/*.cc
  ${AKG_SOURCE_DIR}/src/poly/*.cc
  ${AKG_SOURCE_DIR}/src/poly/schedule_pass/*.cc
  ${AKG_SOURCE_DIR}/src/poly/schedule_pass_gpu/*.cc
  ${AKG_SOURCE_DIR}/src/poly/schedule_analysis/*.cc
  ${AKG_SOURCE_DIR}/src/poly/schedule_pass_cpu/*.cc
  ${AKG_SOURCE_DIR}/src/poly/tiling/*.cc
  ${AKG_SOURCE_DIR}/src/poly/tiling/hermes/*.cc
  ${AKG_SOURCE_DIR}/src/poly/gpu_emit/*.cc
  ${AKG_SOURCE_DIR}/src/api/*.cc
  ${AKG_SOURCE_DIR}/src/auto_tune/*.cc
  ${AKG_SOURCE_DIR}/src/pass/*.cc
  ${AKG_SOURCE_DIR}/src/schedule/*.cc
  ${AKG_SOURCE_DIR}/src/codegen/*.cc
  ${AKG_SOURCE_DIR}/src/composite/*.cc
  ${AKG_SOURCE_DIR}/src/composite/lower_tree/*.cc
  ${AKG_SOURCE_DIR}/src/composite/utils/*.cc
  ${AKG_SOURCE_DIR}/src/composite/optimize/*.cc
  ${AKG_SOURCE_DIR}/src/common/*.cc)

# Check if auto tune lib exists and remove stub if exists
if (USE_AUTO_TUNE)
  find_library(AUTO_TUNE_LIB NAMES tune HINTS ${CMAKE_CURRENT_BINARY_DIR})
  if (EXISTS ${AUTO_TUNE_LIB})
    message("-- Find ${AUTO_TUNE_LIB}")
    file(GLOB AUTO_TUNE_STUB_SRC ${AKG_SOURCE_DIR}/src/auto_tune/*.cc)
    list(REMOVE_ITEM COMPILER_SRCS ${AUTO_TUNE_STUB_SRC})
  else()
    find_python_package(PY_LIB)
    find_library(AUTO_TUNE_LIB NAMES tune HINTS ${PY_LIB}/auto_tune)
    if (EXISTS ${AUTO_TUNE_LIB})
      message("-- Find ${AUTO_TUNE_LIB} and copy to ${CMAKE_CURRENT_BINARY_DIR}")
      file(COPY ${AUTO_TUNE_LIB} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
      file(GLOB AUTO_TUNE_STUB_SRC ${AKG_SOURCE_DIR}/src/auto_tune/*.cc)
      list(REMOVE_ITEM COMPILER_SRCS ${AUTO_TUNE_STUB_SRC})
    else()
      message("Cannot use auto tune: lib not found.")
    endif()
  endif()
endif()

if(USE_CCE_PROFILING)
  file (
    GLOB
    CCE_PROFILING_SRCS
    ${AKG_SOURCE_DIR}/src/profiler/ascend/*.cc)
  list(APPEND COMPILER_SRCS ${CCE_PROFILING_SRCS})
  include_directories(AFTER "${AKG_SOURCE_DIR}/src/profiler/ascend")
endif()

if (ENABLE_D)
  file(
    GLOB
    ASCEND_RUNTIME_SRCS
    ${AKG_SOURCE_DIR}/src/runtime/ascend/*.cc)
  list(APPEND COMPILER_SRCS ${ASCEND_RUNTIME_SRCS})
  include_directories(AFTER "${AKG_SOURCE_DIR}/src/runtime/ascend")
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

if(USE_CUDA OR ENABLE_D)
  include(${TVM_DIR}/cmake/util/FindLLVM.cmake)
  if(USE_LLVM)
    find_llvm_macro(${USE_LLVM})
    if(NOT LLVM_FOUND)
      message(WARNING "Not found LLVM, AKG CPU backend will be not support!")
      set(USE_LLVM OFF)
    endif()
  endif()
endif()

if(USE_LLVM)
  include(${TVM_DIR}/cmake/util/FindLLVM.cmake)
  include(${TVM_DIR}/cmake/modules/LLVM.cmake)
  file(GLOB RUNTIME_CPU_SRCS ${AKG_SOURCE_DIR}/src/runtime/*.cc)
  list(APPEND COMPILER_SRCS ${RUNTIME_CPU_SRCS})
  set_source_files_properties(${RUNTIME_CPU_SRCS}
                              PROPERTIES COMPILE_FLAGS "${CPU_SRCS_FLAGS}")
endif()

if(USE_OPENMP)
  include(${TVM_DIR}/cmake/modules/OpenMP.cmake)
endif()

message(STATUS "Build with Profiling support...")
file(GLOB RUNTIME_PROF_SRCS ${TVM_DIR}/src/runtime/profiling/*.cc)
list(APPEND RUNTIME_SRCS ${RUNTIME_PROF_SRCS})

file(GLOB COMPILER_OFF_SRC ${TVM_DIR}/src/codegen/opt/build_*_off.cc)
list(APPEND COMPILER_SRCS ${COMPILER_OFF_SRC})

if(USE_CUDA)
  list(REMOVE_ITEM COMPILER_SRCS ${TVM_DIR}/src/codegen/opt/build_cuda_off.cc)
endif(USE_CUDA)

add_library(akg SHARED ${COMPILER_SRCS} ${RUNTIME_SRCS} ${TOPI_SRCS} ${AKG_EXTEND})

add_dependencies(akg akg::isl_fixed)
include(cmake/polytops.cmake)

string(FIND "${TVM_LINKER_LIBS}" "-ltinfo" TINFO)
if (NOT ${TINFO} MATCHES "-1")
  string(REPLACE "-ltinfo" "" TVM_LINKER_LIBS "${TVM_LINKER_LIBS}")
  include(cmake/external_libs/ncurses.cmake)
  target_link_libraries(akg akg::isl_fixed ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS} akg::ncurses pthread -s)
else()
  target_link_libraries(akg akg::isl_fixed ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS} pthread -s)
endif()

if (EXISTS ${AUTO_TUNE_LIB})
  target_link_libraries(akg ${AUTO_TUNE_LIB})
endif()

if(USE_CCE_PROFILING)
  set(ASCEND_PATH /usr/local/Ascend)
  set(ASCEND_CANN_RUNTIME_PATH ${ASCEND_PATH}/latest/lib64)
  find_library(PROFILING msprofiler ${ASCEND_RUNTIME_PATH} ${ASCEND_CANN_RUNTIME_PATH})
  find_library(GERT gert ${ASCEND_RUNTIME_PATH} ${ASCEND_CANN_RUNTIME_PATH})
  message(STATUS "PROFILING: ${PROFILING}")
  target_link_libraries(akg -Wl,--no-as-needed ${PROFILING} ${GERT})
endif()

# Related headers
target_include_directories(akg PRIVATE "${TVM_DIR}/topi/include")

# Installation rules
if(ENABLE_AKG)
  set(AKG_PATH "lib")
  install(TARGETS akg DESTINATION ${AKG_PATH})
  if (EXISTS ${AUTO_TUNE_LIB})
    install(FILES ${AUTO_TUNE_LIB} DESTINATION ${AKG_PATH}))
  endif()
  install(
    DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/akg
    DESTINATION ${AKG_PATH}
  )
else()
  set(AKG_PATH ${CMAKE_CURRENT_BINARY_DIR})
  install(TARGETS akg DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/akg/lib)
endif()

install(
  DIRECTORY
  ${TVM_DIR}/python/tvm
  ${TVM_DIR}/topi/python/topi
  DESTINATION ${AKG_PATH}/akg)
install(
  DIRECTORY
  ${AKG_SOURCE_DIR}/src/akg_reduce
  ${AKG_SOURCE_DIR}/src/paris_reduce
  ${AKG_SOURCE_DIR}/src/akg_mma_lib
  ${AKG_SOURCE_DIR}/src/akg_random
  DESTINATION ${AKG_PATH}/akg/include)
file(GLOB REPOSITORY_FILE_LIST ${AKG_SOURCE_DIR}/python/akg/composite/*.json)
install(FILES ${REPOSITORY_FILE_LIST} DESTINATION ${AKG_PATH}/akg/config)
