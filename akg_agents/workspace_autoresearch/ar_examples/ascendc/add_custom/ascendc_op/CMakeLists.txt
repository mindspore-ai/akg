cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
project(add_custom LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_HOME_PATH)
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "ASCEND_HOME_PATH is not set")
endif()

set(ACL_INCLUDE_DIR "${ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "${ASCEND_HOME_PATH}/lib64")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

add_library(add_custom_ops SHARED
  op_kernel/add_custom_kernel.asc
  op_extension/add_custom_torch.cpp
  op_extension/register.cpp
)

target_include_directories(add_custom_ops PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_INCLUDE_DIRS}
  ${ACL_INCLUDE_DIR}
  ${Python3_INCLUDE_DIRS}
)

target_link_directories(add_custom_ops PRIVATE
  ${ACL_LIB_DIR}
  "${ASCEND_HOME_PATH}/aarch64-linux/lib64"
  ${TORCH_NPU_LIBRARIES}
)

target_link_libraries(add_custom_ops PRIVATE
  torch_npu
  tiling_api
  register
  platform
  unified_dlog
  dl
  m
  graph_base
  ascendcl
  ascendc_runtime
)

target_compile_options(add_custom_ops PRIVATE
  ${TORCH_CXX_FLAGS}
  $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
