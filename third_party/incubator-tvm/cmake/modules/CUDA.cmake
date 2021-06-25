# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#
# 2020.7.14 - Modify current directory of tvm.
# 2020.11.18 - Include CUPTI path.
# 2021.6.22 - Give hint when cuda library is not found.
#

# CUDA Module
find_cuda(${USE_CUDA})

if(CUDA_FOUND)
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
  include_directories(${CUDA_INCLUDE_DIRS})
  string(REGEX REPLACE "(.+)\\include.*" "\\1" CUPTI_ROOT_DIR ${CUDA_INCLUDE_DIRS})
  include_directories("${CUPTI_ROOT_DIR}/extras/CUPTI/include")
endif(CUDA_FOUND)

if(USE_CUDA)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=${USE_CUDA}. Please set environment variable CUDA_HOME to cuda \
            installation position. (e.g. export CUDA_HOME=/home/user_name/cuda)")
  endif()
  message(STATUS "Build with CUDA support")
  file(GLOB RUNTIME_CUDA_SRCS ${TVM_DIR}/src/runtime/cuda/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_SRCS})
  list(APPEND COMPILER_SRCS ${TVM_DIR}/src/codegen/opt/build_cuda_on.cc)

  if(NOT CUDA_NVRTC_LIBRARY)
    message(FATAL_ERROR "Failed to find nvrtc library file, please set environment variable CUDA_HOME to cuda \
            installation position. (e.g. export CUDA_HOME=/home/user_name/cuda)")
  endif()
  if(NOT CUDA_CUDART_LIBRARY)
    message(FATAL_ERROR "Failed to find cudart library file, please set environment variable CUDA_HOME to cuda \
            installation position. (e.g. export CUDA_HOME=/home/user_name/cuda)")
  endif()
  if(NOT CUDA_CUDA_LIBRARY)
    message(FATAL_ERROR "Failed to find cuda library file, please set environment variable CUDA_HOME to cuda \
            installation position. (e.g. export CUDA_HOME=/home/user_name/cuda)")
  endif()

  list(APPEND TVM_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})

  if(USE_CUDNN)
    message(STATUS "Build with cuDNN support")
    file(GLOB CONTRIB_CUDNN_SRCS ${TVM_DIR}/src/runtime/contrib/cudnn/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUDNN_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDNN_LIBRARY})
  endif(USE_CUDNN)

  if(USE_CUBLAS)
    message(STATUS "Build with cuBLAS support")
    file(GLOB CONTRIB_CUBLAS_SRCS ${TVM_DIR}/src/runtime/contrib/cublas/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUBLAS_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUBLAS_LIBRARY})
  endif(USE_CUBLAS)

else(USE_CUDA)
	list(APPEND COMPILER_SRCS ${TVM_DIR}/src/codegen/opt/build_cuda_off.cc)
endif(USE_CUDA)
