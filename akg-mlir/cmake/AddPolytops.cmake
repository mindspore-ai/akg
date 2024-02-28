################################################################################
#
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
#
################################################################################

option(AKG_MLIR_USE_POLYTOPS "Use PolyTOPS polyhedral scheduler" ON)
set(AKG_MLIR_POLYTOPS_SOURCE "lib" CACHE STRING "which PolyTOPS to use: system|lib|git|build (default lib)")

################################################################################
# Download/Build/Load
################################################################################

if (AKG_MLIR_USE_POLYTOPS)
  message(STATUS "${PROJECT_NAME}: PolyTOPS AKG_MLIR_POLYTOPS_SOURCE=${AKG_MLIR_POLYTOPS_SOURCE}")
  if (AKG_MLIR_POLYTOPS_SOURCE STREQUAL "system")
    find_package(polytops)
    if (polytops_FOUND)
      # We found polytops via find_package()...
      message(STATUS "${PROJECT_NAME}: PolyTOPS found on system (most likely via `find_package()`)")
    endif ()
    add_compile_definitions(AKG_MLIR_USE_POLYTOPS)
  elseif (AKG_MLIR_POLYTOPS_SOURCE STREQUAL "lib")
    if (NOT DEFINED AKG_MLIR_POLYTOPS_LIB_FOLDER)
      set(AKG_MLIR_POLYTOPS_LIB_FOLDER "${AKG_MLIR_SOURCE_DIR}/../third-party/polytops-lib")
    endif ()
    if (NOT DEFINED AKG_MLIR_POLYTOPS_LIB_FILE)
      set(AKG_MLIR_POLYTOPS_LIB_FILE "libpolytops-schedule-opt.a")
    endif ()

    add_library(libpolytops-schedule-opt STATIC IMPORTED GLOBAL)
    set_target_properties(libpolytops-schedule-opt PROPERTIES
       IMPORTED_LOCATION
         ${AKG_MLIR_POLYTOPS_LIB_FOLDER}/${CMAKE_SYSTEM_PROCESSOR}/${AKG_MLIR_POLYTOPS_LIB_FILE}
       INTERFACE_INCLUDE_DIRECTORIES
         ${AKG_MLIR_POLYTOPS_LIB_FOLDER}/include
    )
    message(STATUS "${PROJECT_NAME}: PolyTOPS found library (${AKG_MLIR_POLYTOPS_LIB_FOLDER}/${CMAKE_SYSTEM_PROCESSOR}/${AKG_MLIR_POLYTOPS_LIB_FILE})")

    set(polytops_FOUND ON)
  elseif (AKG_MLIR_POLYTOPS_SOURCE STREQUAL "git")
    if (NOT DEFINED AKG_MLIR_POLYTOPS_GIT_TAG)
      message(FATAL_ERROR "${PROJECT_NAME}: required value for `AKG_MLIR_POLYTOPS_GIT_TAG` when `AKG_MLIR_POLYTOPS_SOURCE` set to `git`")
    endif ()
    if (NOT DEFINED AKG_MLIR_POLYTOPS_GIT_REPOSITORY)
      message(FATAL_ERROR "${PROJECT_NAME}: required value for `AKG_MLIR_POLYTOPS_GIT_REPOSITORY` when `AKG_MLIR_POLYTOPS_SOURCE` set to `git`")
    endif ()

    set(POLYTOPS_USE_OSL OFF)
    set(POLYTOPS_USE_ISL ON)
    set(POLYTOPS_USE_FPL ON)
    set(POLYTOPS_USE_PICOJSON OFF)
    set(POLYTOPS_ENABLE_SANITIZERS OFF)
    set(POLYTOPS_BUILD_MAIN_TOOL OFF)
    set(POLYTOPS_ENABLE_MLIR_DIALECT ON)
    set(POLYTOPS_BUILD_MLIR_TOOL OFF)
    set(POLYTOPS_FPL_FOUND ON)
    set(POLYTOPS_MLIR_FOUND ON)
    set(POLYTOPS_LLVM_FOUND ON)

    FetchContent_Declare(polytops
      GIT_REPOSITORY "${AKG_MLIR_POLYTOPS_GIT_REPOSITORY}"
      GIT_TAG "${AKG_MLIR_POLYTOPS_GIT_TAG}"
      GIT_SHALLOW ON
      GIT_PROGRESS ON
    )
    FetchContent_MakeAvailable(polytops)
    message(STATUS "${PROJECT_NAME}: PolyTOPS will be built from tag ${AKG_MLIR_POLYTOPS_GIT_TAG} of ${AKG_MLIR_POLYTOPS_GIT_REPOSITORY}")

    set(polytops_FOUND ON)
    add_compile_definitions(AKG_MLIR_USE_POLYTOPS)
  elseif (AKG_MLIR_POLYTOPS_SOURCE STREQUAL "build")
    message(FATAL_ERROR "${PROJECT_NAME}: not supported value `${AKG_MLIR_POLYTOPS_SOURCE}` for AKG_MLIR_POLYTOPS_SOURCE yet!")
  else ()
    message(FATAL_ERROR "${PROJECT_NAME}: unsupported value `${AKG_MLIR_POLYTOPS_SOURCE}` for AKG_MLIR_POLYTOPS_SOURCE!")
  endif ()
endif()
