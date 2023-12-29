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

################################################################################
# cmake options for SymEngine
################################################################################

set(HAVE_SYMENGINE_NOEXCEPT OFF)
# set(CMAKE_BUILD_TYPE:STRING=${_BUILD_TYPE})
set(WITH_BFD OFF)
set(WITH_SYMENGINE_ASSERT OFF)
set(WITH_SYMENGINE_RCP ON)
set(WITH_SYMENGINE_THREAD_SAFE OFF)
set(WITH_ECM OFF)
set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
set(BUILD_BENCHMARKS_GOOGLE OFF)
set(BUILD_SHARED_LIBS ON)
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

################################################################################
# Fetch SymEngine
################################################################################

FetchContent_Declare(SymEngine
  SOURCE_DIR "${AKG_MLIR_SOURCE_DIR}/../third-party/symengine"
  DOWNLOAD_COMMAND git -C "${AKG_MLIR_SOURCE_DIR}" submodule update --init --progress --depth 1 -- <SOURCE_DIR>
  UPDATE_COMMAND git  -C "${AKG_MLIR_SOURCE_DIR}"submodule update --init --progress --depth 1 -- <SOURCE_DIR>
)
