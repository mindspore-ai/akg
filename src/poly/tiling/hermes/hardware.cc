/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "poly/tiling/hermes/hardware.h"

namespace akg {
namespace ir {
namespace poly {
size_t Hardware::mem_VC_alloc_failed_ = 0;

Hardware::Hardware(size_t num_core, size_t mem_VC_size, size_t mem_C1_size, size_t mem_C0_size, size_t mem_VC_align,
                   size_t mem_C1_align, size_t vblocknum, size_t vblocksize, size_t mem_L0C_size, std::string section)
    : num_core_{num_core},
      mem_VC_size_{mem_VC_size / (1 << mem_VC_alloc_failed_)},  // we divide VC by 2 for each VC alloc error
      mem_C1_size_{mem_C1_size},
      mem_C0_size_{mem_C0_size},
      mem_VC_align_{mem_VC_align},
      mem_C1_align_{mem_C1_align},
      vblocknum_{vblocknum},
      vblocksize_{vblocksize},
      mem_L0C_size_{mem_L0C_size},
      section_{section} {}

bool Hardware::HasVCFail(const std::string &allocation_error_buf) { return allocation_error_buf == "local.UB"; }
}  // namespace poly
}  // namespace ir
}  // namespace akg
