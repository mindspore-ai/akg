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
#ifndef POLY_TILING_HERMES_HARDWARE_H_
#define POLY_TILING_HERMES_HARDWARE_H_

#include <string>

namespace akg {
namespace ir {
namespace poly {
class Hardware {
 public:
  Hardware(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

  static bool HasVCFail(const std::string &);
  static void AddVCFailCounter() { Hardware::mem_VC_alloc_failed_++; }
  static void ResetVCFailCounter() { Hardware::mem_VC_alloc_failed_ = 0; }

  size_t num_core_;
  size_t mem_VC_size_;
  size_t mem_C1_size_;
  size_t mem_C0_size_;
  size_t mem_VC_align_;
  size_t mem_C1_align_;
  size_t vblocknum_;
  size_t vblocksize_;

 private:
  static size_t mem_VC_alloc_failed_;
};

const size_t kNumCore = 32;
const size_t kMemVCSize = 262144;
const size_t kMemC1Size = 1048576;
const size_t kMemC0Size = 65536;
const size_t kMemVCAlign = 32;
const size_t kMemC1Align = 512;
const size_t kVBlockNum = 8;
const size_t kVBlockSize = 32;
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_HARDWARE_H_
