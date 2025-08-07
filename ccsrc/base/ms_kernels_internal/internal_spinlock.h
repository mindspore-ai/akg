/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MS_CUSTOM_OPS_INTERNAL_SPINLOCK_H_
#define MS_CUSTOM_OPS_INTERNAL_SPINLOCK_H_

#include <atomic>

namespace ms_custom_ops {
class SimpleSpinLock {
public:
  void lock() {
    while (lock_.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() { lock_.clear(std::memory_order_release); }

private:
  std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
};
} // namespace ms_custom_ops

#endif // MS_CUSTOM_OPS_INTERNAL_SPINLOCK_H_
