/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * 2019.12.30 - Add nullptr param init.
 * 2022.8.9 - Add cpu cache flush and cold down for profiling.
 */

#include <tvm/runtime/registry.h>
#include <memory>
#include <cstring>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include "prof_session.h"

namespace air {
namespace runtime {

inline void CPUCacheFlushImpl(const char* addr, unsigned int len) {
#if (defined(_M_X64) || defined(__x86_64__) || defined(__aarch64__))

#if defined(__aarch64__)
  size_t ctr_el0 = 0;
  asm volatile("mrs %0, ctr_el0" : "=r"(ctr_el0));
  const size_t cache_line = 4 << ((ctr_el0 >> 16) & 15);
#else
  const size_t cache_line = 64;
#endif

  if (addr == nullptr || len <= 0) {
    return;
  }

  for (uintptr_t uptr = (uintptr_t)addr & ~(cache_line - 1); uptr < (uintptr_t)addr + len;
       uptr += cache_line) {
#if defined(__aarch64__)
    asm volatile("dc civac, %0\n\t" : : "r"(reinterpret_cast<const void*>(uptr)) : "memory");
#else
    _mm_clflush(reinterpret_cast<const void*>(uptr));
#endif
  }

#if defined(__aarch64__)
  asm volatile("dmb ishst" : : : "memory");
#endif

#endif
}

inline void CPUCacheFlush(int begin_index, const TVMArgs& args) {
  for (int i = begin_index; i < args.size(); i++) {
    CPUCacheFlushImpl(static_cast<char*>((args[i].operator DLTensor*()->data)),
                      GetDataSize(*(args[i].operator DLTensor*())));
  }
}

TVM_REGISTER_GLOBAL("module._TimeEvaluator")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module m = args[0];
    std::string name = args[1];
    int device_type = args[2];
    int device_id = args[3];
    int number = args[4];
    int repeat = args[5];
    int min_repeat_ms = args[6];
    int cooldown_interval_ms = args[7];
    int repeats_to_cooldown = args[8];
    std::string f_preproc_name = args[9];
    
    std::string tkey = m->type_key();
    TVMContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(device_type);
    ctx.device_id = device_id;

    PackedFunc f_preproc;
    if (!f_preproc_name.empty()) {
      auto* pf_preproc = runtime::Registry::Get(f_preproc_name);
      CHECK(pf_preproc != nullptr)
          << "Cannot find " << f_preproc_name << " in the global function";
      f_preproc = *pf_preproc;
    }
    *rv = WrapTimeEvaluator(
        m.GetFunction(args[1], false), ctx, number, repeat, min_repeat_ms,
                            cooldown_interval_ms, repeats_to_cooldown, f_preproc);
  });

TVM_REGISTER_GLOBAL("cache_flush_cpu_non_first_arg").set_body([](TVMArgs args, TVMRetValue* rv) {
  CPUCacheFlush(1, args);
});

}  // namespace runtime
}  // namespace air
