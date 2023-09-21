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
 * 2019.12.30 - Add some LOG prints.
 * 2022.8.9 - Update time_evaluator: cache flush and cold down
 */

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>
#include <memory>
#include <array>
#include <string>
#include <chrono>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <thread>
#include "prof_session.h"

namespace air {
namespace runtime {

PackedFunc WrapTimeEvaluator(PackedFunc pf,
                             TVMContext ctx,
                             int number,
                             int repeat,
                             int min_repeat_ms,
                             int cooldown_interval_ms, 
                             int repeats_to_cooldown,
                             PackedFunc f_preproc) {
  auto ftimer = [pf, ctx, number, repeat, min_repeat_ms, cooldown_interval_ms, repeats_to_cooldown,
                 f_preproc](TVMArgs args, TVMRetValue *rv) mutable {
    TVMRetValue temp;
    std::ostringstream os;
    // skip first time call, to activate lazy compilation components.
    pf.CallPacked(args, &temp);
    DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);

    for (int i = 0; i < repeat; ++i) {
      if (f_preproc != nullptr) {
        f_preproc.CallPacked(args, &temp);
      }

      std::chrono::time_point<
        std::chrono::high_resolution_clock, std::chrono::nanoseconds> tbegin, tend;
      double duration_ms = 0.0;

      do {
        if (duration_ms > 0.0) {
          number = static_cast<int>(
              std::max((min_repeat_ms / (duration_ms / number) + 1),
                       number * 1.618));   // 1.618 is chosen by random
        }

        tbegin = std::chrono::high_resolution_clock::now();
        // start timing
        for (int i = 0; i < number; ++i) {
          pf.CallPacked(args, &temp);
        }
        DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
        tend = std::chrono::high_resolution_clock::now();

        duration_ms = std::chrono::duration_cast<std::chrono::duration<double> >
            (tend - tbegin).count() * 1000;
      } while (duration_ms < min_repeat_ms);

      double speed = std::chrono::duration_cast<std::chrono::duration<double> >(
          tend - tbegin).count() / number;
      os.write(reinterpret_cast<char*>(&speed), sizeof(speed));

      if (cooldown_interval_ms > 0 && (i % repeats_to_cooldown) == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cooldown_interval_ms));
      }
    }
    std::string blob = os.str();
    TVMByteArray arr;
    arr.size = blob.length();
    arr.data = blob.data();
    // return the time.
    *rv = arr;
  };
  return PackedFunc(ftimer);
}

}  // namespace runtime
}  // namespace air
