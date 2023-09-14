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
 * 2022.8.16 - Update time-evaluator-related funcs: cache flush and cold down
 */

#ifndef TVM_RUNTIME_PROF_PROF_SESSION_H_
#define TVM_RUNTIME_PROF_PROF_SESSION_H_

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <mutex>
#include <string>
#include <memory>
#include <utility>

namespace air {
namespace runtime {

/*!
 * \brief Wrap a timer function to measure the time cost of a given packed function.
 * \param f The function argument.
 * \param ctx The context.
 * \param number The number of times to run this function for taking average.
          We call these runs as one `repeat` of measurement.
 * \param repeat The number of times to repeat the measurement.
          In total, the function will be invoked (1 + number x repeat) times,
          where the first one is warm up and will be discarded.
          The returned result contains `repeat` costs,
          each of which is an average of `number` costs.
 * \param min_repeat_ms The minimum duration of one `repeat` in milliseconds.
          By default, one `repeat` contains `number` runs. If this parameter is set,
          the parameters `number` will be dynamically adjusted to meet the
          minimum duration requirement of one `repeat`.
          i.e., When the run time of one `repeat` falls below this time,
          the `number` parameter will be automatically increased.
 * \return f_timer A timer function.
 */
PackedFunc WrapTimeEvaluator(PackedFunc f,
                             TVMContext ctx,
                             int number,
                             int repeat,
                             int min_repeat_ms,
                             int cooldown_interval_ms,
                             int repeats_to_cooldown,
                             PackedFunc f_preproc);

}  // namespace runtime
}  // namespace air
#endif  // TVM_RUNTIME_PROF_PROF_SESSION_H_
