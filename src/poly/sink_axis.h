/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef POLY_SINK_AXIS_H_
#define POLY_SINK_AXIS_H_

#pragma once
#include "poly/transform.h"

#define MAX_STRIDE 65535

namespace akg {
namespace ir {
namespace poly {

bool FindC0Schedule(const isl::pw_aff_list &paList);
void ExchangeCoincident(std::vector<int> &coincident, const isl::schedule_node &node,
                        const std::unordered_map<int, bool> lastIdxSchedule, const int &n);

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_SINK_AXIS_H_
