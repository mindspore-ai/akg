/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef COMPOSITE_COMPOSITE_H_
#define COMPOSITE_COMPOSITE_H_
#include "picojson.h"
#include "composite/util.h"
namespace akg {
void ExtractBuildInfo(const picojson::value &input_json, BuildInfo &info);
Schedule GetScheduleWithBuildInfo(const BuildInfo &info);
}  // namespace akg
#endif  // COMPOSITE_COMPOSITE_H_
