/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifdef USE_AKG_COMPILE_STUB
#ifdef AKG_USE_MLS
#include "poly/schedule_pass/scheduling_mind_trick.h"

namespace akg {
namespace ir {
namespace poly {
////////////////////////////////////////////////////////////////////////////////
// AutoGenAscend910SoftConstraints
////////////////////////////////////////////////////////////////////////////////

std::tuple<std::string, std::string> AutoGenAscend910SoftConstraints(const ScopInfo &scop_info,
                                                                     const isl::schedule &sch) {
  LOG(ERROR) << "can't generate autoMindtrick need: (1) compile with -e ascend or (2) another version of libakg_ext.a";

  return std::make_tuple("", "");
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // AKG_USE_MLS
#endif
