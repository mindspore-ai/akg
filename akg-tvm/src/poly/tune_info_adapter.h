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
#include "auto_tune/tune_info.h"

#include "scop_info.h"
#include "tiling/tiling_analyzer.h"
#include "tiling/tiling.h"

namespace akg {
namespace ir {
namespace poly {

std::unique_ptr<TuneInfo> GenerateTuningInfo(const isl::schedule &sch, ScopInfo *scop_info, Stmt body);

}  // namespace poly
}  // namespace ir
}  // namespace akg
