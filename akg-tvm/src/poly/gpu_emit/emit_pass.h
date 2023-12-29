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

#ifndef EMIT_PASS_H_
#define EMIT_PASS_H_
#include "../isl_emitter.h"
#include "gpu_isl_emitter.h"
#include "gpu_isl_emitter_reduce.h"
#include "gpu_isl_emitter_tensor_core.h"

namespace akg {
namespace ir {
namespace poly {
Stmt EmitForTensorCore(Stmt stmt, TensorCoreInfo &info, ScopInfo &scop_info);
Stmt EmitForReduce(Stmt stmt, ScopInfo &scop_info);
Stmt EmitForTensorCoreDesignOne(Stmt stmt, TensorCoreInfo &info);
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // EMIT_PASS_H_