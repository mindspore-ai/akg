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
#ifndef POLY_TILING_HERMES_CLASSIFY_AXIS_H_
#define POLY_TILING_HERMES_CLASSIFY_AXIS_H_

#include "poly/tiling/hermes/node.h"

namespace akg {
namespace ir {
namespace poly {
void ClassifyAxis(const Node &node);
void SetAxisTypeAsMultiCore(int dim_axis);
void SetAxisTypeAsVectorization(int dim_axis);
void SetAxisTypeAsMultiCoreAndVectorization(int dim_axis);
void ClassifyMatmulAxis();
void DefineAxisType(int dim_axis);
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_CLASSIFY_AXIS_H_
