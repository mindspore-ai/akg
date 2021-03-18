/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef POLY_TILING_ALGORITHM_H_
#define POLY_TILING_ALGORITHM_H_

namespace tiling_algorithm {
namespace intrinsic {
/*!
 * \brief See pesudo code
 * Args: tile -> T1 ; memory_limit -> MEM ; shape_range -> I1;
 * Return: factor -> F (Same type as T1)
 * Temp var: div -> div
 * Algorithm:
 * if (I1 < MEM) {
 *   let T1 = I1
 * } else {
 *   let T1 = 1
 *   for (div, max(2, floordiv(I1, MEM)), I1/2 - max(2, floordiv(I1, MEM))) {
 *     if (floormod(I1, div) == 0) {
 *       T1 = I1/div
 *       break
 *     }
 *   }
 * }
 */
constexpr const char *FL_find_divisible_tiling_factor = "FL_find_divisible_tiling_factor";
constexpr const char *FL_get_gcd = "FL_get_gcd";
}  // namespace intrinsic
}  // namespace tiling_algorithm

#endif  // POLY_TILING_ALGORITHM_H_
