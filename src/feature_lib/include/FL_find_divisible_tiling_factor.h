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
#ifndef FEATURE_LIB_FL_FIND_DIVISIBLE_TILING_FACTOR_H_
#define FEATURE_LIB_FL_FIND_DIVISIBLE_TILING_FACTOR_H_

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ ([aicore])
#endif

__aicore__ int32_t FL_find_divisible_tiling_factor(int32_t mem_limit, int32_t shape);

#endif  // FEATURE_LIB_FL_FIND_DIVISIBLE_TILING_FACTOR_H_
