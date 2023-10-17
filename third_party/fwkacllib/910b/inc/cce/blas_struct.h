/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef CC_BLAS_STRUCT_API__
#define CC_BLAS_STRUCT_API__

#include <stdint.h>

typedef enum { CCBLAS_FILL_MODE_LOWER = 0, CCBLAS_FILL_MODE_UPPER = 1 } ccblasFillMode_t;

typedef enum {
  CCBLAS_OP_N = 0,
  CCBLAS_OP_T = 1,
} ccblasOperation_t;

typedef enum { CCBLAS_DIAG_NON_UNIT = 0, CCBLAS_DIAG_UNIT = 1 } ccblasDiagType_t;

#endif  // CC_BLAS_STRUCT_API__
