/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef PASS_OVERFLOW_CHECK_H_
#define PASS_OVERFLOW_CHECK_H_

// Check if the operation will cause overflow.
#define CHECK_ADD_OVERFLOW(A, B, RES)     \
  if ((A) > 0 && (B) > 0) {               \
    if (((RES) < (A)) || ((RES) < (B))) { \
      LOG(FATAL) << "Add overflow";       \
    }                                     \
  } else if ((A) < 0 && (B) < 0) {        \
    if (((RES) > (A)) || ((RES) > (B))) { \
      LOG(FATAL) << "Add overflow";       \
    }                                     \
  }

#define CHECK_SUB_OVERFLOW(A, B, RES)     \
  if ((A) > 0 && (B) < 0) {               \
    if (((RES) < (A)) || ((RES) < (B))) { \
      LOG(FATAL) << "Sub overflow";       \
    }                                     \
  } else if ((A) < 0 && (B) > 0) {        \
    if (((RES) > (A)) || ((RES) > (B))) { \
      LOG(FATAL) << "Sub overflow";       \
    }                                     \
  }

#define CHECK_MUL_OVERFLOW(A, B, RES) \
  if ((A) != 0) {                     \
    if (((RES) / (A)) != (B)) {       \
      LOG(FATAL) << "Mul overflow";   \
    }                                 \
  }

#endif  // PASS_OVERFLOW_CHECK_H_
