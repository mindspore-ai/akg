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

#ifndef RUNTIME_CSIM_AICORE_FAST_SIM_H_
#define RUNTIME_CSIM_AICORE_FAST_SIM_H_

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>

#define __CCE_KT_TEST__

#ifndef __global__
#define __global__
#endif

#ifndef __aicore__
#define __aicore__
#endif

#define __ubuf__
#define __gm__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__

#include "half_float.h"

using float_t = float;
using half_t = half;
using float32_t = float;
using float16_t = half;

using int64 = int64_t;
using uint64 = uint64_t;

#ifndef ENABLE_CDIFF

using float16 = half;
using float32 = float;

using int32 = int32_t;
using int16 = int16_t;
using int8 = int8_t;
using int1 = bool;
using uint32 = uint32_t;
using uint16 = uint16_t;
using uint8 = uint8_t;
using uint1 = bool;
using iterator_t = size_t;
#define iterator_t(var) size_t var

#else  // ENABLE_CDIFF

#include "compute_tracker.h"

using float16 = CallTrackingInfo16;
using float32 = CallTrackingInfo32;

#define float float32
#define half float16

using int32 = CallTrackingInfoI32;
using int16 = CallTrackingInfoI16;
using int8 = CallTrackingInfoI8;
using int1 = CallTrackingInfoI8;
using uint32 = CallTrackingInfoU32;
using uint16 = CallTrackingInfoU16;
using uint8 = CallTrackingInfoU8;
using uint1 = CallTrackingInfoU8;
using iterator_t = TrackedIterator;
#define iterator_t(var, init) TrackedIterator var(#var, init)

#endif  // ENABLE_CDIFF

#include "aicore_debug_funcs.h"
#include "halide_intrinsics.h"

#endif  // RUNTIME_CSIM_AICORE_FAST_SIM_H_
