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

#ifndef AKG_RANDOM_H
#define AKG_RANDOM_H

#include "curand_kernel.h"

namespace akg_random {

/**
 * @brief This function returns a single normally distributed float with mean 0.0 and
          standard deviation 1.0. This result can be scaled and shifted to produce normally
          distributed values with any mean and standard deviation.
 */
__inline__ __device__ float StandardNormal(unsigned long long seed, unsigned long long id)
{
   if (seed == 0) seed = clock64();
   curandState s;
   curand_init(seed, id, 0, &s);
   return curand_normal(&s);
}

} // namespace akg_random

#endif // AKG_RANDOM_H
