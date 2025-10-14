/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "kernel_operator.h"

class KernelGridSample {
 public:
  __aicore__ inline KernelGridSample() {}
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR grid, GM_ADDR output, GridSampleTilingData *tiling) {
    this->input = input;
    this->grid = grid;
    this->output = output;
    this->h_in = tiling->h_in;
    this->w_in = tiling->w_in;
    this->h_out = tiling->h_out;
    this->w_out = tiling->w_out;
    this->n_in = tiling->n_in;
    this->c_in = tiling->c_in;
  }
  __aicore__ inline void Process() {
    grid_sample_0_kernel(this->input, this->grid, this->output, this->h_in, this->w_in, this->h_out, this->w_out, this->n_in, this->c_in);
  }

private:
  __aicore__ inline void CopyIn(int32_t index) {
  }
  __aicore__ inline void Compute() {
  }
  __aicore__ inline void CopyOut(int32_t index) {
  }

private:
    GM_ADDR input;
    GM_ADDR grid;
    GM_ADDR output;
    float h_in;
    float w_in;
    int32_t h_out;
    int32_t w_out;
    int32_t n_in;
    int32_t c_in;
};

extern "C" __global__ __aicore__ void grid_sample(GM_ADDR input, GM_ADDR grid, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelGridSample op;
    op.Init(input, grid, out, &tilingData);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void grid_sample_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *input, uint8_t *grid,
                       uint8_t *output, uint8_t *workspace, uint8_t *tiling) {
  grid_sample<<<blockDim, l2ctrl, stream>>>(input, grid, output, workspace, tiling);
}
#endif
