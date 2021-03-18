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

#ifndef DSA_UTILS_H_
#define DSA_UTILS_H_

#include <vector>
#include <string>

namespace akg {
namespace ir {
namespace poly {
extern const char *const LOCAL_BUF;
extern const char *const LOCAL_C1;
extern const char *const _FRACTAL_C1;
extern const char *const FRACTAL_C1;
extern const char *const LOCAL_C0C;
extern const char *const LOCAL_C1_LOCAL_C0A;
extern const char *const LOCAL_C1_LOCAL_C0B;
extern const char *const LOCAL_BUF_LOCAL_C0C;
extern const char *const FRACTAL_C1_LOCAL_C0A;
extern const char *const FRACTAL_C1_LOCAL_C0B;

extern const char *const C1_LOCAL_C0A;
extern const char *const C1_LOCAL_C0B;
extern const char *const LOCAL_C0B;

extern const char *const BUF;
extern const char *const C1;
extern const char *const C0;
extern const char *const C0A;
extern const char *const C0B;
extern const char *const C0C;
extern const char *const REG;

extern const char *const DOT_LOCAL_BUF;
extern const char *const DOT_LOCAL_C1;
extern const char *const DOT_LOCAL_C1_TMP;
extern const char *const DOT_LOCAL_C0A;
extern const char *const DOT_LOCAL_C0B;
extern const char *const DOT_LOCAL_C0C;

extern const char *const LOAD_IM2COL;
extern const char *const REALIZE_C1;
extern const char *const REALIZE_C0;
extern const char *const REALIZE_BUF;
extern const char *const REALIZE_BUFC0;
extern const char *const REALIZE_BUFC1;
extern const char *const REALIZE_C1BUFC1;
extern const char *const PRAGMA_BYPATH_FILTER_C0;
extern const char *const PRAGMA_BYPATH_FILTER_C1;

extern const int FLOW_S;
extern const int FLOW_V;
extern const int FLOW_M;
extern const int FLOW_DMA1;
extern const int FLOW_DMA2;
extern const int FLOW_DMA3;

extern const char *const PRAGMA_MMU_C0WRITE;
extern const char *const PRAGMA_MMU_C1WRITE;
extern const char *const K_C1;
extern const char *const PRAGMA_GEMM_C0;

enum MemType { DDR = 1, C1_, BUF_, C0A_, C0B_, C0C_, BUF_C0_, BUF_C1_, SHARED_, LOCAL_ };
using DataFlowAttrs = std::vector<std::pair<MemType, std::string>>;

extern const DataFlowAttrs Mmu_Conv_A;
extern const DataFlowAttrs Mmu_Conv_B;
extern const DataFlowAttrs Mmu_Conv_C;
extern const DataFlowAttrs Mmu_Spec_Gemm_A;
extern const DataFlowAttrs Mmu_Spec_Gemm_A_;
extern const DataFlowAttrs Mmu_Gemm_A;
extern const DataFlowAttrs Mmu_Spec_Gemm_B;
extern const DataFlowAttrs Mmu_Spec_Gemm_B_;
extern const DataFlowAttrs Mmu_Gemm_B;
extern const DataFlowAttrs Mmu_Spec_Gemm_C;
extern const DataFlowAttrs Mmu_Gemm_C;
extern const DataFlowAttrs Inst_BUF;
extern const DataFlowAttrs Im2Col_C1;

int GetCoreValue(const std::string &name);
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // DSA_UTILS_H_
