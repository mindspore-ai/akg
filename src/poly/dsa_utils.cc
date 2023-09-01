/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifdef USE_AKG_COMPILE_STUB
#include "poly/dsa_utils.h"

namespace akg {
namespace ir {
namespace poly {
const char *const LOCAL_BUF = "LOCAL_BUF";
const char *const LOCAL_C1 = "LOCAL_C1";
const char *const _FRACTAL_C1 = "_FRACTAL_C1";
const char *const FRACTAL_C1 = "FRACTAL_C1";
const char *const LOCAL_C0C = "LOCAL_C0C";
const char *const LOCAL_C1_LOCAL_C0A = "LOCAL_C1_LOCAL_C0A";
const char *const LOCAL_C1_LOCAL_C0B = "LOCAL_C1_LOCAL_C0B";
const char *const LOCAL_BUF_LOCAL_C0C = "LOCAL_BUF_LOCAL_C0C";
const char *const FRACTAL_C1_LOCAL_C0A = "FRACTAL_C1_LOCAL_C0A";
const char *const FRACTAL_C1_LOCAL_C0B = "FRACTAL_C1_LOCAL_C0B";

const char *const C1_LOCAL_C0A = "C1_LOCAL_C0A";
const char *const C1_LOCAL_C0B = "C1_LOCAL_C0B";
const char *const LOCAL_C0B = "LOCAL_C0B";

const char *const BUF = "BUF";
const char *const C1 = "C1";
const char *const C0 = "C0";
const char *const C0A = "C0A";
const char *const C0B = "C0B";
const char *const C0C = "C0C";
const char *const REG = "REG";

const char *const DOT_LOCAL_BUF = "DOT_LOCAL_BUF";
const char *const DOT_LOCAL_C1 = "DOT_LOCAL_C1";
const char *const DOT_LOCAL_C1_TMP = "DOT_LOCAL_C1_TMP";
const char *const DOT_LOCAL_C0A = "DOT_LOCAL_C0A";
const char *const DOT_LOCAL_C0B = "DOT_LOCAL_C0B";
const char *const DOT_LOCAL_C0C = "DOT_LOCAL_C0C";

const char *const LOAD_IM2COL = "LOAD_IM2COL";
const char *const REALIZE_C1 = "REALIZE_C1";
const char *const REALIZE_C0 = "REALIZE_C0";
const char *const REALIZE_BUF = "REALIZE_BUF";
const char *const REALIZE_BUFC0 = "REALIZE_BUFC0";
const char *const REALIZE_BUFC1 = "REALIZE_BUFC1";
const char *const REALIZE_C1BUFC1 = "REALIZE_C1BUFC1";
const char *const PRAGMA_BYPATH_FILTER_C0 = "PRAGMA_BYPATH_FILTER_C0";
const char *const PRAGMA_BYPATH_FILTER_C1 = "PRAGMA_BYPATH_FILTER_C1";

const int FLOW_S = 0;
const int FLOW_V = 1;
const int FLOW_M = 2;
const int FLOW_DMA1 = 0;
const int FLOW_DMA2 = 1;
const int FLOW_DMA3 = 2;

const char *const PRAGMA_MMU_C0WRITE = "PRAGMA_MMU_C0WRITE";
const char *const PRAGMA_MMU_C1WRITE = "PRAGMA_MMU_C1WRITE";
const char *const K_C1 = "K_C1";
const char *const PRAGMA_GEMM_C0 = "PRAGMA_GEMM_C0";

const DataFlowAttrs Mmu_Conv_A = {{MemType::DDR, ""}};
const DataFlowAttrs Mmu_Conv_B = {{MemType::DDR, ""}};
const DataFlowAttrs Mmu_Conv_C = {{MemType::DDR, ""}};
const DataFlowAttrs Mmu_Spec_Gemm_A = {{MemType::DDR, ""}};
const DataFlowAttrs Mmu_Spec_Gemm_A_ = {{MemType::DDR, ""}};
const DataFlowAttrs Mmu_Gemm_A = {{MemType::DDR, ""}};
const DataFlowAttrs Mmu_Spec_Gemm_B = {{MemType::DDR, ""}};
const DataFlowAttrs Mmu_Spec_Gemm_B_ = {{MemType::DDR, ""}};
const DataFlowAttrs Mmu_Gemm_B = {{MemType::DDR, ""}};
const DataFlowAttrs Mmu_Spec_Gemm_C = {{MemType::DDR, ""}};
const DataFlowAttrs Mmu_Gemm_C = {{MemType::DDR, ""}};
const DataFlowAttrs Inst_BUF = {{MemType::DDR, ""}};
const DataFlowAttrs Im2Col_C1 = {{MemType::DDR, ""}};
const DataFlowAttrs Mmu_BIAS = {{MemType::DDR, ""}};

int GetCoreValue(const std::string &name) {
  if (name == "Core_num") {
    return 32;
  }
  return -1;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif