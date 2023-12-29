/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef COMPILER_INCLUDE_AKG_CONVERSION_PASSES_H_
#define COMPILER_INCLUDE_AKG_CONVERSION_PASSES_H_

#include "akg/Conversion/FusionToMemVec/FusionToMemVecPass.h"
#include "akg/Conversion/LinalgExtLower/LinalgExtLower.h"
#include "akg/Conversion/MathExtLower/MathExtLower.h"
#include "akg/Conversion/MathExtToLibm/MathExtToLibm.h"
#include "akg/Conversion/MindSporeFinalizingLower/MindSporeFinalizingLower.h"
#include "akg/Conversion/MindSporeToLinalg/MindSporeToLinalg.h"
#include "akg/Conversion/MindSporeToTosa/MindSporeToTosa.h"
#include "akg/Conversion/PureOpenMPToLLVM/PureOpenMPToLLVM.h"
#include "akg/Conversion/TosaToLinalgUpdate/TosaMultiReduceToLinalg.h"
#include "akg/Conversion/VectorTransferLower/VectorTransferLower.h"

namespace mlir {

void populateMindSporeLowerPattern(RewritePatternSet &patterns);

/// Generate the code for registering conversion passes.
#ifndef GEN_PASS_REGISTRATION
#define GEN_PASS_REGISTRATION
#include "akg/Conversion/Passes.h.inc"
#endif
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_CONVERSION_PASSES_H_
