/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MFUSION_CONVERSION_PDLL_HELPER_H
#define MFUSION_CONVERSION_PDLL_HELPER_H

#include <cstddef>

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

template <int N>
static Value getIndex(PatternRewriter &rewriter, ValueRange values) {
  (void)rewriter;
  if (N < 0 || static_cast<size_t>(N) >= values.size()) {
    return Value();
  }
  return values[N];
}

struct PDLRewriteHelpers {
  static Attribute getF64Attr(PatternRewriter &rewriter, Value val) {
    APFloat floatVal(0.0);
    if (matchPattern(val, m_ConstantFloat(&floatVal))) {
      return rewriter.getF64FloatAttr(floatVal.convertToDouble());
    }

    APInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal))) {
      return rewriter.getF64FloatAttr(static_cast<double>(intVal.getSExtValue()));
    }

    return rewriter.getF64FloatAttr(0);
  }
};

// Register PDL native rewrite functions shared by conversion patterns.
inline void registerPDLLHelperFunctions(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerRewriteFunction("Get0", getIndex<0>);
  patterns.getPDLPatterns().registerRewriteFunction("Get1", getIndex<1>);
  patterns.getPDLPatterns().registerRewriteFunction("Get2", getIndex<2>);
  patterns.getPDLPatterns().registerRewriteFunction("Get3", getIndex<3>);
  patterns.getPDLPatterns().registerRewriteFunction("Get4", getIndex<4>);
  patterns.getPDLPatterns().registerRewriteFunction("Get5", getIndex<5>);
  patterns.getPDLPatterns().registerRewriteFunction("Get6", getIndex<6>);
  patterns.getPDLPatterns().registerRewriteFunction("Get7", getIndex<7>);
  patterns.getPDLPatterns().registerRewriteFunction("Get8", getIndex<8>);
  patterns.getPDLPatterns().registerRewriteFunction("Get9", getIndex<9>);
  patterns.getPDLPatterns().registerRewriteFunction("Get10", getIndex<10>);
  patterns.getPDLPatterns().registerRewriteFunction("GetF64Attr", PDLRewriteHelpers::getF64Attr);
}

}  // namespace mlir

#endif  // MFUSION_CONVERSION_PDLL_HELPER_H
