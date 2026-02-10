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

#include "mfusion/Dialect/Mfuse/Mfuse.h"

//===----------------------------------------------------------------------===//
// FusedOp Verifier
//===----------------------------------------------------------------------===//

namespace mlir::mfuse {

llvm::LogicalResult FusedOp::verify() {
  // Verify that the body has exactly one block
  if (getBody().empty()) {
    return emitOpError("expected non-empty body region");
  }

  Block &body = getBody().front();

  // Verify that the block has a terminator
  if (body.empty()) {
    return emitOpError("expected non-empty body block");
  }

  Operation *terminator = body.getTerminator();
  if (!terminator) {
    return emitOpError("expected body block to have a terminator");
  }

  // Verify that the terminator is a YieldOp
  auto yield_op = dyn_cast<YieldOp>(terminator);
  if (!yield_op) {
    return emitOpError("expected body block to be terminated with 'mfuse.yield'");
  }

  // Verify that the number of yielded values matches the number of results
  if (yield_op.getValues().size() != getNumResults()) {
    return emitOpError("expected ") << getNumResults() << " yielded values but got " << yield_op.getValues().size();
  }

  // Verify that the types of yielded values match the result types
  for (auto [yielded, result] : llvm::zip(yield_op.getValues(), getResults())) {
    if (yielded.getType() != result.getType()) {
      return emitOpError("type mismatch between yielded value and result: ")
             << yielded.getType() << " vs " << result.getType();
    }
  }

  // Verify that the number of block arguments matches the number of inputs
  if (body.getNumArguments() != getInputs().size()) {
    return emitOpError("expected ") << getInputs().size() << " block arguments but got " << body.getNumArguments();
  }

  // Verify that block argument types match input types
  for (auto [input, arg] : llvm::zip(getInputs(), body.getArguments())) {
    if (input.getType() != arg.getType()) {
      return emitOpError("type mismatch between input and block argument: ")
             << input.getType() << " vs " << arg.getType();
    }
  }

  return llvm::success();
}

}  // namespace mlir::mfuse
