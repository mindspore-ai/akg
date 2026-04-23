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

#ifndef AKG_MLIR_DIALECT_HIVM_PASSES
#define AKG_MLIR_DIALECT_HIVM_PASSES

include "mlir/Pass/PassBase.td"

def AlignAllocBuffer : Pass<"align-alloc-buffer", "func::FuncOp"> {
  let summary = "Pad memref.alloc for HIVM vtranspose/vcast alignment requirements";
  let description = [{
    For each `func.func`, inspects `hivm.hir.vtranspose` and `hivm.hir.vcast` uses, derives per-dimension
    alignment in bytes (32-byte hardware line baseline; f32 two-axis transpose double-align when needed;
    i32/i16 to i8 cast layout rules), records requirements on root `memref.alloc` via
    `akg.alloc_align_dims` / `akg.alloc_align_bytes`, then greedily rewrites those allocs to larger
    backing buffers with `memref.subview` and propagates `builtin.unrealized_conversion_cast` through
    memref view ops so types stay consistent.
  }];
  let constructor = "mlir::createAlignAllocBufferPass()";
}

#endif // AKG_MLIR_DIALECT_HIVM_PASSES
