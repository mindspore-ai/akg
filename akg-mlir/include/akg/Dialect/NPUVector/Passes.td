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

#ifndef AKG_MLIR_DIALECT_NPUVECTOR_PASSES
#define AKG_MLIR_DIALECT_NPUVECTOR_PASSES

include "mlir/Pass/PassBase.td"

def EliminateNPUVectorRedundantOps
    : Pass<"eliminate-npuvector-redundant-ops", "mlir::func::FuncOp"> {
  let summary = "Eliminate semantically redundant NPUVector reads and seed reads";
  let description = [{
    Performs local forwarding for NPUVector transfer_read operations. Reads from
    the same source/index tile are reused when there is no intervening write,
    and reads immediately satisfied by a previous transfer_write to the same
    tile are replaced by the written SSA value. A following CSE can then merge
    duplicate broadcasts and elementwise ops.
  }];
  let constructor = "mlir::npuvector::createEliminateNPUVectorRedundantOpsPass()";
}

def OutlineVectorFunction : Pass<"outline-vector-function", "mlir::ModuleOp"> {
  let summary = "This pass is designed to support SMID Reg";
  let description = [{
    This pass scans each function in a module and outlines contiguous regions of
    NPUVector-based computation into separate private functions. It is designed
    to separate pure NPUVector compute logic from surrounding control flow and
    memory transfer operations.
  }];
  let constructor = "mlir::npuvector::createOutlineVectorFunctionPass()";
}

def ElimScfIterArgs : Pass<"elim-scf-iter-args", "mlir::ModuleOp"> {
  let summary = "The pass eliminates NPUVector-typed iter_args of scf.for via memref buffers";
  let description = [{
    This pass rewrites `scf.for` loops whose `iter_args` contain NPUVector
    values into an equivalent form where cross-iteration data flow is
    materialized through explicit `memref` buffers and
    `npuvector.transfer_read` / `npuvector.transfer_write` ops, instead of
    implicit SSA `iter_args`.

    For each target loop, the pass:
      1. Allocates one `memref` per NPUVector `iter_arg` before the loop and
         seeds it with the original init value via `transfer_write`.
      2. Rebuilds the loop without `iter_args`: reloads each value from its
         buffer at the top of the body, clones the original body through an
         `IRMapping`, and writes each yielded value back to its buffer at
         the end of the body.
      3. Reloads the final values from the buffers after the loop, replaces
         all uses of the original `scf.for` results, and erases the old loop.

    Loops are processed in reverse walk order so inner loops are rewritten
    before their enclosing ones. Functions tagged with `kVectorFunctionAttr`
    are skipped.
  }];
  let constructor = "mlir::npuvector::createElimScfIterArgsPass()";
}

#endif // AKG_MLIR_DIALECT_NPUVECTOR_PASSES
