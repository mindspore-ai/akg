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

#ifndef AKG_MLIR_DIALECT_MINDSPORE_PASSES
#define AKG_MLIR_DIALECT_MINDSPORE_PASSES

include "mlir/Pass/PassBase.td"

include "akg/Dialect/MindSpore/IR/MindSporeBase.td"
include "akg/Dialect/MindSpore/IR/MindSporeOps.td"

def RemoveRedundantReduce : Pass<"remove-redundant-reduce", "func::FuncOp"> {
  let summary = "Remove redundant reduce operations";
  let description = [{
    If the shapes of all the reduced axis equal to 1, the reduce operation is redundant.
  }];

  let options = [];

  let constructor = "mlir::createRemoveRedundantReducePass()";
}

def MoveDownReductionOps : Pass<"move-down-reduction-ops", "func::FuncOp"> {
  let summary = "Move down reduction related ops";
  let description = [{
    Move reduction related ops down, so that we can generate better performance kernels
  }];

  let options = [];

  let constructor = "mlir::createMoveDownReductionOpsPass()";
}

def EliminateDimension : Pass<"eliminate-dimension", "func::FuncOp"> {
  let summary = "MindSpore eliminate dimension";
  let description = [{
    Pass that enables broadcast by making all input arrays have the same shape in
    shape scenario. Insert mindspore.reshape op to preprocess all input parameters.
  }];

  let constructor = "mlir::createEliminateDimensionPass()";
}

def LegalizeType : Pass<"legalize-type", "func::FuncOp"> {
  let summary = "MindSpore legalize type";
  let description = [{
    support legalize ui8 to i64 for sub op.
  }];

  let constructor = "mlir::createLegalizeTypePass()";
}

def AlignDimension : Pass<"align-dimension", "func::FuncOp"> {
  let summary = "MindSpore align dimension";
  let description = [{
    Pass that enables broadcast by making all input arrays have the same shape in
    shape scenario. Insert mindspore.reshape op to preprocess all input parameters.
  }];

  let constructor = "mlir::createAlignDimensionPass()";
}

def MindsporeMakeBroadcastable : Pass<"mindspore-make-broadcastable", "func::FuncOp"> {
  let summary = "MindSpore rank Reshape to enable Broadcasting";
  let description = [{
    Pass that enables broadcast by making all input arrays have the same shape in
    shape scenario. Insert mindspore.broadcast_to op to preprocess all input parameters.
  }];

  let constructor = "mlir::createMindsporeMakeBroadcastablePass()";
}

def MakeDynamicBroadcastable : Pass<"make-dyn-broadcastable", "func::FuncOp"> {
  let summary = "MindSpore dynaic rank Reshape to enable Broadcasting";
  let description = [{
    Pass that enables broadcast by making all input arrays have the same shape in dynamic
    shape scenario. Insert mindspore.broadcast_to op to preprocess all input parameters.
  }];

  let options = [
    Option<"IgnoreImplicitBroadcast", "ignore-implicit-broadcast", "bool", "false",
    "ignore implicit broadcast, cast inputs' shape to out's shape directly. Note that this will generate incorrect code, which needs to be corrected in the following pass.">,
  ];

  let constructor = "mlir::createMakeDynamicBroadcastablePass()";
}

def EliminateReshape : Pass<"eliminate-reshape", "func::FuncOp"> {
    let summary = "Eliminate the ReshapeOp nearly inputs and outputs.";
    let constructor = "mlir::createEliminateReshapePass()";
}

#endif // AKG_MLIR_DIALECT_MINDSPORE_PASSES