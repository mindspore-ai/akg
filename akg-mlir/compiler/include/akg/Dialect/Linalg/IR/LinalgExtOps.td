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

#ifndef AKG_DIALECT_LINALG_IR_LINALG_EXT_OPS
#define AKG_DIALECT_LINALG_IR_LINALG_EXT_OPS

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"

include "akg/Dialect/Linalg/IR/LinalgExtBase.td"
include "akg/Dialect/Linalg/IR/LinalgExtInterfaces.td"

// class LinalgExt_Op for the operation in this dialect
class LinalgExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<LinalgExt_Dialect, mnemonic, traits>;

// Base Tablegen class for Linalg ops.
// Linalg ops that correspond to library calls operate on ShapedType as their
// first operands. These may be optionally followed by non-view operands
// depending on the specific Linalg op.
class LinalgExtStructuredBase_Op<string mnemonic, list<Trait> props>
  : Op<LinalgExt_Dialect, mnemonic, !listconcat([
       SingleBlockImplicitTerminator<"mlir::linalg::YieldOp">,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DestinationStyleOpInterface,
       LinalgStructuredInterface,
       RegionBranchOpInterface,
       ReifyRankedShapedTypeOpInterface], props)> {
  code structuredOpsBaseDecls = [{
    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      return !this->getBody()->getOps<mlir::linalg::IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return llvm::cast<mlir::linalg::LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }

    void getSuccessorRegions(
        std::optional<unsigned> index, ArrayRef<Attribute> operands,
        SmallVectorImpl<RegionSuccessor> &regions) {
      // Op has a region, but conceptually the control flow does not enter the
      // region.
    }
  }];
}

def LinalgExt_GatherOp : LinalgExt_Op<"gather", [
  DestinationStyleOpInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LinalgExt gather operation";
  let description = [{
    The `gather` operation returns the slice of the input tensor
    corresponding to the elements of input_indices on the specified axis.
  }];
  let arguments = (ins
    TensorOrMemrefType:$a,
    TensorOrMemrefType:$b,
    I64Attr:$axis,
    TensorOrMemrefType:$output
  );

  let results = (outs
    Variadic<TensorOrMemrefType>:$res
  );
  let extraClassDeclaration = [{
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - 1, getNumOperands};
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def LinalgExt_UnsortedSegmentSumOp : LinalgExt_Op<"unsorted_segment_sum", [
  DestinationStyleOpInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "LinalExt unsortedSegmentSum operation";
  let description = [{
    The `unsorted_segment_sum` operation computes the sum of a tensor along segments.
    Slices are taken from the original data and summed to the corresponding slice in
    the output according to the value in the indices tensor.
  }];
  let arguments = (ins
    TensorOrMemrefType:$a,
    TensorOrMemrefType:$b,
    I64Attr:$num_segments,
    TensorOrMemrefType:$output
  );

  let results = (outs
    Variadic<TensorOrMemrefType>:$res
  );
  let extraClassDeclaration = [{
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      int64_t getNumOperands = this->getNumOperands();
      return {getNumOperands - 1, getNumOperands};
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

#endif // AKG_DIALECT_LINALG_IR_LINALG_EXT_OPS