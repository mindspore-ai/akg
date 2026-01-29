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

#ifndef MUSE_DIALECT_MUSE_ACLNN_OPS_TD
#define MUSE_DIALECT_MUSE_ACLNN_OPS_TD

include "mfusion/Dialect/Muse/MuseTypes.td"

//===----------------------------------------------------------------------===//
// MUSE Aclnn Operations
//
// NOTE: All operations in this file are sorted alphabetically by their class name.
//===----------------------------------------------------------------------===//

def Muse_AclnnAddOp : Muse_Op<"aclnn.add", [Pure]> {
  let summary = "The aclnnAdd operation";
  let description = [{
    Element-wise addition of two tensors with an optional alpha scaling factor.
    Computes: result = x + y * alpha
  }];

  let arguments = (ins
    AnyRankedTensor:$x,
    AnyRankedTensor:$y,
    AnyScalar:$alpha
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $x `,` $y `,` $alpha attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_AclnnAddRmsNormOp : Muse_Op<"aclnn.add_rms_norm", [Pure]> {
  let summary = "The aclnnAddRmsNorm operation";
  let description = [{
    Operation for fused Add and RmsNorm computation.
    Accepts two input tensors for addition, a gamma scaling factor tensor, and epsilon value for numerical stability.
    Produces normalized output tensor, the standard deviation, and the intermediate addition result.
  }];

  let arguments = (ins
    AnyRankedTensor:$x1,
    AnyRankedTensor:$x2,
    AnyRankedTensor:$gamma,
    F64Attr:$epsilon
  );

  let results = (outs
    AnyRankedTensor:$yOut,
    AnyRankedTensor:$rstdOut,
    AnyRankedTensor:$xOut
  );

  let assemblyFormat = [{
    $x1 `,` $x2 `,` $gamma
    attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_AclnnRmsNormOp : Muse_Op<"aclnn.rms_norm", [Pure]> {
  let summary = "The aclnnRmsNorm operation";
  let description = [{
    Operation for RmsNorm computation.
    Accepts input tensor, gamma scaling factor, and epsilon value for numerical stability.
    Produces normalized output tensor and reciprocal of standard deviation.
  }];

  let arguments = (ins
    AnyRankedTensor:$x,
    AnyRankedTensor:$gamma,
    F64Attr:$epsilon
  );

  let results = (outs
    AnyRankedTensor:$yOut,
    AnyRankedTensor:$rstdOut
  );

  let assemblyFormat = [{
    $x `,` $gamma
    attr-dict `:` functional-type(operands, results)
  }];
}

def Muse_AclnnSubOp : Muse_Op<"aclnn.sub", [Pure]> {
  let summary = "The aclnnSub operation";
  let description = [{
    Element-wise subtraction of two tensors with an optional alpha scaling factor.
    Computes: result = x - y * alpha
  }];

  let arguments = (ins
    AnyRankedTensor:$x,
    AnyRankedTensor:$y,
    AnyScalar:$alpha
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $x `,` $y `,` $alpha attr-dict `:` functional-type(operands, results)
  }];
}

#endif // MUSE_DIALECT_MUSE_ACLNN_OPS_TD
