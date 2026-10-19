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

#ifndef AKG_DIALECT_PYPTO_IR_PYPTOATTRS_TD
#define AKG_DIALECT_PYPTO_IR_PYPTOATTRS_TD

include "akg/Dialect/Pypto/IR/PyptoBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Pypto attribute definitions
//===----------------------------------------------------------------------===//

/// Combining function of a `pypto.reduction`; mirrors the `pl.row_*` /
/// `pl.col_*` families and the scalar-form combine ops of the PyPTO
/// runtime.
def Pypto_ReduceKind : I32EnumAttr<"ReduceKind",
    "combining function of a tile reduction", [
  I32EnumAttrCase<"sum", 0>,
  I32EnumAttrCase<"prod", 1>,
  I32EnumAttrCase<"min", 2>,
  I32EnumAttrCase<"max", 3>
]> {
  let cppNamespace = "::mlir::pypto";
  let genSpecializedAttr = 0;
}

def Pypto_ReduceKindAttr : EnumAttr<Pypto_Dialect, Pypto_ReduceKind, "kind">;

/// Axis folded by a partial `pypto.reduction`. `row` folds the last axis
/// onto the `pl.row_*` family, `col` the first axis onto `pl.col_*`.
def Pypto_ReduceAxis : I32EnumAttr<"ReduceAxis",
    "axis folded by a partial tile reduction", [
  I32EnumAttrCase<"row", 0>,
  I32EnumAttrCase<"col", 1>
]> {
  let cppNamespace = "::mlir::pypto";
  let genSpecializedAttr = 0;
}

def Pypto_ReduceAxisAttr : EnumAttr<Pypto_Dialect, Pypto_ReduceAxis, "axis">;

/// Predicate of a `pypto.cmp`; the numeric values match the PyPTO `cmp_type`
/// argument codes (0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge).
def Pypto_CmpPredicate : I32EnumAttr<"CmpPredicate",
    "predicate of a tile comparison", [
  I32EnumAttrCase<"eq", 0>,
  I32EnumAttrCase<"ne", 1>,
  I32EnumAttrCase<"lt", 2>,
  I32EnumAttrCase<"le", 3>,
  I32EnumAttrCase<"gt", 4>,
  I32EnumAttrCase<"ge", 5>
]> {
  let cppNamespace = "::mlir::pypto";
  let genSpecializedAttr = 0;
}

def Pypto_CmpPredicateAttr : EnumAttr<Pypto_Dialect, Pypto_CmpPredicate,
                                      "predicate">;

#endif // AKG_DIALECT_PYPTO_IR_PYPTOATTRS_TD
