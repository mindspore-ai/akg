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

#ifndef AKG_DIALECT_PYPTO_IR_PYPTOBASE_TD
#define AKG_DIALECT_PYPTO_IR_PYPTOBASE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Pypto dialect definition.
//===----------------------------------------------------------------------===//

def Pypto_Dialect : Dialect {
    let name = "pypto";
    let cppNamespace = "::mlir::pypto";

    let summary = "Tile-level intermediate dialect for the PyPTO execution model.";

    let description = [{
      The `pypto` dialect is a typed intermediate layer between the tile-level
      computation dialects (vector, npuvector, ...) and `emitpy`. Ops and types
      in this dialect model what the PyPTO execution model prescribes (tiles,
      tile movement between tensor memory and cores, tile math), leaving all
      Python syntax encoding to the `emitpy` lowering.
    }];

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base Pypto operation definition.
//===----------------------------------------------------------------------===//

class Pypto_Op<string mnemonic, list<Trait> traits = []>
    : Op<Pypto_Dialect, mnemonic, traits>;

#endif // AKG_DIALECT_PYPTO_IR_PYPTOBASE_TD
