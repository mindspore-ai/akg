// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2026 Huawei Technologies Co., Ltd
//
// SPDX-License-Identifier: Apache-2.0
//
// This file was copied from project tenstorrent/tt-mlir
// (https://github.com/tenstorrent/tt-mlir) and modified for AKG-MLIR.

#ifndef AKG_DIALECT_EMITPY_IR_EMITPYBASE_TD
#define AKG_DIALECT_EMITPY_IR_EMITPYBASE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// EmitPy dialect definition.
//===----------------------------------------------------------------------===//

def EmitPy_Dialect : Dialect {
    let name = "emitpy";
    let cppNamespace = "::mlir::emitpy";

    let summary = "Dialect to generate Python from MLIR.";

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base EmitPy operation definition.
//===----------------------------------------------------------------------===//

class EmitPy_Op<string mnemonic, list<Trait> traits = []>
    : Op<EmitPy_Dialect, mnemonic, traits>;

#endif // AKG_DIALECT_EMITPY_IR_EMITPYBASE_TD
