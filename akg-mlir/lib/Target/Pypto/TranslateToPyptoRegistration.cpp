// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2026 Huawei Technologies Co., Ltd
//
// SPDX-License-Identifier: Apache-2.0
//
// This file was copied from project tenstorrent/tt-mlir
// (https://github.com/tenstorrent/tt-mlir) and modified for AKG-MLIR.

#include "akg/Dialect/EmitPy/IR/EmitPy.h"
#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Target/Pypto/PyptoEmitter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"

namespace mlir::emitpy {

// Command-line option for file ID filtering.
static llvm::cl::opt<std::string>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    fileIdOption("pypto-file-id",
                 llvm::cl::desc("Filter emitpy.file ops by ID. Only files "
                                "with matching ID will be emitted."),
                 llvm::cl::init(""));

void registerToPyptoTranslation() {
  TranslateFromMLIRRegistration reg(
      "mlir-to-pypto", "translate from mlir to pypto",
      [](Operation *op, raw_ostream &output) {
        std::string fileIdValue = fileIdOption.getValue();
        std::optional<std::string> fileId =
            fileIdValue.empty() ? std::nullopt
                                : std::optional<std::string>(fileIdValue);
        return emitpy::translateToPython(op, output, fileId);
      },
      [](DialectRegistry &registry) {
        // Tile values produced by `emitpy.call_opaque` keep their source
        // (npuvector/vector) types; those dialects must be parseable.
        registry.insert<emitpy::EmitPyDialect, func::FuncDialect,
                        npuvector::NPUVectorDialect, vector::VectorDialect>();
      });
}

}  // namespace mlir::emitpy
