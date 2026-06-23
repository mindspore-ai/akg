/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#include <atomic>
#include "akg/Dialect/Linalg/Passes.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DECL_LINALGTEMPLATED
#define GEN_PASS_DEF_LINALGTEMPLATED
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

namespace {
using mlir::AffineMap;
using mlir::DialectRegistry;
using mlir::dyn_cast;
using mlir::failure;
using mlir::FailureOr;
using mlir::isa;
using mlir::MemRefType;
using mlir::ModuleOp;
using mlir::NamedAttribute;
using mlir::OpBuilder;
using mlir::OperationPass;
using mlir::OwningOpRef;
using mlir::parseSourceFile;
using mlir::SmallVector;
using mlir::StringAttr;
using mlir::SymbolRefAttr;
using mlir::TensorType;
using mlir::Type;
using mlir::TypeRange;
using mlir::Value;
using mlir::ValueRange;
using mlir::WalkResult;

std::atomic<int> instanceIndex{0};

struct LinalgTemplatedPass : public mlir::impl::LinalgTemplatedBase<LinalgTemplatedPass> {
  LinalgTemplatedPass() = default;
  explicit LinalgTemplatedPass(const std::string &templatePath = "") {
    this->templatePath = templatePath;
    instanceIndex = 0;
  }
  LinalgTemplatedPass(const LinalgTemplatedPass &) = default;
  LinalgTemplatedPass &operator=(const LinalgTemplatedPass &) = delete;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::impl::LinalgTemplatedBase<LinalgTemplatedPass>::getDependentDialects(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  }

 private:
  FailureOr<std::string> getTemplateFile(mlir::linalg::LinalgOp linalgOp);
  FailureOr<mlir::func::FuncOp> insertTemplatedFunc(std::string filePath, mlir::func::FuncOp &funcOp);

  void templateLinalgOp(mlir::linalg::LinalgOp linalgOp, mlir::func::FuncOp &insertedFunc);

  std::map<std::string, std::string> op2file = {{"linalg.matmul", "matmul"},
                                                {"linalg.batch_matmul", "batch_matmul"},
                                                {"linalg.matmul_transpose_b", "matmul_transpose_b"},
                                                {"linalg.batch_matmul_4d", "batch_matmul_4d"},
                                                {"linalg.batch_matmul_4d_transpose_b", "batch_matmul_4d_transpose_b"}};
};
}  // namespace

FailureOr<std::string> LinalgTemplatedPass::getTemplateFile(mlir::linalg::LinalgOp linalgOp) {
  auto opName = linalgOp->getName().getStringRef();
  if (op2file.count(opName.lower()) == 0u) {
    return failure();
  }

  std::string type_postfix = "";
  llvm::raw_string_ostream os(type_postfix);
  for (auto type : linalgOp->getOperandTypes()) {
    if (auto tensorType = dyn_cast<TensorType>(type)) {
      os << "_" << tensorType.getElementType();
    } else if (auto memType = dyn_cast<MemRefType>(type)) {
      os << "_" << memType.getElementType();
    } else {
      os << "_" << type;
    }
  }

  auto filePath = this->templatePath.empty()
                    ? "./" + op2file[opName.lower()]
                    : this->templatePath + "/" + op2file[opName.lower()] + type_postfix + ".mlir";
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filePath);

  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open templated file " << filePath << " : " << ec.message() << "\n";
    return failure();
  }

  return filePath;
}

FailureOr<mlir::func::FuncOp> LinalgTemplatedPass::insertTemplatedFunc(std::string filePath,
                                                                       mlir::func::FuncOp &funcOp) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filePath);

  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open templated file: " << ec.message() << "\n";
    return failure();
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, funcOp.getContext());

  auto template_func = module->getOps<mlir::func::FuncOp>().begin();
  if ((*template_func).empty()) {
    llvm::errs() << "No templated function in templated file: " << filePath << "\n";
    return failure();
  }

  OpBuilder insertBuilder(funcOp);
  auto insertedOp = insertBuilder.clone(*(*template_func).getOperation());
  auto insertedFunc = dyn_cast<mlir::func::FuncOp>(insertedOp);
  assert(insertedFunc);

  // set different name for different linalg.op
  insertedFunc.setName(insertedFunc.getSymName().str() + "_" + std::to_string(instanceIndex++));
  return insertedFunc;
}

void LinalgTemplatedPass::templateLinalgOp(mlir::linalg::LinalgOp linalgOp, mlir::func::FuncOp &insertedFunc) {
  SmallVector<Value> inputOperands = linalgOp.getDpsInputs();
  SmallVector<Value> outputOperands = linalgOp.getDpsInits();
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  SmallVector<mlir::utils::IteratorType> iterators = linalgOp.getIteratorTypesArray();
  SmallVector<Type> resultTypes =
    linalgOp.hasPureTensorSemantics() ? TypeRange(ValueRange(outputOperands)) : TypeRange{};
  SmallVector<Type> types(resultTypes.begin(), resultTypes.end());

  // All named ops have a region attached that can be inlined.
  assert(linalgOp->getNumRegions() == 1 && "expect named op to have one region attached");

  OpBuilder builder(linalgOp);

  auto fn = SymbolRefAttr::get(builder.getContext(), insertedFunc.getSymName());
  SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(NamedAttribute(StringAttr::get(builder.getContext(), mlir::TemplateFuncAttrName), fn));

  auto templateOp = builder.create<mlir::linalgExt::TemplateOp>(linalgOp.getLoc(), types, inputOperands, outputOperands,
                                                                indexingMaps, iterators, nullptr, attrs);

  templateOp.getRegion().getBlocks().splice(templateOp.getRegion().begin(), linalgOp->getRegion(0).getBlocks());

  linalgOp->replaceAllUsesWith(templateOp);
  linalgOp->erase();
}

void LinalgTemplatedPass::runOnOperation() {
  for (auto func : getOperation().getOps<mlir::func::FuncOp>()) {
    auto funcWalkResult = func.walk([this, &func](mlir::linalg::LinalgOp linalgOp) {
      if (isa<mlir::linalg::GenericOp>(linalgOp) || isa<mlir::linalgExt::TemplateOp>(linalgOp)) {
        return WalkResult::advance();
      }

      auto filePathOrFailure = getTemplateFile(linalgOp);
      if (failed(filePathOrFailure)) {
        // no templated implementation and continue
        return WalkResult::advance();
      }

      auto filePath = *filePathOrFailure;
      auto insertedFuncOrFailure = insertTemplatedFunc(filePath, func);
      if (failed(insertedFuncOrFailure)) {
        // cannot get template func and continue
        return WalkResult::advance();
      }

      auto insertedFunc = *insertedFuncOrFailure;
      templateLinalgOp(linalgOp, insertedFunc);
      return WalkResult::advance();
    });
    if (funcWalkResult.wasInterrupted()) {
      return signalPassFailure();
    }
  }
}

namespace mlir {
std::unique_ptr<OperationPass<ModuleOp>> createLinalgTemplatedPass(std::string templatePath) {
  return std::make_unique<LinalgTemplatedPass>(templatePath);
}
}  // namespace mlir
