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

#include "akg/Dialect/MindSpore/Transforms/AKGSplitGraph.h"

#include <experimental/filesystem>
#include <iostream>
#include "akg/Dialect/MindSpore/Spliter/Spliter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"

namespace mlir {
#define GEN_PASS_DEF_AKGSPLITGRAPH
#define GEN_PASS_DECL_AKGSPLITGRAPH
#include "akg/Dialect/MindSpore/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
namespace fs = std::experimental::filesystem;

namespace {
struct AKGSplitGraph : public impl::AKGSplitGraphBase<AKGSplitGraph> {
  AKGSplitGraph() = default;
  explicit AKGSplitGraph(const std::string &newDumpDir) : dumpDir(newDumpDir) {}
  void runOnOperation() override;
  const std::string dumpDir;
};

ModuleOp modulePack(func::FuncOp func) {
  auto context = func.getContext();
  auto packModule = ModuleOp::create(UnknownLoc::get(context));
  packModule.push_back(func->clone());
  return packModule;
}

bool createDirectory(const std::string &path) {
  if (fs::exists(path)) {
    if (fs::is_directory(path)) {
      return true;
    } else {
      llvm::errs() << "A file with the same name already exists: " << path << "\n";
      return false;
    }
  }

  if (fs::create_directories(path)) {
    return true;
  } else {
    llvm::errs() << "Failed to create directory: " << path << "\n";
    return false;
  }
}

bool dumpMainFuncJson(ModuleOp moduleOp, const std::string &dumpDir, const std::string &funcName) {
  auto jsonStr = mlirToJson(moduleOp);
  if (jsonStr.empty()) {
    llvm::errs() << "Convert mlir to json failed.\n";
    return false;
  }
  std::string fileName = dumpDir + "/" + funcName + "_split.json";
  if (llvm::writeFileAtomically("tmp_%%%%%%%%.json", fileName, jsonStr)) {
    llvm::errs() << "Write json file to " << fileName << " failed.\n";
    return false;
  }
  return true;
}

bool dumpSplitedFuncsMlir(const llvm::SmallVector<func::FuncOp> &splitedFuncs, const std::string &dumpDir) {
  std::vector<std::pair<ModuleOp, std::string>> packModules;
  (void)std::transform(splitedFuncs.begin(), splitedFuncs.end(), std::back_inserter(packModules),
                       [](func::FuncOp func) { return std::pair(modulePack(func), func.getSymName().str()); });
  for (auto &pair : packModules) {
    auto op = pair.first;
    std::string fileName = dumpDir + "/" + pair.second + ".mlir";
    if (llvm::writeFileAtomically("tmp_%%%%%%%%.mlir", fileName, [&op](llvm::raw_ostream &OS) {
          OS << op;
          return llvm::Error::success();
        })) {
      llvm::errs() << "Dump splited mlir to " << fileName << " failed.\n";
      return false;
    }
  }
  return true;
}
}  // namespace

void AKGSplitGraph::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  auto splitedFuncs = spliter::split(funcOp);
  if (dumpDir.empty() || !createDirectory(dumpDir)) {
    return;
  }
  if (splitedFuncs.size() > 1 &&
      !dumpMainFuncJson(funcOp->getParentOfType<ModuleOp>(), dumpDir, funcOp.getSymName().str())) {
    llvm::errs() << "Dump main func json failed.\n";
    return;
  }

  (void)dumpSplitedFuncsMlir(splitedFuncs, dumpDir);
  return;
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGSplitGraphPass(const std::string &dumpDir) {
  return std::make_unique<AKGSplitGraph>(dumpDir);
}
