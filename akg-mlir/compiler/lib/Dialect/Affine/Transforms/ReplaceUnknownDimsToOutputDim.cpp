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

#include "akg/Dialect/Affine/Transforms/ReplaceUnknownDimsToOutputDim.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <regex>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_REPLACEUNKNOWNDIMSTOOUTPUTDIM
#define GEN_PASS_DECL_REPLACEUNKNOWNDIMSTOOUTPUTDIM
GEN_PASS_DECL_REPLACEUNKNOWNDIMSTOOUTPUTDIM
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "replace-unknown-dims-to-output-dim"

using namespace mlir;
using namespace llvm;
using namespace akgglobal;

namespace {

std::vector<std::string> findSymbol(const std::string &str) {
  std::vector<std::string> result;

  if (str.find("ScalarMax") == std::string::npos) {
    return result;
  }
  std::regex r("s[0-9]+");
  std::smatch m;
  std::string s = str;
  while (std::regex_search(s, m, r)) {
    for (auto x : m) {
      result.push_back(x);
    }
    s = m.suffix().str();
  }

  return result;
}

std::string getOutputSymbol(mlir::DictionaryAttr dictAttr, std::string target) {
  for (auto &keyValuePair : dictAttr) {
    auto key = keyValuePair.getName().str();
    auto value = keyValuePair.getValue().cast<mlir::StringAttr>().str();
    std::vector<std::string> symbols = findSymbol(value);
    bool flag = false;
    for (auto s : symbols) {
      if (s == target) {
        flag = true;
        break;
      }
    }
    if (flag) {
      return key;
    }
  }
  return "";  // should raise error
}

void collectRelatedDimsAndAffineMax(func::FuncOp funcOp, SmallVector<SmallVector<Operation *, 8>, 8> &dimPack,
                                    SmallVector<Operation *, 8> &maxList) {
  funcOp.walk([&](affine::AffineMaxOp maxOp) {
    bool flag = true;
    dimPack.push_back(SmallVector<Operation *, 8>());
    auto new_idx = dimPack.size() - 1;
    for (auto operand : maxOp.getOperation()->getOperands()) {
      auto op = operand.getDefiningOp();
      if (isa<memref::DimOp>(op)) {
        dimPack[new_idx].push_back(op);
      } else {
        flag = false;
        break;
      }
    }
    if (!flag) {
      dimPack.pop_back();
    } else {
      maxList.push_back(maxOp.getOperation());
    }
  });
}

int getArgIdx(func::FuncOp funcOp, BlockArgument arg) {
  size_t argIdx = 0;
  for (BlockArgument &bArg : funcOp.getArguments()) {
    if (bArg == arg) {
      return argIdx;
    }
    argIdx++;
  }
  return -1;
}

std::tuple<int, int> getOutputFromSymbol(const std::string &target, std::vector<ShapeInfo> shapeInfoList) {
  int argIdx = 0;
  int pos = 0;
  for (auto &shape : shapeInfoList) {
    pos = 0;
    for (auto &s : shape) {
      if (s == target) {
        return std::make_tuple(argIdx, pos);
      }
      pos++;
    }
    argIdx++;
  }
  return std::make_tuple(-1, -1);
}

int getPos(mlir::Value value) {
  auto op = value.getDefiningOp();

  int v = -1;
  if (auto constOp = llvm::dyn_cast_or_null<mlir::arith::ConstantIndexOp>(op)) {
    v = constOp.value();
  }
  assert(v != -1);
  return v;
}

class ReplaceUnknownDimsToOutputDimPass
    : public impl::ReplaceUnknownDimsToOutputDimBase<ReplaceUnknownDimsToOutputDimPass> {
 public:
  ReplaceUnknownDimsToOutputDimPass() {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    Operation *moduleOp = funcOp->getParentOp();
    MLIRContext *context = &getContext();
    auto dict = DictionaryAttr::get(context);
    if (isa<ModuleOp>(moduleOp)) {
      if (moduleOp->hasAttr("mindspore.symbol_calc_expr")) {
        dict = dyn_cast<DictionaryAttr>(moduleOp->getAttr("mindspore.symbol_calc_expr"));
      }
    }

    if (dict.empty()) {
      return;
    }
    SmallVector<SmallVector<Operation *, 8>, 8> dimPack;
    SmallVector<Operation *, 8> maxList;
    collectRelatedDimsAndAffineMax(funcOp, dimPack, maxList);

    ShapeAlignTool &tool = ShapeAlignTool::getInstance();

    OpBuilder builder(funcOp);

    for (size_t i = 0; i < dimPack.size(); i++) {
      // 1. get corresponded dim's output symbol's argIdx & position
      auto firstDim = dimPack[i][0];
      auto pos = getPos(firstDim->getOperands()[1]);
      auto argIdx = getArgIdx(funcOp, dyn_cast<BlockArgument>(firstDim->getOperands()[0]));
      auto symbol = tool.getCurrShapeInfo(argIdx)[pos];
      auto outSymbol = getOutputSymbol(dict, symbol);
      int outPos = -1, outArgIdx = -1;
      std::tie(outArgIdx, outPos) = getOutputFromSymbol(outSymbol, tool.getDeviceShapesList());
      auto output = funcOp.getBody().front().getArgument(outArgIdx);
      // 2. replace those memref.dim to output dim
      builder.setInsertionPoint(firstDim);
      auto outDim = builder.create<memref::DimOp>(firstDim->getLoc(), output, static_cast<int64_t>(outPos));
      maxList[i]->replaceAllUsesWith(outDim);
      maxList[i]->erase();
      for (auto &dim : dimPack[i]) {
        dim->replaceAllUsesWith(outDim);
        dim->erase();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createReplaceUnknownDimsToOutputDimPass() {
  return std::make_unique<ReplaceUnknownDimsToOutputDimPass>();
}
