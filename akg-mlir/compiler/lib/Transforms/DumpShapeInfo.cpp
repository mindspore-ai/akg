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

#include "akg/Transforms/Passes.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/IOHelper.hpp"
#include "llvm/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_set>

namespace mlir {
#define GEN_PASS_DECL_DUMPSHAPEINFO
#define GEN_PASS_DEF_DUMPSHAPEINFO
#define GEN_PASS_CLASSES
#include "akg/Transforms/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "dump-shape-info"

using namespace mlir;
using namespace akgglobal;
using json = nlohmann::json;

namespace {
constexpr auto kHostShapes = "hostShapes";
constexpr auto kDeviceShapes = "deviceShapes";
constexpr auto kRuntimeVars = "runtimeVars";
constexpr auto kRuntimeVarsPrime = "prime";
constexpr auto kRuntimeVarsArgIndex = "argIndex";
constexpr auto kRuntimeVarsMapping = "mapping";
constexpr auto kRuntimeVarsMapDim = "mapDim";
constexpr auto kRuntimeVarsExpr = "expr";
constexpr auto kRuntimeVarsMark = "mark";

constexpr auto kKernelNameAttrKey = "sym_name";

// ===----------------------------------------------------------------------===//
// DumpShapeInfoPass
// AKG may reconstruct the inputs' and outputs' shapes during optimization
// passes like FoldDimension, UnifyShape and that lead to imcompactble shapes
// and strides between host-side and device-side.
// In static-shape cases, the shapes and strides are constants, while in dynamic-shape
// cases, they are also input arguments and we pass them during runtime.
// Therefore, we need to know the shape info (i.e. optimized device-side shapes and strides)
// so each time we modify the shapes, we record it in the `ShapeAlignTool`.
// This pass will dump shape info into `kernel_name_shape_info.json` file.
// ===----------------------------------------------------------------------===//

class DumpShapeInfoPass : public impl::DumpShapeInfoBase<DumpShapeInfoPass> {
 public:
  DumpShapeInfoPass() {}
  explicit DumpShapeInfoPass(const std::string &jsonFileName) { fileName = jsonFileName; }
  void runOnOperation() override;
  bool save(const std::string &res);
  std::string getAkgKernelName();

 private:
  void dumpGpuRuntimeVars(json &jsonResults) const;
  void dumpGpuSupportInfo(json &jsonResults);
  void dumpGpuSchedule(json &jsonResults) const;
};

std::string DumpShapeInfoPass::getAkgKernelName() {
  std::string defaultName = "akg_kernel";
  func::FuncOp funcOp;
  getOperation()->walk([&](func::FuncOp op) { funcOp = op; });
  if (!funcOp) {
    return defaultName;
  }
  for (auto attr : funcOp->getAttrs()) {
    auto keyStr = dyn_cast<StringAttr>(attr.getName()).getValue().str();
    if (keyStr != kKernelNameAttrKey) {
      continue;
    }
    return dyn_cast<StringAttr>(attr.getValue()).getValue().str();
  }
  return defaultName;
}

bool DumpShapeInfoPass::save(const std::string &res) {
  if (res.empty()) {
    llvm::errs() << "Save json failed: string empty.\n";
    return false;
  }
  (void)DirUtils::CheckOrCreateDirectory("./akg_kernel_meta/");
  if (!fileName.empty()) {
    std::string search = ".info";
    std::string replacement = "_shape_info.json";
    size_t pos = fileName.find(search);
    if (pos != std::string::npos) {
      (void)fileName.replace(pos, search.length(), replacement);
    }
  } else {
    fileName = getAkgKernelName() + "_shape_info.json";
  }

  std::string output_filename = "./akg_kernel_meta/" + fileName;
  llvm::outs() << "Dump to " << output_filename << "\n";
  if (llvm::writeToOutput(output_filename, [&](llvm::raw_ostream &OS) -> llvm::Error {
        OS << res;
        return llvm::Error::success();
      })) {
    llvm::errs() << "Write json file to " << output_filename << " failed.\n";
    return false;
  }
  return true;
}

void DumpShapeInfoPass::runOnOperation() {
  json jsonResults;
  ShapeAlignTool &tool = ShapeAlignTool::getInstance();
  jsonResults[kHostShapes] = tool.getHostShapesList();
  jsonResults[kDeviceShapes] = tool.getDeviceShapesList();
  dumpGpuRuntimeVars(jsonResults);
  dumpGpuSupportInfo(jsonResults);
  dumpGpuSchedule(jsonResults);
  save(jsonResults.dump());
}
}  // end anonymous namespace

void DumpShapeInfoPass::dumpGpuRuntimeVars(json &jsonResults) const {
  std::vector<json> temp;
  auto &gpuTool = akgglobal::GpuScheduleTool::getInstance();
  for (auto it : gpuTool.getRuntimeVars()) {
    json subJson;
    auto var = it.second;
    subJson[kRuntimeVarsPrime] = var.prime;
    subJson[kRuntimeVarsArgIndex] = var.argIndex;
    subJson[kRuntimeVarsMapping] = var.mapping;
    subJson[kRuntimeVarsMapDim] = var.mapDim;
    subJson[kRuntimeVarsExpr] = var.expr;
    subJson[kRuntimeVarsMark] = var.mark;
    temp.push_back(subJson);
  }
  jsonResults[kRuntimeVars] = temp;
}
void DumpShapeInfoPass::dumpGpuSupportInfo(json &jsonResults) {
  auto &gpuTool = akgglobal::GpuScheduleTool::getInstance();
  std::string opType{"Unknown"};
  getOperation()->walk([&](func::FuncOp funcOp) {
    auto op = funcOp.getOperation();
    if (op->hasAttr("OperatorType")) {
      opType = dyn_cast<StringAttr>(op->getAttr("OperatorType")).getValue().str();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  json dynAlgo;
  dynAlgo["OperatorType"] = opType;
  dynAlgo["DynAlgorithm"] = gpuTool.dynAlgorithm;
  dynAlgo["ReduceSizeStatic"] = gpuTool.reduceSizeStatic;
  dynAlgo["ParallelSizeStatic"] = gpuTool.parallelSizeStatic;
  dynAlgo["EnableAtomic"] = gpuTool.enableAtomic;
  jsonResults["SupportInfo"] = dynAlgo;
}

void DumpShapeInfoPass::dumpGpuSchedule(json &jsonResults) const {
  auto &gpuTool = akgglobal::GpuScheduleTool::getInstance();
  json gpuSchedules;
  gpuSchedules["scheduleSize"] = gpuTool.scheduleSize();
  gpuSchedules["loopSize"] = gpuTool.loopSize();
  gpuSchedules["loopStructure"] = gpuTool.getLoopStructure();
  json axisInfoMap;
  for (auto it : gpuTool.getAxisInfoMap()) {
    std::vector<json> axisInfoList;
    for (auto axisInfo : it.second) {
      json subJson;
      subJson["name"] = axisInfo.name;
      subJson["loc"] = axisInfo.loc;
      subJson["size"] = axisInfo.size;
      subJson["constSize"] = axisInfo.constSize;
      subJson["mapLevel"] = axisInfo.mapLevel;
      subJson["mapDim"] = axisInfo.mapDim;
      subJson["tileLevel"] = axisInfo.tileLevel;
      axisInfoList.push_back(subJson);
    }
    axisInfoMap[it.first] = axisInfoList;
  }
  gpuSchedules["axisInfoMap"] = axisInfoMap;
  gpuSchedules["axisRootName"] = gpuTool.getAxisRootName();
  jsonResults["gpuSchedules"] = gpuSchedules;
}

std::unique_ptr<Pass> mlir::createDumpShapeInfoPass() { return std::make_unique<DumpShapeInfoPass>(); }

std::unique_ptr<Pass> mlir::createDumpShapeInfoPass(const std::string &fileName) {
  return std::make_unique<DumpShapeInfoPass>(fileName);
}
