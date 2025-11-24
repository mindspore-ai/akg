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

#include <nlohmann/json.hpp>
#include <string>

namespace mlir {
#define GEN_PASS_DECL_LOADGLOBALCONFIG
#define GEN_PASS_DEF_LOADGLOBALCONFIG
#define GEN_PASS_CLASSES
#include "akg/Transforms/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "load-global-config"

using namespace mlir;
using namespace akgglobal;
using json = nlohmann::json;

namespace {
constexpr auto kHostShapes = "hostShapes";
constexpr auto kDeviceShapes = "deviceShapes";
constexpr auto kOutputIndices = "outputIndices";
constexpr auto kGpuSchedules = "gpuSchedules";
constexpr auto kAxisRootName = "axisRootName";
constexpr auto kAxisInfoMap = "axisInfoMap";
constexpr auto kLoopStructure = "loopStructure";

// ===----------------------------------------------------------------------===//
// LoadGlobalConfigPass
// Users may inject some custom configs during lowering pipelines for performance
// considerations. This pass receives user configs as json file and convertes
// supported custom configs into global vars.
// E.g. run pipeline with option `global-config-file=custom_config.json`
// ===----------------------------------------------------------------------===//

class LoadGlobalConfigPass : public impl::LoadGlobalConfigBase<LoadGlobalConfigPass> {
 public:
  LoadGlobalConfigPass() {}
  explicit LoadGlobalConfigPass(const std::string &jsonFileName) { fileName = jsonFileName; }
  void runOnOperation() override;
  void loadFileToGlobal();

 private:
  nlohmann::json rawJson;
  void parseShapeInfo();
  void parseGpuSchedule();
};

void LoadGlobalConfigPass::parseShapeInfo() {
  if (rawJson.contains(kHostShapes)) {
    std::map<size_t, ShapeInfo> init;
    for (size_t i = 0; i < rawJson.at(kHostShapes).size(); ++i) {
      init[i] = rawJson.at(kHostShapes)[i].get<std::vector<std::string>>();
    }
    ShapeAlignTool &tool = ShapeAlignTool::getInstance();
    tool.setHostShapes(init);
  }
  if (rawJson.contains(kDeviceShapes)) {
    std::map<size_t, ShapeInfo> init;
    for (size_t i = 0; i < rawJson.at(kDeviceShapes).size(); ++i) {
      init[i] = rawJson.at(kDeviceShapes)[i].get<std::vector<std::string>>();
    }
    ShapeAlignTool &tool = ShapeAlignTool::getInstance();
    tool.setDeviceShapes(init);
  }
  if (rawJson.contains(kOutputIndices)) {
    auto outputIndices = rawJson.at(kOutputIndices).get<std::unordered_set<size_t>>();
    ShapeAlignTool &tool = ShapeAlignTool::getInstance();
    tool.setOutputIndices(outputIndices);
  }
}

void LoadGlobalConfigPass::parseGpuSchedule() {
  if (rawJson.contains(kGpuSchedules)) {
    auto gpuSchedules = rawJson.at(kGpuSchedules);
    if (gpuSchedules.contains(kAxisRootName)) {
      auto axisRootName = gpuSchedules.at(kAxisRootName).get<std::map<std::string, std::string>>();
      GpuScheduleTool::getInstance().setAxisRootName(axisRootName);
    }

    if (gpuSchedules.contains(kAxisInfoMap)) {
      auto axisInfoMap = gpuSchedules.at(kAxisInfoMap);
      for (auto it = axisInfoMap.begin(); it != axisInfoMap.end(); ++it) {
        std::string name = it.key();
        auto infoList = axisInfoMap[name];
        for (size_t idx = 0; idx < infoList.size(); ++idx) {
          auto eachInfo = infoList[idx];
          auto axisInfo = AxisInfo(eachInfo["name"], eachInfo["loc"]);
          axisInfo.size = eachInfo["size"];
          axisInfo.constSize = eachInfo["constSize"];
          axisInfo.mapLevel = eachInfo["mapLevel"];
          axisInfo.mapDim = eachInfo["mapDim"];
          axisInfo.tileLevel = eachInfo["tileLevel"];
          GpuScheduleTool::getInstance().add(axisInfo);
        }
      }
    }

    if (gpuSchedules.contains(kLoopStructure)) {
      auto loopStructure = gpuSchedules.at(kLoopStructure).get<std::vector<std::string>>();
      GpuScheduleTool::getInstance().updateLoopStructure(loopStructure);
    }

    GpuScheduleTool::getInstance().setIsCustomConfig(true);
  }
}

void LoadGlobalConfigPass::loadFileToGlobal() {
  rawJson = DirUtils::checkAndReadJson(fileName);
  parseShapeInfo();
  parseGpuSchedule();
}

void LoadGlobalConfigPass::runOnOperation() {
  if (fileName.empty()) {
    return;
  }
  loadFileToGlobal();
}
}  // end anonymous namespace

std::unique_ptr<Pass> mlir::createLoadGlobalConfigPass() { return std::make_unique<LoadGlobalConfigPass>(); }

std::unique_ptr<Pass> mlir::createLoadGlobalConfigPass(const std::string &fileName) {
  return std::make_unique<LoadGlobalConfigPass>(fileName);
}
