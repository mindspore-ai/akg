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
#ifndef AKG_UTILS_GLOBALVARS_H_
#define AKG_UTILS_GLOBALVARS_H_

#include <deque>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <array>
#include <vector>
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace akgglobal {

constexpr auto kNeedFix = "need_fix";
constexpr auto kLoopTag = "loop_tag";
constexpr auto kPlaceHolder = "place_holder";
constexpr auto kTileCfg = "Tile";
constexpr auto kGpuSeqCfg = "GpuSeq";
constexpr auto kGpuGridCfg = "GpuGrid";
constexpr auto kGpuBlockCfg = "GpuBlock";
constexpr size_t kPrimeSize = 300;
static const int kGpuSeqMapDim = 3;

// Constexpr prime number generation helpers (C++17 compatible)
namespace prime_helpers {
constexpr bool isPrime(size_t n) {
  if (n < 2) return false;
  if (n == 2) return true;
  if (n % 2 == 0) return false;
  for (size_t i = 3; i * i <= n; i += 2) {
    if (n % i == 0) return false;
  }
  return true;
}

constexpr std::array<size_t, kPrimeSize> generatePrimeList() {
  std::array<size_t, 300> primes{};
  size_t num = 40009;
  for (size_t i = 0; i < 300; ++i) {
    while (!isPrime(num)) ++num;
    primes[i] = num++;
  }
  return primes;
}

constexpr auto primeList = generatePrimeList();

// Compile-time verification
static_assert(primeList[0] == 40009, "First prime should be 40009");
static_assert(primeList[299] == 43093, "Last prime should be 43093");
static_assert(primeList.size() == 300, "Should have 300 primes");
}  // namespace prime_helpers

struct AxisInfo {
  AxisInfo(const std::string &name, const std::pair<int, int> &loc) : name(name), loc(loc) {}
  std::string name;
  std::pair<int, int> loc;
  std::string size;  // maybe symbol
  int constSize{-1};
  int tileLevel{-1};
  std::string mapLevel;  // For GPU
  int mapDim{-1};        // For GPU
  std::string toString() {
    return "AxisInfo:>>Name:" + name + " >>Loc:(" + std::to_string(loc.first) + "," + std::to_string(loc.second) +
           ");" + ";tileLevel " + std::to_string(tileLevel) + ";Size " + size + "; mapLevel " + mapLevel + "." +
           std::to_string(mapDim);
  }
};

struct RuntimeVar {
 public:
  int64_t prime;                   // prime is like a unique id for this var to speedup lower in pipeline
  int argIndex{-1};                // index in the func argument
  std::string mapping{"Default"};  // used for GPU mapping, can be chosen from [Grid, Block, Seq]
  std::string mapDim{""};          // used for GPU mapping, can be chosen from [x, y, z]
  std::string mark{"Default"};

  // use `-1` to represent the whole range of dynamic shape, therefore,
  //  1) `=-1` means tile size equals to dynamic shape;
  //  2) `=min(-1, 32)` means tile size is the small number among dynamic shape and 32;
  std::string expr{""};

  std::string toString() {
    std::string res = "[RuntimeVar " + std::to_string(prime) + "]";
    res += "  -> " + mapping + "." + mapDim + " at " + std::to_string(argIndex) + " input\n";
    res += "  -> expr: " + expr + "\n";
    return res;
  }
};

class GpuScheduleTool {
 public:
  ~GpuScheduleTool() {}
  GpuScheduleTool(const GpuScheduleTool &) = default;
  GpuScheduleTool &operator=(const GpuScheduleTool &) = delete;
  static GpuScheduleTool &getInstance() {
    static GpuScheduleTool instance;
    return instance;
  }

  bool hasGlobalConfig() const { return scheduleSize() > 0; }

  bool getIsCustomConfig() const { return isCustomConfig; }

  void setIsCustomConfig(bool isCustom) { isCustomConfig = isCustom; }

  size_t scheduleSize() const { return axisInfoMap.size(); }

  size_t loopSize() const { return loopStructure.size(); }

  std::vector<std::string> getLoopStructure() const { return loopStructure; }

  void tagLoopWithAxisName(mlir::Operation *funcOp) {
    size_t i = 0;
    size_t loopSize = this->loopSize();
    mlir::OpBuilder builder(funcOp);
    funcOp->walk([&](mlir::Operation *op) {
      if (!mlir::isa<mlir::affine::AffineForOp, mlir::affine::AffineParallelOp, mlir::scf::ParallelOp>(op)) {
        return;
      }
      auto tagName = getNameAt(loopSize - 1 - i);
      if (tagName == kPlaceHolder) {
        return;
      }
      mlir::Attribute attr = builder.getStringAttr(tagName);
      op->setAttr(kLoopTag, attr);
      ++i;
    });
  }

  void recordLoopStructure(const std::string &axisName) { loopStructure.push_back(axisName); }
  void updateLoopStructure(const std::vector<std::string> &newStructure) { loopStructure = newStructure; }

  std::vector<std::string> getNamesAfterTiling(const std::string originName) {
    auto it = axisNameMap.find(originName);
    if (it == axisNameMap.end()) {
      return {};
    }
    return it->second;
  }

  void splitLoop(size_t times) {
    std::vector<std::string> original = loopStructure;
    loopStructure.clear();
    loopStructure.reserve(original.size() * times);
    axisRootName.clear();
    for (size_t i = 0; i < times; ++i) {
      (void)std::transform(original.begin(), original.end(), std::back_inserter(loopStructure),
                           [this, &i](const std::string &origName) {
                             auto newName = origName + "_" + std::to_string(i);
                             axisNameMap[origName].push_back(newName);
                             axisRootName[newName] = origName;
                             return newName;
                           });
    }
  }

  void add(const AxisInfo &axisInfo) {
    auto it = axisRootName.find(axisInfo.name);
    std::string origName = it == axisRootName.end() ? axisInfo.name : it->second;
    axisInfoMap[origName].push_back(axisInfo);
  }

  std::vector<AxisInfo> get(const std::string &name) {
    auto itOrig = axisRootName.find(name);
    std::string origName = itOrig == axisRootName.end() ? name : itOrig->second;
    auto it = axisInfoMap.find(origName);
    if (it == axisInfoMap.end()) {
      return {};
    }
    return it->second;
  }

  std::pair<std::string, int> getMappingResult(const std::string &name) {
    std::pair<std::string, int> defaultConfig = std::make_pair(kGpuSeqCfg, 3);
    for (auto axisInfo : get(name)) {
      if (axisInfo.name != name || axisInfo.mapLevel == kGpuSeqCfg) {
        continue;
      }
      if (defaultConfig.first != kGpuSeqCfg && axisInfo.mapLevel != defaultConfig.first) {
        llvm::errs() << "Mapping result conflict: " << axisInfo.mapLevel << " vs " << defaultConfig.first
                     << ", map to Seq.\n";
        return std::make_pair(kGpuSeqCfg, kGpuSeqMapDim);
      }
      defaultConfig.first = axisInfo.mapLevel;
      defaultConfig.second = std::min<int>(defaultConfig.second, axisInfo.mapDim);
    }
    return defaultConfig;
  }

  void dump() {
    for (auto it : axisInfoMap) {
      llvm::outs() << "MapAxis " << it.first << "\n";
      for (auto axisInfo : it.second) {
        llvm::outs() << "    " << axisInfo.toString() << "\n";
      }
    }
    llvm::outs() << "loopStructure:[";
    for (auto loopName : loopStructure) {
      llvm::outs() << loopName << "->";
    }
    llvm::outs() << "]\n";
  }

  void reset() {
    axisInfoMap.clear();
    loopStructure.clear();
    axisNameMap.clear();
  }

  std::string getNameAt(size_t idx) {
    if (idx >= loopStructure.size()) {
      return kPlaceHolder;
    }
    return loopStructure[idx];
  }

  // update order from (seq, block, thread) to (block, thread, seq)
  void setReductionOrder() {
    updatedOrder.clear();
    for (size_t i = 0; i < loopStructure.size(); i++) {
      updatedOrder.push_back(i);
    }
    // reorder the second & third axis for each loop
    size_t num = loopStructure.size() / 3;
    for (size_t i = 0; i < num; i++) {
      size_t secondIdx = num + i;
      size_t thirdIdx = num * 2 + i;
      updatedOrder[secondIdx] = static_cast<int>(thirdIdx);
      updatedOrder[thirdIdx] = static_cast<int>(secondIdx);
    }
  }
  std::vector<int> getUpdatedOrder() { return updatedOrder; }

  void setReduceDirection(size_t x) { reduceDirection = x; }
  size_t getReduceDirection() const { return reduceDirection; }
  std::map<std::string, int64_t> dynTilingPrimes;
  std::map<int64_t, size_t> dynTilingPrimesArgIndex;

  // Some global attributes for gpu schedule
  bool enableVectorize{false};
  int64_t vectorSize = 4;
  int64_t minBlockSizesForVectorized = 256;

  // dynamic tiling
  RuntimeVar addRuntimeArgument(int64_t prime) {
    auto var = RuntimeVar();
    var.prime = prime;
    runtimeVars[prime] = var;
    return var;
  }

  RuntimeVar getRuntimeArgument(int64_t prime) {
    assert(runtimeVars.find(prime) != runtimeVars.end());
    return runtimeVars[prime];
  }

  void updateRuntimeArgument(const RuntimeVar &newRuntimeVar) {
    assert(runtimeVars.find(newRuntimeVar.prime) != runtimeVars.end());
    runtimeVars[newRuntimeVar.prime] = newRuntimeVar;
  }

  size_t runtimeArgSize() const { return runtimeVars.size(); }

  bool isRuntimeVar(int64_t prime) { return runtimeVars.find(prime) != runtimeVars.end(); }

  std::map<int64_t, RuntimeVar> getRuntimeVars() { return runtimeVars; }
  std::map<std::string, std::vector<AxisInfo>> getAxisInfoMap() { return axisInfoMap; }
  std::map<std::string, std::string> getAxisRootName() { return axisRootName; }
  void setAxisRootName(const std::map<std::string, std::string> &rootName) { axisRootName = rootName; }
  std::string dynAlgorithm{"unknown"};
  int reduceSizeStatic{-1};
  int parallelSizeStatic{-1};
  bool enableAtomic{false};

 private:
  GpuScheduleTool() {}
  std::map<std::string, std::vector<AxisInfo>> axisInfoMap;
  std::vector<std::string> loopStructure;
  std::map<std::string, std::vector<std::string>> axisNameMap;
  std::vector<int> updatedOrder;  // used for coalescing access in affine-loop-reorder pass
  std::map<std::string, std::string> axisRootName;
  size_t reduceDirection = 0;  // { UNKNOWN = 0, X, Y, ALL }
  std::map<int64_t, RuntimeVar> runtimeVars;
  bool isCustomConfig{false};
};

class PrimeNumTool {
 public:
  ~PrimeNumTool() {}
  PrimeNumTool(const PrimeNumTool &) = default;
  PrimeNumTool &operator=(const PrimeNumTool &) = delete;
  static PrimeNumTool &getInstance() {
    static PrimeNumTool instance;
    return instance;
  }

  size_t getOnePrime(size_t idx) const {
    if (idx < kPrimeSize) {
      return primeList[idx];
    }
    llvm::errs() << "Idx " << idx << " exceed of prime list size " << kPrimeSize << ", return a first prime\n";
    return primeList[0];
  }
  size_t getOnePrimeWithIdxUpdate() {
    auto value = primeList[currentIdx++];
    while (visited.find(value) != visited.end()) {
      value = primeList[currentIdx++];
    }
    return value;
  }

  void updateVisited(int num) {
    // to avoid prime match in .mlir, we add `+-saveBound` in visited set
    for (int v = std::max(1, num - saveBound); v < num + saveBound + 1; v++) {
      (void)visited.insert(v);
    }
  }

 private:
  PrimeNumTool() {}
  size_t currentIdx = 0;
  int saveBound = 2;
  // Use externally generated primeList (C++17 compatible)
  static constexpr auto &primeList = prime_helpers::primeList;
  std::set<int> visited;
};

using ShapeInfo = std::vector<std::string>;
class ShapeAlignTool {
 public:
  ~ShapeAlignTool() {}
  ShapeAlignTool(const ShapeAlignTool &) = default;
  ShapeAlignTool &operator=(const ShapeAlignTool &) = delete;
  static ShapeAlignTool &getInstance() {
    static ShapeAlignTool instance;
    return instance;
  }

  void reset() {
    hostShapes.clear();
    deviceShapes.clear();
  }

  void setHostShapes(const std::map<size_t, ShapeInfo> &shapes) { hostShapes = shapes; }

  void setDeviceShapes(const std::map<size_t, ShapeInfo> &shapes) { deviceShapes = shapes; }

  void setOutputIndices(const std::unordered_set<size_t> &outId) { outputIndices = outId; }

  void initHost(const std::map<size_t, ShapeInfo> &init, const std::unordered_set<size_t> &outId) {
    setHostShapes(init);
    setDeviceShapes(init);
    setOutputIndices(outId);
  }

  size_t getFuncArgSizes() const { return hostShapes.size(); }
  std::unordered_set<size_t> getOutputIndices() { return outputIndices; }
  bool isOutput(size_t i) { return outputIndices.count(i) != 0; }
  void updateCurrShapeInfo(const std::map<size_t, ShapeInfo> &currShapes) { deviceShapes = currShapes; }
  void updateCurrShapeInfo(size_t i, const ShapeInfo &currShape) { deviceShapes[i] = currShape; }

  // Make sure each time we update the current shape info
  // so we can assume the deviceShapes is the current shape.
  ShapeInfo getCurrShapeInfo(size_t i) {
    auto it = deviceShapes.find(i);
    if (it == deviceShapes.end()) {
      return ShapeInfo();
    }
    return it->second;
  }

  // Convert map to vector for json dump
  std::vector<ShapeInfo> getHostShapesList() {
    std::vector<ShapeInfo> res;
    for (auto it : hostShapes) {
      res.push_back(it.second);
    }
    return res;
  }

  std::vector<ShapeInfo> getDeviceShapesList() {
    std::vector<ShapeInfo> res;
    for (auto it : deviceShapes) {
      res.push_back(it.second);
    }
    return res;
  }

  // Convert symbolic shapes to const shapes and replace dynamic shape with -1
  void convertToConstShapes(const mlir::ShapedType &shapedType, mlir::SmallVector<int64_t> &constShapes) const {
    auto shapes = shapedType.getShape();
    for (size_t i = 0; i < shapes.size(); ++i) {
      if (shapedType.isDynamicDim(static_cast<int>(i))) {
        constShapes.push_back((int64_t)-1);
      } else {
        constShapes.push_back(shapes[i]);
      }
    }
  }

  // When we align input shapes, we start from input args and end at destOp
  // and the destOp should be the user of input args.
  // Each time we search for ops that need to be aligned and updates
  // the originShapes directly.
  void alignInputShape(mlir::Operation *op, const mlir::Value &destOp, ShapeInfo &originShapes) {
    search(op, originShapes);
    if (reachDest(op, destOp)) {
      return;
    }
    // inputs may be used in alloc (e.g. through memref.Dim) but the users of allocOp
    // should not be involved in input alignment.
    if (mlir::isa<mlir::memref::AllocOp>(op)) {
      return;
    }
    for (auto user : op->getUsers()) {
      alignInputShape(user, destOp, originShapes);
    }
  }

  // Similar to alignInputShape, but for output shapes, we start from output args and end
  // at destOp and the destOp should be the producer of output args.
  void alignOutputShape(mlir::Operation *op, const mlir::Value &destOp, ShapeInfo &originShapes,
                        mlir::Operation *funcOp) {
    search(op, originShapes);
    if (reachDest(op, destOp)) {
      return;
    }

    mlir::SmallVector<mlir::Operation *> parents;
    funcOp->walk([&](mlir::Operation *p) {
      for (auto user : p->getUsers()) {
        if (user == op) {
          parents.push_back(p);
          break;
        }
      }
    });
    for (auto p : parents) {
      alignOutputShape(p, destOp, originShapes, funcOp);
    }
  }

  // When we only reconstruct the static shapes, we can use this function to align shapes.
  // It works when the dynamic shapes' size and their relative positions remain unchanged.
  // For example, it is safe when <1, S0, 3, 4, S1> is reconstruct to any of the following forms:
  //   1. <1, S0, 12, S1> (fold-dim)
  //   2. <S0, 3, 4, S1>  (elim reshape)
  //   3. <1, S0, 3, 4, S1, 1> (expand dim)
  // And if S0 and S1 are reconstruct to a new shape like S2;
  // or they are transposed (e.g. <1, S1, 3, 4, S0>);
  // we cannot use this function to update.
  void alignStaticShapeReconstruct(size_t argIdx, const mlir::Type &oldType, const mlir::Type &newType) {
    if (getFuncArgSizes() == 0 || oldType == newType) {
      return;
    }
    auto oldShape = mlir::cast<mlir::ShapedType>(oldType);
    assert(oldShape && "Old Type should be a ShapedType");
    auto newShape = mlir::cast<mlir::ShapedType>(newType);
    assert(newShape && "New Type should be a ShapedType");
    ShapeInfo oldDeviceShape = getCurrShapeInfo(argIdx);
    mlir::SmallVector<int64_t> oldShapeList;
    convertToConstShapes(oldShape, oldShapeList);
    assert(oldShapeList.size() == oldDeviceShape.size() &&
           "Old shapes sizes should equal to current device shape record.");
    // get the dynamic symbols in order
    std::deque<std::pair<std::string, size_t>> dynamicSymbols;
    auto originalNeedFix = getNeedFixIndice(argIdx);
    for (size_t i = 0; i < oldShapeList.size(); ++i) {
      if (oldShapeList[i] == -1) {
        if (isOutput(argIdx)) {
          dynamicSymbols.push_back(std::make_pair(oldDeviceShape[i], 0));
        } else {
          assert(i < originalNeedFix.size());
          dynamicSymbols.push_back(std::make_pair(oldDeviceShape[i], originalNeedFix[i]));
        }
      }
    }

    ShapeInfo currShape;
    mlir::SmallVector<int64_t> newShapeList;
    convertToConstShapes(newShape, newShapeList);
    mlir::SmallVector<int64_t> updatedFix;
    for (size_t i = 0; i < newShapeList.size(); ++i) {
      if (newShapeList[i] == -1) {
        // pad with the original dynamic symbols
        assert(!dynamicSymbols.empty() && "Old shape and new shape has different number of dynamic symbols");
        auto pairs = dynamicSymbols.front();
        currShape.push_back(pairs.first);
        (void)updatedFix.push_back((int64_t)pairs.second);
        dynamicSymbols.pop_front();
      } else {
        // record the new static shapes
        currShape.push_back(std::to_string(newShapeList[i]));
        updatedFix.push_back(0);
      }
    }
    recordNeedFixIndice(argIdx, updatedFix);
    updateCurrShapeInfo(argIdx, currShape);
  }

  void recordNeedFixIndice(size_t argIdx, mlir::SmallVector<int64_t> indice) { needFixMap[argIdx] = indice; }

  mlir::SmallVector<int64_t> getNeedFixIndice(size_t argIdx) {
    auto it = needFixMap.find(argIdx);
    if (it == needFixMap.end()) {
      return {};
    }
    return it->second;
  }

  bool isDynamicShape() const { return getFuncArgSizes() > 0; }

 private:
  ShapeAlignTool() {}

  void doAlign(mlir::SmallVector<mlir::ReassociationIndices, 4> reassociation, ShapeInfo &originShapes,
               mlir::SmallVector<int64_t> &destShapes) const {
    ShapeInfo updatedShapes;
    size_t reIdx = 0;
    for (auto newIndex : reassociation) {
      auto srcShape = originShapes[reIdx];
      assert(reIdx < originShapes.size());

      // init the newSymbol as original shape, e.g. "S0"
      std::string newSymbol(srcShape);
      for (size_t vi : newIndex) {
        vi = static_cast<unsigned>(vi);
        assert(vi < destShapes.size());
        // destShapes[vi] != -1 means this dim has static shape
        // so we can divide it from the original symbolic shape
        if (destShapes[vi] != (int64_t)-1) {
          newSymbol += "/" + std::to_string(destShapes[vi]);
        }
      }
      // So until here we can have a newSymbol that is the original
      // shape divides all static shapes from dest, e.g. "S0/2/4"

      // used to simpify the expr, e.g. "S0/2/4" -> "S0/8"
      SymEngine::Expression expr(newSymbol);
      mlir::SymbolicShapeAnalysis &analysis = mlir::SymbolicShapeAnalysis::getInstance();
      newSymbol = analysis.getSymbolicDimFromExpression(expr);

      for (size_t vi : newIndex) {
        if (destShapes[vi] != (int64_t)-1) {
          // place all static shape here so in the end we multiply updatedShapes
          // can get back to the original shape, e.g. [2, 4]
          updatedShapes.push_back(std::to_string(destShapes[vi]));
        } else {
          // place new symbol here, e.g. [2, 4, "S0/8"]
          updatedShapes.push_back(newSymbol);
        }
      }
      ++reIdx;
    }
    originShapes = updatedShapes;
  }

  // Currently we support two types of alignment: ExpandShapeOp and CollapseShapeOp.
  // And we do align based on the reassociation indices.
  void search(mlir::Operation *op, ShapeInfo &originShapes) const {
    if (auto expandShape = mlir::dyn_cast<mlir::memref::ExpandShapeOp>(op)) {
      auto shapedType = mlir::cast<mlir::ShapedType>(expandShape.getResultType());
      if (!shapedType) {
        (void)op->emitError("Op is not shapedType, cannot align shape.");
        return;
      }
      mlir::SmallVector<int64_t> destShapes;
      convertToConstShapes(shapedType, destShapes);
      doAlign(expandShape.getReassociationIndices(), originShapes, destShapes);
    } else if (auto collapse = mlir::dyn_cast<mlir::memref::CollapseShapeOp>(op)) {
      auto shapedType = mlir::cast<mlir::ShapedType>(collapse.getSrcType());
      if (!shapedType) {
        (void)op->emitError("Op is not shapedType, cannot align shape.");
        return;
      }
      mlir::SmallVector<int64_t> destShapes;
      convertToConstShapes(shapedType, destShapes);
      doAlign(collapse.getReassociationIndices(), originShapes, destShapes);
    }
  }

  bool reachDest(const mlir::Operation *currOp, const mlir::Value &destOp) const {
    return destOp && destOp.getDefiningOp() && currOp && destOp.getDefiningOp() == currOp;
  }
  std::map<size_t, ShapeInfo> hostShapes;
  std::map<size_t, ShapeInfo> deviceShapes;
  // Records the indices of output tensors in func arguments because in AKG we may
  // invoke `createBufferResultsToOutParamsPass` so that inputs and outputs are mixed.
  std::unordered_set<size_t> outputIndices;
  // Record the indice of func args that need to be fixed due to implicit broadcast.
  std::map<size_t, mlir::SmallVector<int64_t>> needFixMap;
};
}  // namespace akgglobal
#endif  // AKG_UTILS_GLOBALVARS_H_
