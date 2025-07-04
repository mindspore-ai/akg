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
#ifndef COMPILER_INCLUDE_AKG_UTILS_AKGGLOBALVARS_H_
#define COMPILER_INCLUDE_AKG_UTILS_AKGGLOBALVARS_H_

#include <deque>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace akgglobal {
constexpr auto kNeedFix = "need_fix";
constexpr auto kLoopTag = "loop_tag";
constexpr auto kPlaceHolder = "place_holder";
constexpr auto kTileCfg = "Tile";
constexpr auto kGpuSeqCfg = "GpuSeq";
constexpr auto kGpuGridCfg = "GpuGrid";
constexpr auto kGpuBlockCfg = "GpuBlock";
const static int kGpuSeqMapDim = 3;

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

  void tagLoopWithAxisName(Operation *funcOp) {
    size_t i = 0;
    size_t loopSize = this->loopSize();
    OpBuilder builder(funcOp);
    funcOp->walk([&](Operation *op) {
      if (!isa<affine::AffineForOp, affine::AffineParallelOp, scf::ParallelOp>(op)) {
        return;
      }
      auto tagName = getNameAt(loopSize - 1 - i);
      if (tagName == kPlaceHolder) {
        return;
      }
      Attribute attr = builder.getStringAttr(tagName);
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
  std::vector<int> updatedOrder;  // used for coalescing acceess in affine-loop-reorder pass
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
    if (idx < size) {
      return primeList[idx];
    }
    llvm::errs() << "Idx " << idx << " exceed of prime list size " << size << ", return a first prime\n";
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
  static constexpr size_t size = 300;
  static constexpr size_t primeList[size] = {
    40009, 40013, 40031, 40037, 40039, 40063, 40087, 40093, 40099, 40111, 40123, 40127, 40129, 40151, 40153, 40163,
    40169, 40177, 40189, 40193, 40213, 40231, 40237, 40241, 40253, 40277, 40283, 40289, 40343, 40351, 40357, 40361,
    40387, 40423, 40427, 40429, 40433, 40459, 40471, 40483, 40487, 40493, 40499, 40507, 40519, 40529, 40531, 40543,
    40559, 40577, 40583, 40591, 40597, 40609, 40627, 40637, 40639, 40693, 40697, 40699, 40709, 40739, 40751, 40759,
    40763, 40771, 40787, 40801, 40813, 40819, 40823, 40829, 40841, 40847, 40849, 40853, 40867, 40879, 40883, 40897,
    40903, 40927, 40933, 40939, 40949, 40961, 40973, 40993, 41011, 41017, 41023, 41039, 41047, 41051, 41057, 41077,
    41081, 41113, 41117, 41131, 41141, 41143, 41149, 41161, 41177, 41179, 41183, 41189, 41201, 41203, 41213, 41221,
    41227, 41231, 41233, 41243, 41257, 41263, 41269, 41281, 41299, 41333, 41341, 41351, 41357, 41381, 41387, 41389,
    41399, 41411, 41413, 41443, 41453, 41467, 41479, 41491, 41507, 41513, 41519, 41521, 41539, 41543, 41549, 41579,
    41593, 41597, 41603, 41609, 41611, 41617, 41621, 41627, 41641, 41647, 41651, 41659, 41669, 41681, 41687, 41719,
    41729, 41737, 41759, 41761, 41771, 41777, 41801, 41809, 41813, 41843, 41849, 41851, 41863, 41879, 41887, 41893,
    41897, 41903, 41911, 41927, 41941, 41947, 41953, 41957, 41959, 41969, 41981, 41983, 41999, 42013, 42017, 42019,
    42023, 42043, 42061, 42071, 42073, 42083, 42089, 42101, 42131, 42139, 42157, 42169, 42179, 42181, 42187, 42193,
    42197, 42209, 42221, 42223, 42227, 42239, 42257, 42281, 42283, 42293, 42299, 42307, 42323, 42331, 42337, 42349,
    42359, 42373, 42379, 42391, 42397, 42403, 42407, 42409, 42433, 42437, 42443, 42451, 42457, 42461, 42463, 42467,
    42473, 42487, 42491, 42499, 42509, 42533, 42557, 42569, 42571, 42577, 42589, 42611, 42641, 42643, 42649, 42667,
    42677, 42683, 42689, 42697, 42701, 42703, 42709, 42719, 42727, 42737, 42743, 42751, 42767, 42773, 42787, 42793,
    42797, 42821, 42829, 42839, 42841, 42853, 42859, 42863, 42899, 42901, 42923, 42929, 42937, 42943, 42953, 42961,
    42967, 42979, 42989, 43003, 43013, 43019, 43037, 43049, 43051, 43063, 43067, 43093};
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
  void convertToConstShapes(const ShapedType &shapedType, SmallVector<int64_t> &constShapes) const {
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
  void alignInputShape(Operation *op, const Value &destOp, ShapeInfo &originShapes) {
    search(op, originShapes);
    if (reachDest(op, destOp)) {
      return;
    }
    // inputs may be used in alloc (e.g. through memref.Dim) but the users of allocOp
    // should not be involved in input alignment.
    if (isa<memref::AllocOp>(op)) {
      return;
    }
    for (auto user : op->getUsers()) {
      alignInputShape(user, destOp, originShapes);
    }
  }

  // Similar to alignInputShape, but for output shapes, we start from output args and end
  // at destOp and the destOp should be the producer of output args.
  void alignOutputShape(Operation *op, const Value &destOp, ShapeInfo &originShapes, Operation *funcOp) {
    search(op, originShapes);
    if (reachDest(op, destOp)) {
      return;
    }

    SmallVector<Operation *> parents;
    funcOp->walk([&](Operation *p) {
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
  void alignStaticShapeReconstruct(size_t argIdx, const Type &oldType, const Type &newType) {
    if (getFuncArgSizes() == 0 || oldType == newType) {
      return;
    }
    auto oldShape = oldType.cast<ShapedType>();
    assert(oldShape && "Old Type should be a ShapedType");
    auto newShape = newType.cast<ShapedType>();
    assert(newShape && "New Type should be a ShapedType");
    ShapeInfo oldDeviceShape = getCurrShapeInfo(argIdx);
    SmallVector<int64_t> oldShapeList;
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
    SmallVector<int64_t> newShapeList;
    convertToConstShapes(newShape, newShapeList);
    SmallVector<int64_t> updatedFix;
    for (size_t i = 0; i < newShapeList.size(); ++i) {
      if (newShapeList[i] == -1) {
        // pad with the original dynamic symbols
        assert(!dynamicSymbols.empty() && "Old shape and new shape has differnt number of dynamic symbols");
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

  void recordNeedFixIndice(size_t argIdx, SmallVector<int64_t> indice) { needFixMap[argIdx] = indice; }

  SmallVector<int64_t> getNeedFixIndice(size_t argIdx) {
    auto it = needFixMap.find(argIdx);
    if (it == needFixMap.end()) {
      return {};
    }
    return it->second;
  }

  bool isDynamicShape() const { return getFuncArgSizes() > 0; }

 private:
  ShapeAlignTool() {}

  void doAlign(SmallVector<ReassociationIndices, 4> reassociation, ShapeInfo &originShapes,
               SmallVector<int64_t> &destShapes) const {
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
      SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
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
  void search(Operation *op, ShapeInfo &originShapes) const {
    if (auto expandShape = dyn_cast<memref::ExpandShapeOp>(op)) {
      auto shapedType = expandShape.getResultType().cast<ShapedType>();
      if (!shapedType) {
        (void)op->emitError("Op is not shapedType, cannot align shape.");
        return;
      }
      SmallVector<int64_t> destShapes;
      convertToConstShapes(shapedType, destShapes);
      doAlign(expandShape.getReassociationIndices(), originShapes, destShapes);
    } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(op)) {
      auto shapedType = collapse.getSrcType().cast<ShapedType>();
      if (!shapedType) {
        (void)op->emitError("Op is not shapedType, cannot align shape.");
        return;
      }
      SmallVector<int64_t> destShapes;
      convertToConstShapes(shapedType, destShapes);
      doAlign(collapse.getReassociationIndices(), originShapes, destShapes);
    }
  }

  bool reachDest(const Operation *currOp, const Value &destOp) const {
    return destOp && destOp.getDefiningOp() && currOp && destOp.getDefiningOp() == currOp;
  }

  std::map<size_t, ShapeInfo> hostShapes;
  std::map<size_t, ShapeInfo> deviceShapes;
  // Records the indices of output tensors in func arguments because in AKG we may
  // invoke `createBufferResultsToOutParamsPass` so that inputs and outputs are mixed.
  std::unordered_set<size_t> outputIndices;
  // Record the indice of func args that need to be fixed due to implicit broadcast.
  std::map<size_t, SmallVector<int64_t>> needFixMap;
};
}  // namespace akgglobal
#endif  // COMPILER_INCLUDE_AKG_UTILS_AKGGLOBALVARS_H_