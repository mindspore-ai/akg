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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_MODEL_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_MODEL_H_
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "akg/Dialect/Affine/Analysis/Axis.h"
#include "akg/Dialect/Affine/Analysis/Config.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace akg {
namespace autotiling {
class Op {
 public:
  enum class OpCategory { Injective, Broadcast, Reshape, Transpose, ReduceX, ReduceY, AllReduce, MatMul, Unknown };

  explicit Op(const std::string &op_name, OpCategory op_category = OpCategory::Unknown)
      : name(op_name), opCategory(op_category) {}

  std::string name;
  OpCategory opCategory;
};

enum GraphTemplate { DEFAULT = 0, CONV, MATMUL, REDUCTION, BROADCAST_OP, TRANSPOSE_OP, PURE_ELEM, TEMPLATE_BULK };

class Tensor {
 public:
  Tensor(mlir::Operation *op, const std::vector<AxisPtr> &loopNest);

  template <typename T>
  void SetLoadTensor(T loadOp, const std::vector<AxisPtr> &loopNest);

  template <typename T>
  void SetStoreTensor(T StoreOp, const std::vector<AxisPtr> &loopNest);
  std::vector<AxisPtr> loopNest() { return loopNest_; }
  std::string toString() {
    std::stringstream ss;
    ss << "Tensor name : " << name << " op type: " << opType << " loop depth: " << loopNest_.size() << "\n";
    return ss.str();
  }

  mlir::Operation *op() { return op_; }

  std::string name{"default-name"};
  std::string opType{"unknown-op"};
  Type dataType{nullptr};
  mlir::Operation *op_{nullptr};
  std::vector<AxisPtr> loopNest_;
};

class Node : public Tensor {
 public:
  Node(mlir::Operation *op, const std::vector<AxisPtr> &loopNest) : Tensor(op, loopNest) {}
};
using NodePtr = std::shared_ptr<Node>;

class InitGraph {
 public:
  InitGraph(const std::string &, const std::vector<NodePtr> &, const std::vector<NodePtr> &,
            const std::vector<NodePtr> &);
  explicit InitGraph(const std::string &);
  InitGraph() = default;
  virtual ~InitGraph() = default;

  virtual void dump();

  void setGraphType(const StringAttr &attrs);
  void setHardware(const std::string &hw);
  void setFeature(const std::string &fea);
  void setIsDynamicShape(bool isDyn);
  void setTilingMode(const std::string &tm);

  void drawNode(const NodePtr node) { nodes_.push_back(node); }
  void drawInputs(const NodePtr node) { inputs_.push_back(node); }
  void drawOutputs(const NodePtr node) { outputs_.push_back(node); }

  std::vector<NodePtr> nodes() { return nodes_; }
  NodePtr getMaxRankTensor();
  void updateMaxRankTensor(const NodePtr &node) { maxRankNode = node; }
  std::string name;
  std::vector<NodePtr> nodes_;
  std::vector<NodePtr> inputs_;
  std::vector<NodePtr> outputs_;
  AxisPtr rootAxis{nullptr};
  std::string graphType;
  std::string hardware;
  std::string feature;
  std::string tilingMode{"auto"};
  Operation *funcOp{nullptr};
  bool isDynamicShape{false};

 private:
  NodePtr maxRankNode = nullptr;
};
using InitGraphPtr = std::shared_ptr<InitGraph>;

class ModelGraph : public InitGraph {
 public:
  explicit ModelGraph(const InitGraphPtr &initGraph);
  virtual ~ModelGraph() = default;

  void AnalyzeGraphTemplate();
  std::string ShowGraphTemplate() { return templateMap[static_cast<size_t>(graphTemplate)]; }
  std::vector<int> getLoopExtentsAfterTiling(const AxisPtr axis) const;

  size_t levelToTile{1};  // number of tiling level
  bool hasMinMax{false};  // that comes from indivisible tile sizes
  GraphTemplate graphTemplate{GraphTemplate::DEFAULT};
  std::unordered_map<int, std::string> templateMap = {{0, "DEFAULT"},   {1, "CONV"},         {2, "MATMUL"},
                                                      {3, "REDUCTION"}, {4, "BROADCAST_OP"}, {5, "TRANSPOSE_OP"},
                                                      {6, "PURE_ELEM"}};
  std::map<std::string, Attribute> globalConfigs;
  std::deque<AxisPtr> sortedAxes;

 private:
  GraphTemplate AnalyzeTransposeGraph();
};
using ModelGraphPtr = std::shared_ptr<ModelGraph>;

struct Resource {
  std::vector<int64_t> availbleSize;
  std::map<AxisPtr, int64_t> allocSize;
  std::string resourceType;
  int64_t currSize{1};
  int64_t totalAvailableSize;  // only for gpu block
  int64_t rest();
  bool canApply(int64_t size);
  ConfigPtr alloc(const AxisPtr axis, int64_t size);
  bool seen(const AxisPtr axis) { return allocSize.find(axis) != allocSize.end(); }
  int64_t get(const AxisPtr axis) {
    if (!seen(axis)) {
      return 1;
    }
    return allocSize[axis];
  }
  size_t currAllocDim() const { return allocSize.size(); }
};

class GpuModelGraph : public ModelGraph {
 public:
  explicit GpuModelGraph(const InitGraphPtr &initGraph);
  virtual ~GpuModelGraph() = default;

  void InitResource();
  int problemSize = 1;
  Resource gpuGrid;
  Resource gpuBlock;
};
using GpuModelGraphPtr = std::shared_ptr<GpuModelGraph>;

class CpuModelGraph : public ModelGraph {
 public:
  explicit CpuModelGraph(const InitGraphPtr &initGraph);
  virtual ~CpuModelGraph() = default;

  int64_t l1Cache = 8388608;  // 8MB
  int64_t byteUnit = 1024;
  int64_t bitUnit = 8;
  int tileNum = 0;
};
using CpuModelGraphPtr = std::shared_ptr<CpuModelGraph>;
}  // namespace autotiling
}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_MODEL_H_

