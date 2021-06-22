/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "composite/optimize/optimize.h"

namespace akg {

class AffinityAnalyzer : public IRVisitor {
 public:
  enum {
    AF_NONE = 0,
    AF_ELEMWISE,
    AF_BROADCAST,
    AF_REDUCE,
    AF_RESHAPE,    
  };

  struct Tensor;
  struct Dim {
    Tensor *tensor{nullptr};
    int index{0};
    int64_t size{0};
    std::vector<std::pair<Dim*, int>> prod;
    std::vector<std::pair<Dim*, int>> cons;
  };
  struct Tensor {
    FunctionRef ref; 
    std::string op;
    std::vector<Tensor*> prod;
    std::vector<Tensor*> cons;
    std::vector<std::unique_ptr<Dim>> dims;
  };

  AffinityAnalyzer() = default;
  ~AffinityAnalyzer() = default;

  void Analyze(Stmt stmt);

  void VisitProd(Dim *dim, std::function<bool(Dim*, Dim*, int)> fun);
  void VisitCons(Dim *dim, std::function<bool(Dim*, Dim*, int)> fun);

  void Dump(std::ostringstream &os);

  void Visit_(const AttrStmt *op) override;
  void Visit_(const Provide *op) override;

  std::vector<std::unique_ptr<Tensor>> tensors_;

 private:
  std::unordered_map<FunctionRef, Tensor*, NodeHash, NodeEqual> tensor_map_;
  Map<std::string, NodeRef> attrs_;

  void AddElemBroadRelation(Tensor *input, Tensor *output);
  void AddReduceRelation(Tensor *input, Tensor *output);
  void AddTransposeRelation(Tensor *input, Tensor *output);

  std::vector<int64_t> ExtractIntVector(Array<Expr> &vec);
  Tensor *NewTensor(FunctionRef ref, std::string op, Array<Expr> shape);
};

class DimensionPeeler {
 public:
  using Peeling = std::vector<std::pair<int, int64_t>>; // dim, split_val
  using Tensor = AffinityAnalyzer::Tensor;
  using Dim = AffinityAnalyzer::Dim;

  DimensionPeeler() = default;
  ~DimensionPeeler() = default;

  void Analyze(Stmt s);
  std::vector<int64_t> GetAxisSpace();
  std::vector<Peeling> GetPeelSpace(int limit_depth = 0, std::unordered_set<int> *limit_range = nullptr);
  Stmt GetPeelBody(const Peeling &peeling);
  std::vector<int> GetPeelDims(FunctionRef tensor, const Peeling &peeling);

 private:
  struct Axis {
    int size{0};
    std::vector<int64_t> peel_val;
  };

  Stmt stmt_;
  std::vector<std::unique_ptr<Axis>> axis_space_;
  // axis_idx -> dim_idx
  std::unordered_map<FunctionRef, std::vector<int>, NodeHash, NodeEqual> dim_map_;  
  
  std::vector<int64_t> GetDivisors(int64_t n);
  Tensor* BuildAxisSpace(AffinityAnalyzer &aff);
  void MapDimToSpace(AffinityAnalyzer &aff, Dim *dom_dim, int axis_idx);
  bool Propagation(int axis_idx, Dim *from, Dim *to, int affinity);
  void AddDimMap(Dim *dim, int axis_idx);
};

///////////////////////////////////////////////////////////////////////////////
// TODO: The following is test code. should remove later
///////////////////////////////////////////////////////////////////////////////
class DumpPeelDims: public IRVisitor {
 public:
  DumpPeelDims(DimensionPeeler &peeler, DimensionPeeler::Peeling &peeling) 
  : peeler_(peeler),  peeling_(peeling) {
  }
  void Visit(const NodeRef &node) override {
    const Provide *op = node.as<Provide>();
    if (op != nullptr) {
      std::cout<<"//AxisIdx-DimIdx: ";
      PrintPeeling(op->func);
      std::cout<<"=(";
      auto prim = op->value.as<Call>();
      for (size_t i = 0; i < prim->args.size(); ++i) {
        auto t = prim->args[i].as<Call>();
        if (t != nullptr) {
          PrintPeeling(t->func);
        } else {
          std::cout<<prim->args[i];
        }
        if (i < prim->args.size() - 1) std::cout<<",";
      }
      std::cout<<")"<<std::endl;
      std::cout<<node;
    }
    IRVisitor::Visit(node);
  }
  void PrintPeeling(FunctionRef ref) {
    auto dims = peeler_.GetPeelDims(ref, peeling_);
    std::cout<<"[";
    for (size_t i = 0; i < dims.size(); ++i) {
      std::cout<<dims[i];
      if (i < dims.size() - 1) std::cout<<",";
    }
    std::cout<<"]";
  }
  DimensionPeeler &peeler_;
  DimensionPeeler::Peeling &peeling_;
};

class PeelDimensionTester : public CompositeOptPass {
 public:
  PeelDimensionTester() { pass_name_ = __FUNCTION__; }
  ~PeelDimensionTester() = default;
  Stmt Run(const Stmt &s) {
    const char* test_idx = getenv("PEEL_IDX");
    if (test_idx == nullptr) {
      return s;
    }
    auto PrintPeeling = [](const DimensionPeeler::Peeling &peeling) {
      std::cout<<"{";
      for (size_t i = 0; i < peeling.size(); ++i) {
        std::cout<<peeling[i].first<<":"<<peeling[i].second;
        if (i < peeling.size() - 1) {
          std::cout<<", ";
        }
      }
      std::cout<<"}";
    };
    DimensionPeeler peeler;
    peeler.Analyze(s);
    auto axis_space = peeler.GetAxisSpace();
    std::cout<<"axis_space : [";
    for (auto v : axis_space) {
      std::cout<<v<<", ";
    }
    std::cout<<"]\n";
    auto peeling_space = peeler.GetPeelSpace();
    std::cout<<"peeling_space size = "<<peeling_space.size()<<std::endl;
    for (size_t i = 0; i < peeling_space.size(); ++i) {
      std::cout<<i<<": ";
      PrintPeeling(peeling_space[i]);
      std::cout<<std::endl;
    }
    int idx = std::stoi(std::string(test_idx));
    Stmt body = peeler.GetPeelBody(peeling_space[idx]);
    std::cout<<"*********** input stmt *************\n";
    std::cout<<s;
    std::cout<<"*********** peel dim: ";
    PrintPeeling(peeling_space[idx]);
    std::cout<<" *************\n";
    DumpPeelDims(peeler, peeling_space[idx]).Visit(s);
    std::cout<<"*********** peel body: ";
    PrintPeeling(peeling_space[idx]);
    std::cout<<" ***********"<<std::endl;
    std::cout<<body;
    exit(0);
    return s;
  }
};

}  // namespace akg

