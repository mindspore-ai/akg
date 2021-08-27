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

#include "composite/dimension_peeling.h"

#define PEEL_DUMP 0

namespace akg {

void AffinityAnalyzer::Analyze(Stmt stmt) { IRVisitor::Visit(stmt); }

void AffinityAnalyzer::VisitProd(Dim *dim, std::function<bool(Dim *, Dim *, int)> fun) {
  std::vector<Dim *> stack;
  stack.push_back(dim);
  while (!stack.empty()) {
    auto d = stack.back();
    stack.pop_back();
    for (auto &prod : d->prod) {
      if (fun(d, prod.first, prod.second)) {
        stack.emplace_back(prod.first);
      }
    }
  }
}

void AffinityAnalyzer::VisitCons(Dim *dim, std::function<bool(Dim *, Dim *, int)> fun) {
  std::vector<Dim *> stack;
  stack.push_back(dim);
  while (!stack.empty()) {
    auto d = stack.back();
    stack.pop_back();
    for (auto &cons : d->cons) {
      if (fun(d, cons.first, cons.second)) {
        stack.emplace_back(cons.first);
      }
    }
  }
}

void AffinityAnalyzer::Dump(std::ostringstream &os) {
  std::vector<std::string> aff_str = {"None", "Elem", "Broad", "Red", "Resh"};
  os << "*********** Affinity Dump **************" << std::endl;
  for (auto &t : tensors_) {
    std::unordered_map<Tensor *, int> prod_idx;
    os << t->ref.get() << "=" << t->op << "(";
    for (size_t i = 0; i < t->prod.size(); ++i) {
      os << t->prod[i]->ref.get();
      if (i < t->prod.size() - 1) {
        os << ",";
      }
      prod_idx[t->prod[i]] = i;
    }
    os << ")\n";
    for (size_t i = 0; i < t->dims.size(); ++i) {
      auto dim = t->dims[i].get();
      os << "  " << i << "%%[" << dim->size << "] : ";
      for (size_t j = 0; j < dim->prod.size(); ++j) {
        auto &prod = dim->prod[j];
        os << prod_idx[prod.first->tensor] << "." << prod.first->index << "%%" << aff_str[prod.second];
        if (j < dim->prod.size() - 1) {
          os << ",";
        }
      }
      os << std::endl;
    }
  }
}

void AffinityAnalyzer::Visit_(const AttrStmt *op) {
  if (op->attr_key == "attrs") {
    attrs_ = Downcast<Map<std::string, NodeRef>>(op->node);
  }
  IRVisitor::Visit_(op);
}

void AffinityAnalyzer::Visit_(const Provide *op) {
  auto prim = op->value.as<Call>();
  std::vector<Tensor *> inputs;
  for (Expr arg : prim->args) {
    auto t = arg.as<Call>();
    if (t == nullptr) {
      continue;
    }
    auto it = tensor_map_.find(t->func);
    if (it != tensor_map_.end()) {
      inputs.emplace_back(it->second);
    } else {
      auto input = NewTensor(t->func, "Para", t->args);
      inputs.emplace_back(input);
    }
  }
  Tensor *output = NewTensor(op->func, prim->name, op->args);
  for (auto t : inputs) {
    output->prod.emplace_back(t);
    t->cons.emplace_back(t);
  }
  if (IsElemwise(prim->name) || prim->name == "BroadcastTo") {
    for (Tensor *input : inputs) {
      AddElemBroadRelation(input, output);
    }
  } else if (IsReduce(prim->name)) {
    CHECK(inputs.size() == 1);
    AddReduceRelation(inputs[0], output);
  } else if (prim->name == "InplaceAssign") {
    CHECK(inputs.size() == 3);
    AddElemBroadRelation(inputs[1], inputs[0]);
    AddElemBroadRelation(inputs[2], output);
  } else if (prim->name == "Transpose") {
    AddTransposeRelation(inputs[0], output);
  } else {
    CHECK(0);
  }
}

void AffinityAnalyzer::AddElemBroadRelation(Tensor *input, Tensor *output) {
  CHECK(input->dims.size() <= output->dims.size());
  size_t dim_offset = output->dims.size() - input->dims.size();
  for (size_t i = dim_offset; i < output->dims.size(); ++i) {
    auto prod_dim = input->dims[i - dim_offset].get();
    auto cons_dim = output->dims[i].get();
    int affinity = prod_dim->size == cons_dim->size ? AF_ELEMWISE : AF_BROADCAST;
    cons_dim->prod.emplace_back(std::make_pair(prod_dim, affinity));
    prod_dim->cons.emplace_back(std::make_pair(cons_dim, affinity));
  }
}

void AffinityAnalyzer::AddReduceRelation(Tensor *input, Tensor *output) {
  Array<Expr> axis = Downcast<Array<Expr>>(attrs_["axis"]);
  auto axis_vec = ExtractIntVector(axis);
  bool keep_dim = input->dims.size() == output->dims.size();
  int output_idx = 0;
  for (size_t i = 0; i < input->dims.size(); ++i) {
    bool reduce_mode = std::find(axis_vec.begin(), axis_vec.end(), i) != axis_vec.end();
    auto prod_dim = input->dims[i].get();
    auto cons_dim = output->dims[output_idx].get();
    if (reduce_mode) {
      if (keep_dim) {
        output_idx++;
        cons_dim->prod.emplace_back(std::make_pair(prod_dim, AF_REDUCE));
        prod_dim->cons.emplace_back(std::make_pair(cons_dim, AF_REDUCE));
      } else {
        prod_dim->cons.emplace_back(std::make_pair(nullptr, AF_REDUCE));
      }
    } else {
      output_idx++;
      cons_dim->prod.emplace_back(std::make_pair(prod_dim, AF_ELEMWISE));
      prod_dim->cons.emplace_back(std::make_pair(cons_dim, AF_ELEMWISE));
    }
  }
}

void AffinityAnalyzer::AddTransposeRelation(Tensor *input, Tensor *output) {
  Array<Expr> axis = Downcast<Array<Expr>>(attrs_["perm"]);
  auto perm = ExtractIntVector(axis);
}

std::vector<int64_t> AffinityAnalyzer::ExtractIntVector(Array<Expr> &vec) {
  std::vector<int64_t> res;
  for (Expr s : vec) {
    int64_t val = -1;
    if (s.as<IntImm>()) {
      val = s.as<IntImm>()->value;
    } else if (s.as<UIntImm>()) {
      val = s.as<UIntImm>()->value;
    } else {
      CHECK(0);
    }
    res.push_back(val);
  }
  return res;
}

AffinityAnalyzer::Tensor *AffinityAnalyzer::NewTensor(FunctionRef ref, std::string op, Array<Expr> shape) {
  std::unique_ptr<Tensor> t(new Tensor());
  auto dims = ExtractIntVector(shape);
  int index = 0;
  for (auto dim_size : dims) {
    std::unique_ptr<Dim> d(new Dim());
    d->tensor = t.get();
    d->index = index++;
    d->size = dim_size;
    t->dims.emplace_back(std::move(d));
  }
  t->op = op;
  t->ref = ref;
  auto p = t.get();
  tensors_.emplace_back(std::move(t));
  tensor_map_[ref] = p;
  return p;
}

class PeelMutator : public IRMutator {
 public:
  PeelMutator(std::function<Array<Expr>(FunctionRef, Array<Expr>)> peel_func) : peel_func_(peel_func) {}

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    auto prim_op = op->value.as<Call>();
    Array<Expr> args;
    for (auto &arg : prim_op->args) {
      if (auto tensor = arg.as<Call>()) {
        auto shape = ReplaceShape(tensor->func, tensor->args);
        args.push_back(Call::make(tensor->type, tensor->name, shape, tensor->call_type, tensor->func));
      } else {
        args.push_back(arg);
      }
    }
    auto prim_expr = Call::make(prim_op->type, prim_op->name, args, prim_op->call_type, prim_op->func);
    auto out_shape = ReplaceShape(op->func, op->args);
    return Provide::make(op->func, op->value_index, prim_expr, out_shape);
  }

 private:
  Array<Expr> ReplaceShape(FunctionRef ref, Array<Expr> shape) {
    auto it = replace_shape_.find(ref);
    if (it != replace_shape_.end()) {
      return it->second;
    }
    auto new_shape = peel_func_(ref, shape);
    replace_shape_[ref] = new_shape;
    return new_shape;
  }

  std::function<Array<Expr>(FunctionRef, Array<Expr>)> peel_func_;
  std::unordered_map<FunctionRef, Array<Expr>, NodeHash, NodeEqual> replace_shape_;
};

void DimensionPeeler::Analyze(Stmt s) {
  stmt_ = s;
  AffinityAnalyzer aff;
  aff.Analyze(s);
#if PEEL_DUMP
  std::ostringstream os;
  aff.Dump(os);
  std::cout << os.str();
#endif
  auto dom = BuildAxisSpace(aff);
  for (size_t i = 0; i < axis_space_.size(); ++i) {
    MapDimToSpace(aff, dom->dims[i].get(), i);
  }
}

std::vector<int64_t> DimensionPeeler::GetAxisSpace() {
  std::vector<int64_t> space;
  for (auto &a : axis_space_) {
    space.emplace_back(a->size);
  }
  return space;
}

std::vector<DimensionPeeler::Peeling> DimensionPeeler::GetPeelSpace(int limit_depth,
                                                                    std::unordered_set<int> *limit_range) {
  std::vector<Peeling> peelings;
  std::vector<std::pair<Axis *, int>> cand_axis;
  std::function<void(Peeling, int)> CollectDepth;
  CollectDepth = [&](Peeling seed, int index) {
    auto a = cand_axis[index].first;
    auto d = cand_axis[index].second;
    if (index + 1 < static_cast<int>(cand_axis.size())) {
      CollectDepth(seed, index + 1);
    }
    for (int64_t val : a->peel_val) {
      Peeling peel(seed);
      peel.emplace_back(std::make_pair(d, val));
      if (static_cast<int>(peel.size()) < limit_depth && index + 1 < static_cast<int>(cand_axis.size())) {
        CollectDepth(peel, index + 1);
      }
      peelings.emplace_back(std::move(peel));
    }
  };
  if (limit_depth <= 0) {
    limit_depth = axis_space_.size();
  }
  for (size_t i = 0; i < axis_space_.size(); ++i) {
    if ((limit_range == nullptr || limit_range->count(i)) && !axis_space_[i]->peel_val.empty()) {
      cand_axis.emplace_back(std::make_pair(axis_space_[i].get(), i));
    }
  }
  if (!cand_axis.empty()) {
    CollectDepth(Peeling(), 0);
  }
  return peelings;
}

std::unordered_map<std::string, Peeling> DimensionPeeler::GetPeelTensors(const Peeling &peeling) {
  std::unordered_map<std::string, Peeling> peel_tensors;
  for (auto &kv : dim_map_) {
    auto dim = GetPeelDims(kv.first, peeling);
    if (!dim.empty() &&
        !std::all_of(dim.begin(), dim.end(), [](std::pair<int, int64_t> &i) { return i.first == -1; })) {
      peel_tensors.insert({kv.first, dim});
    }
  }
  return peel_tensors;
}

Stmt DimensionPeeler::GetPeelBody(const Peeling &peeling) {
  auto PeelFunc = [&peeling, this](const FunctionRef &tensor, Array<Expr> shape) -> Array<Expr> {
    auto it = this->dim_map_.find(tensor->func_name());
    if (it == this->dim_map_.end()) {
      return shape;
    }
    auto &dim_map = it->second;
    Array<Expr> new_shape = shape;
    for (auto &p : peeling) {
      auto dim_idx = dim_map[p.first];
      if (dim_idx == -1) {
        continue;
      }
      int64_t dim_val = 0;
      if (auto op = new_shape[dim_idx].as<IntImm>()) {
        dim_val = op->value;
      } else if (auto op = new_shape[dim_idx].as<UIntImm>()) {
        dim_val = op->value;
      } else {
        CHECK(0);
      }
      if (dim_val != 1) {
        CHECK(dim_val % p.second == 0);
        new_shape.Set(dim_idx, make_const(Int(32), dim_val / p.second));
      }
    }
    return new_shape;
  };
  return PeelMutator(PeelFunc).Mutate(stmt_);
}

Stmt DimensionPeeler::GetPeelBody(std::unordered_map<std::string, Peeling> config) {
  auto PeelFunc = [&config](const FunctionRef &tensor, Array<Expr> shape) -> Array<Expr> {
    if (!config.count(tensor->func_name())) {
      return shape;
    }
    auto &dim_map = config[tensor->func_name()];
    Array<Expr> new_shape = shape;
    for (auto &kv : dim_map) {
      auto dim_idx = kv.first;
      if (dim_idx == -1) {
        continue;
      }
      int64_t dim_val = 0;
      if (auto op = new_shape[dim_idx].as<IntImm>()) {
        dim_val = op->value;
      } else if (auto op = new_shape[dim_idx].as<UIntImm>()) {
        dim_val = op->value;
      } else {
        CHECK(0);
      }
      if (dim_val != 1) {
        CHECK(dim_val % kv.second == 0);
        new_shape.Set(dim_idx, make_const(Int(32), dim_val / kv.second));
      }
    }
    return new_shape;
  };
  return PeelMutator(PeelFunc).Mutate(stmt_);
}

Peeling DimensionPeeler::GetPeelDims(const std::string &tensor, const Peeling &peeling) {
  Peeling dims(peeling.size(), {-1, 1});
  auto it = dim_map_.find(tensor);
  if (it != dim_map_.end()) {
    auto &map = it->second;
    for (size_t i = 0; i < peeling.size(); ++i) {
      if (map[peeling[i].first] != -1) {
        dims[i] = {map[peeling[i].first], peeling[i].second};
      }
    }
  }
  return dims;
}

std::vector<int64_t> DimensionPeeler::GetDivisors(int64_t n) {
  std::vector<int64_t> res;
  for (int64_t div = n; div > 1; --div) {
    if (n % div == 0) {
      res.push_back(div);
    }
  }
  return res;
}

DimensionPeeler::Tensor *DimensionPeeler::BuildAxisSpace(AffinityAnalyzer &aff) {
  auto DomLevel = [](const std::string op) -> int {
    if (IsElemwise(op) || op == "InplaceAssign" || op == "Assign" || op == "BroadcastTo") {
      return 1;
    } else if (IsTransform(op)) {
      return 2;
    } else if (IsReduce(op)) {
      return 3;
    } else if (IsOtherOp(op)) {
      return 4;
    }
    return 0;
  };
  Tensor *dom = nullptr;
  int dom_level = -1;
  for (auto it = aff.tensors_.rbegin(); it != aff.tensors_.rend(); it++) {
    auto t = it->get();
    auto l = DomLevel(t->op);
    if (l > dom_level) {
      dom = t;
      dom_level = l;
    }
  }
  if (IsReduce(dom->op)) {
    dom = dom->prod[0];
  }
  for (auto &dim : dom->dims) {
    std::unique_ptr<Axis> axis(new Axis());
    axis->size = dim->size;
    axis->peel_val = GetDivisors(dim->size);
    axis_space_.emplace_back(std::move(axis));
  }
#if PEEL_DUMP
  std::cout << "Select Dom Op: op=" << dom->ref.get() << ", name=" << dom->op << ", AxisSpace:" << std::endl;
  for (size_t i = 0; i < axis_space_.size(); ++i) {
    Axis *axis = axis_space_[i].get();
    std::cout << "  " << i << "%%[" << axis->size << "] : ";
    for (size_t j = 0; j < axis->peel_val.size(); ++j) {
      std::cout << axis->peel_val[j];
      if (j < axis->peel_val.size() - 1) std::cout << ",";
    }
    std::cout << std::endl;
  }
#endif
  return dom;
}

void DimensionPeeler::MapDimToSpace(AffinityAnalyzer &aff, Dim *dom_dim, int axis_idx) {
  std::vector<Dim *> prod_visit;
  std::vector<Dim *> cons_visit;
  std::unordered_set<Dim *> visited;
  auto ProdVisitor = [&](Dim *dim, Dim *prod, int affinity) -> bool {
    if (visited.count(prod)) return false;
    if (prod != nullptr) visited.insert(prod);
    if (!this->Propagation(axis_idx, dim, prod, affinity)) return false;
    for (auto &c : prod->cons) {
      if (!visited.count(c.first)) {
        cons_visit.emplace_back(prod);
        break;
      }
    }
    return true;
  };
  auto ConsVisitor = [&](Dim *dim, Dim *cons, int affinity) -> bool {
    if (visited.count(cons)) return false;
    if (cons != nullptr) visited.insert(cons);
    if (!this->Propagation(axis_idx, dim, cons, affinity)) return false;
    for (auto &p : cons->prod) {
      if (!visited.count(p.first)) {
        prod_visit.emplace_back(cons);
        break;
      }
    }
    return true;
  };
  AddDimMap(dom_dim, axis_idx);
  visited.insert(dom_dim);
  if (!dom_dim->prod.empty()) prod_visit.emplace_back(dom_dim);
  if (!dom_dim->cons.empty()) cons_visit.emplace_back(dom_dim);
  while (!prod_visit.empty() || !cons_visit.empty()) {
    while (!prod_visit.empty()) {
      Dim *dim = prod_visit.back();
      prod_visit.pop_back();
      aff.VisitProd(dim, ProdVisitor);
    }
    while (!cons_visit.empty()) {
      Dim *dim = cons_visit.back();
      cons_visit.pop_back();
      aff.VisitCons(dim, ConsVisitor);
    }
  }
}

bool DimensionPeeler::Propagation(int axis_idx, Dim *from, Dim *to, int affinity) {
  if (affinity == AffinityAnalyzer::AF_ELEMWISE) {
    AddDimMap(to, axis_idx);
  } else if (affinity == AffinityAnalyzer::AF_BROADCAST) {
    return false;
  } else if (affinity == AffinityAnalyzer::AF_REDUCE) {
    axis_space_[axis_idx]->peel_val.clear();
    return false;
  } else {
    CHECK(0);  // TODO: add other type and update axis->peel_val
  }
  return true;
}

void DimensionPeeler::AddDimMap(Dim *dim, int axis_idx) {
  auto it = this->dim_map_.find(dim->tensor->ref->func_name());
  if (it != this->dim_map_.end()) {
    it->second[axis_idx] = dim->index;
  } else {
    std::vector<int> map(this->axis_space_.size(), -1);
    map[axis_idx] = dim->index;
    dim_map_.emplace(dim->tensor->ref->func_name(), std::move(map));
  }
#if PEEL_DUMP
  std::cout << "DimMap: " << dim->tensor->ref.get() << "%%" << dim->tensor->op << ": " << dim->index << " -> "
            << axis_idx << std::endl;
#endif
}

DimensionPeeler::Peeling Str2Peeling(const std::string &peeling) {
  std::vector<std::string> vec;
  std::istringstream iss(peeling);
  std::string s;
  while (iss >> s) {
    vec.push_back(s);
  }

  const int entry_size = 2;
  if (vec.size() % entry_size != 0) {
    LOG(FATAL)
      << "Parse peeling failed, valid peeling's format is a string composed of axis value pair(e.g. \"0 1024 1 "
         "1024\", \"0 1024\", \"1 1024\"), but current is: "
      << peeling;
  }

  DimensionPeeler::Peeling ret;
  char *endptr = nullptr;
  const int base = 10;
  for (size_t i = 0; i < vec.size(); i += entry_size) {
    int axis = static_cast<int>(strtol(vec[i].c_str(), &endptr, base));
    if (*endptr) {
      LOG(FATAL) << "Parse peeling axis failed, axis should be integer, but got " << vec[i];
    }
    int64_t value = strtol(vec[i + 1].c_str(), &endptr, base);
    if (*endptr) {
      LOG(FATAL) << "Parse peeling value failed, value should be integer, but got " << vec[i + 1];
    }
    ret.push_back(std::make_pair(axis, value));
  }

  return ret;
}

Expr Peeling2Str(const DimensionPeeler::Peeling &peeling) {
  std::string ret;
  for (size_t i = 0; i < peeling.size(); ++i) {
    ret += std::to_string(peeling[i].first);
    ret += " ";
    ret += std::to_string(peeling[i].second);
    if (i != peeling.size() - 1) {
      ret += " ";
    }
  }
  return Expr(ret);
}
}  // namespace akg
