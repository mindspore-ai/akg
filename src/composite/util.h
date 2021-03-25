/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef COMPOSITE_UTIL_H_
#define COMPOSITE_UTIL_H_
#include "tvm.h"
#include "picojson.h"

namespace akg {
constexpr auto kMsAscendKernelPath = "./kernel_meta/";
constexpr auto kMsGpuKernelPath = "./cuda_meta";
constexpr auto BLOCK_IDX_X = "blockIdx.x";
constexpr auto BLOCK_IDX_Y = "blockIdx.y";
constexpr auto BLOCK_IDX_Z = "blockIdx.z";
constexpr auto THREAD_IDX_X = "threadIdx.x";
constexpr auto THREAD_IDX_Y = "threadIdx.y";
constexpr auto THREAD_IDX_Z = "threadIdx.z";
constexpr auto BLOCKIDX = "blockIdx.";
constexpr auto BLOCKIDX_LEN = 9;
constexpr auto SHARED = "shared";
constexpr auto ALLOC = "ALLOC";
constexpr auto MEM_LIMIT = 49152;
static std::unordered_map<std::string, air::Type> type_mapping = {
  {"float64", air::Float(64)}, {"float32", air::Float(32)}, {"float16", air::Float(16)}, {"bool", air::Bool()},
  {"int64", air::Int(64)},     {"int32", air::Int(32)},     {"int16", air::Int(16)},     {"int8", air::Int(8)},
  {"uint64", air::UInt(64)},   {"uint32", air::UInt(32)},   {"uint16", air::UInt(16)},   {"uint8", air::UInt(8)},
};

std::string GetProcess(const std::string &json_str);
std::string GetSchedule(Array<Tensor> &outputs);
bool IsBlockIdx(const std::string &name);
bool IsBlockIdxX(const std::string &name);
bool IsBlockIdxY(const std::string &name);
bool IsBlockIdxZ(const std::string &name);
bool IsThreadIdxX(const std::string &name);
bool IsThreadIdxY(const std::string &name);
bool IsThreadIdxZ(const std::string &name);
picojson::value String2Json(const std::string &json_str);
bool IsReduce(const std::string &op_name);
bool IsTransform(const std::string &op_name);
bool IsInplaceAssign(const std::string &op_name);
bool IsAssign(const std::string &op_name);
bool IsOtherOp(const std::string &op_name);
bool IsElemwise(const std::string &op_name);
bool EqualShape(const Array<Expr> &shape1, const Array<Expr> &shape2);
bool ShapeIsOne(const Array<Expr> &shape);
std::string GetOpName(const Provide *p);
std::string CreateDataFormatKey(const std::string &tensor_name);

using FuncRefList = std::vector<FunctionRef>;
using FuncRefMap = std::unordered_map<FunctionRef, FunctionRef, NodeHash, NodeEqual>;
using FuncRefSet = std::unordered_set<FunctionRef, NodeHash, NodeEqual>;
using FuncRefGraph = std::unordered_map<FunctionRef, FuncRefSet, NodeHash, NodeEqual>;
using FuncTensorMap = std::unordered_map<FunctionRef, Tensor, NodeHash, NodeEqual>;
using FuncStmtMap = std::unordered_map<FunctionRef, const Provide *, NodeHash, NodeEqual>;
using FuncShape = std::unordered_map<FunctionRef, Array<Expr>, NodeHash, NodeEqual>;
using FuncExprMap = std::unordered_map<FunctionRef, Expr, NodeHash, NodeEqual>;

struct BuildOpt {
  FuncExprMap inplaces;          // the tensors which should be in bind
  FuncRefMap sames;              // the tensors which are same
  FuncRefSet fakeout;            // the tensors which are not output
  std::vector<Tensor> sch_only;  // the tensors which should only used in sch, not output

  std::string target;
  bool stitch{false};
  size_t stitch_ir_idx_{0};
  bool fold_dim{true};
  FuncRefSet input_funcs;
  FuncRefList output_funcs;

  bool enable_dump{true};
};
inline std::ostream &operator<<(std::ostream &os, const BuildOpt &x) {
  std::string indent = "  ";
  os << "[BuildOpt] : " << std::endl;
  os << "- input: " << std::endl;
  for (const auto &a : x.input_funcs) {
    os << indent << a->func_name() << ", ";
  }
  os << std::endl;
  os << "- output: " << std::endl;
  for (const auto &a : x.output_funcs) {
    os << indent << a->func_name() << ", ";
  }
  os << std::endl;
  os << "- inplaces: " << std::endl;
  for (const auto &kv : x.inplaces) {
    os << indent << kv.first->func_name() << " : " << kv.second << std::endl;
  }
  os << "- sames: " << std::endl;
  for (const auto &kv : x.sames) {
    os << indent << kv.first->func_name() << " : " << kv.second->func_name() << std::endl;
  }
  os << "- fakeout: " << std::endl;
  for (const auto &a : x.fakeout) {
    os << indent << a->func_name() << std::endl;
  }
  os << "- sch_only: " << std::endl;
  for (const auto &a : x.sch_only) {
    os << indent << a->op->func_name() << std::endl;
  }
  os << "- target: " << std::endl;
  os << x.target << std::endl;
  os << "- stitch: " << std::endl;
  os << x.stitch << std::endl;
  os << "- fold_dim: " << std::endl;
  os << x.fold_dim << std::endl;
  return os;
}

struct BuildInfo {
  std::vector<std::string> input_names;   // names of kernel's inputs
  std::vector<std::string> output_names;  // names of kernel's outputs
  Array<Tensor> tensors;                  // topi's output tensor, which should be compute node
  Array<NodeRef> args;                    // the composite kernel's inputs and outputs
  Map<Tensor, Buffer> in_binds;           // the tensors which should be in bind
  std::string kernel_name;                // the composite kernel's name
  BuildOpt opt;
};
inline std::ostream &operator<<(std::ostream &os, const BuildInfo &x) {
  std::string indent = "  ";
  os << "[BuildInfo] : " << std::endl;
  os << "- input names: " << std::endl;
  for (const auto &in_name : x.input_names) {
    os << indent << in_name << std::endl;
  }
  os << "- output names: " << std::endl;
  for (const auto &out_name : x.output_names) {
    os << indent << out_name << std::endl;
  }
  os << "- tensors: " << std::endl;
  for (const auto &a : x.tensors) {
    os << indent << a->op->func_name() << std::endl;
  }
  os << "- args: " << std::endl;
  for (const auto &a : x.args) {
    os << indent << a << std::endl;
  }
  os << "- in_binds: " << std::endl;
  for (const auto &kv : x.in_binds) {
    os << indent << kv.first->op->func_name() << " : " << kv.second << std::endl;
  }
  os << x.opt;
  return os;
}

struct TensorInfo {
  std::string name_;
  std::string format_;
  Array<Expr> shape_;
  Type dtype_;
  bool has_value_{false};
  picojson::value value_;
};

struct OpDesc {
  std::string op_name;
  Map<std::string, NodeRef> attrs;
  Array<NodeRef> input_descs;
  Array<NodeRef> output_descs;
  std::vector<TensorInfo> input_tensor_info;
  std::vector<TensorInfo> output_tensor_info;
};

struct Graph {
  FuncRefGraph pre_graph;
  FuncRefGraph post_graph;
  FuncStmtMap func_stmts;
  FuncRefSet input_funcs;
  FuncRefList output_funcs;
  FuncRefSet visited_funcs;
  FuncShape func_shape;
  bool CanChangeElem(const FunctionRef &output) {
    // if all input shape same as output shape, it can be changed.
    // consider special case: if elemwise input tensor shape is [1], can auto broadcast
    auto inputs = pre_graph[output];
    for (const auto &input : inputs) {
      if (!EqualShape(func_shape[input], func_shape[output]) && !ShapeIsOne(func_shape[input])) {
        return false;
      }
    }
    return true;
  }
};

class StmtToGraph : public IRVisitor {
 public:
  StmtToGraph(const FuncRefSet &input_funcs, const FuncRefList &output_funcs) {
    g_.input_funcs = input_funcs;
    g_.output_funcs = output_funcs;
  };

 private:
  void Visit_(const Provide *op) override {
    auto call = op->value.as<Call>();
    CHECK(call);
    FuncRefSet inputs = GetInputsFunc(call->args);
    FunctionRef output = op->func;
    g_.pre_graph[output] = inputs;
    for (const auto &input : inputs) {
      g_.post_graph[input].insert(output);
    }
    g_.func_stmts[op->func] = op;
    g_.func_shape[op->func] = op->args;
  }
  FuncRefSet GetInputsFunc(const Array<Expr> &inputs) {
    FuncRefSet set;
    for (const auto &item : inputs) {
      if (auto call = item.as<Call>()) {
        set.insert(call->func);
        g_.func_shape[call->func] = call->args;
      }
    }
    return set;
  }

 public:
  Graph g_;
};

struct NeedReshape {
  FunctionRef func;
  Array<Expr> origin_shape;
};

struct AnalysisResult {
  FuncRefMap to_be_replaced;
  std::unordered_set<const Provide *> to_be_removed;
  std::unordered_map<FunctionRef, Array<Expr>, NodeHash, NodeEqual> changed_shapes;
  std::unordered_map<const Provide *, std::vector<NeedReshape>> need_reshape_map;
  bool ShapeChanged(const FunctionRef &tensor) { return changed_shapes.find(tensor) != changed_shapes.end(); }
  void CollectReshape(const Provide *op, const FunctionRef &func, const Array<Expr> &origin_shape,
                      const Array<Expr> &changed_shape) {
    if (EqualShape(origin_shape, changed_shape)) return;
    NeedReshape nr;
    nr.func = to_be_replaced.count(func) ? to_be_replaced[func] : func;
    nr.origin_shape = origin_shape;
    need_reshape_map[op].emplace_back(nr);
  }
  void Dump() {
    LOG(INFO) << "\n=======to_be_replaced=======\n";
    for (const auto &item : to_be_replaced) {
      LOG(INFO) << item.first->func_name() << " -> " << item.second->func_name() << "\n";
    }
    LOG(INFO) << "\n=======to_be_removed=======\n";
    for (const auto &item : to_be_removed) {
      LOG(INFO) << item->func->func_name() << " " << item->value.as<Call>()->name << "\n";
    }
    LOG(INFO) << "\n=======changed_shapes=======\n";
    for (const auto &item : changed_shapes) {
      LOG(INFO) << item.first->func_name() << " -> " << item.second << "\n";
    }
    LOG(INFO) << "\n=======need_reshape_map=======\n";
    for (const auto &item : need_reshape_map) {
      LOG(INFO) << item.first->func->func_name() << " -> \n";
      for (const auto &j : item.second) {
        LOG(INFO) << j.func->func_name() << " -> " << j.origin_shape << "\n";
      }
    }
  }
};

class DoAnalysis : public IRMutator {
 public:
  explicit DoAnalysis(AnalysisResult result) : result_(std::move(result)){};

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "attrs") {
      op_attrs_ = Downcast<Map<std::string, NodeRef>>(op->node);
      auto stmt = IRMutator::Mutate_(op, s);
      op_attrs_ = {};
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }
  Expr UpdateInputs(const Call *call) {
    CHECK(call);
    Array<Expr> args;
    for (const auto &arg : call->args) {
      if (auto tensor = arg.as<Call>()) {
        auto shape = tensor->args;
        auto tensor_func = tensor->func;
        if (result_.changed_shapes.find(tensor->func) != result_.changed_shapes.end()) {
          shape = result_.changed_shapes[tensor->func];
        }
        if (result_.to_be_replaced.find(tensor->func) != result_.to_be_replaced.end()) {
          tensor_func = result_.to_be_replaced[tensor->func];
        }
        args.push_back(Call::make(tensor->type, tensor_func->func_name(), shape, tensor->call_type, tensor_func));
      } else {
        args.push_back(arg);
      }
    }
    return Call::make(call->type, call->name, args, call->call_type, call->func);
  }
  Stmt UpdateOutput(const Provide *op, Expr &new_call) {
    Array<Expr> output_shape = op->args;
    if (result_.changed_shapes.count(op->func)) {
      output_shape = result_.changed_shapes[op->func];
    }
    return Provide::make(op->func, op->value_index, new_call, output_shape);
  }
  Expr InputsTryAddReshape(const Expr &new_call, const std::vector<NeedReshape> &nr_vec, std::vector<Stmt> &stmts) {
    auto new_call_p = new_call.as<Call>();
    // input need reshape
    Array<Expr> new_args;
    for (auto &arg : new_call_p->args) {
      auto tmp_arg = arg;
      if (auto input = arg.as<Call>()) {
        for (const auto &nr : nr_vec) {
          if (nr.func == input->func) {
            // if input shape changed, input need reshape
            // b = reduce(a) -> t = trans(a); b = reduce(t)
            auto tensor = NewTensor(nr.origin_shape);
            auto reshape = AddReshape(input->func, tensor->op, input->args, tensor->shape, input->type);
            stmts.emplace_back(reshape);
            tmp_arg =
              Call::make(input->type, tensor->op->func_name(), tensor->shape, new_call_p->call_type, tensor->op);
            break;
          }
        }
      }
      new_args.push_back(tmp_arg);
    }
    return Call::make(new_call_p->type, new_call_p->name, new_args, new_call_p->call_type, new_call_p->func);
  }
  void OutputTryAddReshape(const FunctionRef &output, const Provide *provide, const std::vector<NeedReshape> &nr_vec,
                           std::vector<Stmt> &stmts) {
    for (const auto &nr : nr_vec) {
      if (nr.func == output) {
        // if output shape changed, output need reshape
        // b = reduce(a) -> t = reduce(a); b = trans(t)
        auto tensor = NewTensor(nr.origin_shape);
        auto reshape = AddReshape(tensor->op, provide->func, tensor->shape, provide->args, Int(32));
        auto stmt = Provide::make(tensor->op, provide->value_index, provide->value, tensor->shape);
        if (!op_attrs_.empty()) {
          stmt = AttrStmt::make(op_attrs_, "attrs", Expr(1), stmt);
        }
        stmts.pop_back();  // provide need update
        stmts.emplace_back(stmt);
        stmts.emplace_back(reshape);
        break;
      }
    }
  }
  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    if (result_.to_be_removed.count(op)) {
      return Evaluate::make(0);
    }
    auto call = UpdateInputs(op->value.as<Call>());
    auto provide = UpdateOutput(op, call);
    if (result_.need_reshape_map.count(op)) {
      std::vector<Stmt> stmts;
      auto new_call = InputsTryAddReshape(call, result_.need_reshape_map[op], stmts);
      auto provide_p = provide.as<Provide>();
      auto new_provide = Provide::make(provide_p->func, provide_p->value_index, new_call, provide_p->args);
      auto stmt = new_provide;
      if (!op_attrs_.empty()) {
        stmt = AttrStmt::make(op_attrs_, "attrs", Expr(1), new_provide);
      }
      stmts.emplace_back(stmt);
      OutputTryAddReshape(op->func, new_provide.as<Provide>(), result_.need_reshape_map[op], stmts);
      return Block::make(stmts);
    }
    return provide;
  }

  static Stmt AddReshape(const FunctionRef &input_func, const FunctionRef &output_func, const Array<Expr> &input_shape,
                         const Array<Expr> &output_shape, const air::DataType &type) {
    Array<Expr> input;
    input.push_back(Call::make(type, input_func->func_name(), input_shape, Call::CallType::Halide, input_func));
    auto stmt =
      Provide::make(output_func, 0, Call::make(Int(32), "Reshape", input, Call::CallType::PureIntrinsic), output_shape);
    Map<std::string, NodeRef> attrs;
    attrs.Set("shape", output_shape);
    stmt = AttrStmt::make(attrs, "attrs", Expr(1), stmt);
    return stmt;
  }

  Tensor NewTensor(const Array<Expr> &shape) {
    std::stringstream ss;
    ss << "tmp_" << count_++;
    return placeholder(shape, Int(1), ss.str());
  }

 private:
  AnalysisResult result_;
  int count_{0};
  Map<std::string, NodeRef> op_attrs_;
};

struct GridBlockDims {
  int blockdim_x{1};
  int blockdim_y{1};
  int blockdim_z{1};
  int griddim_x{1};
  int griddim_y{1};
  int griddim_z{1};

  GridBlockDims &operator=(const GridBlockDims &s) {
    blockdim_x = s.blockdim_x;
    blockdim_y = s.blockdim_y;
    blockdim_z = s.blockdim_z;
    griddim_x = s.griddim_x;
    griddim_y = s.griddim_y;
    griddim_z = s.griddim_z;
    return *this;
  }
};
inline std::ostream &operator<<(std::ostream &os, const GridBlockDims &x) {
  os << "GridBlockDims: " << x.griddim_x << " " << x.griddim_y << " " << x.griddim_z << " " << x.blockdim_x << " "
     << x.blockdim_y << " " << x.blockdim_z << "\n";
  return os;
}

class GridBlockDimsAttr : public IRVisitor {
 public:
  GridBlockDimsAttr() = default;
  void Visit_(const AttrStmt *op) {
    if (op->attr_key == air::ir::attr::thread_extent) {
      const IterVarNode *iv = op->node.as<IterVarNode>();
      CHECK(iv);
      std::string name = iv->thread_tag;
      if (IsThreadIdxX(name)) {
        dims.blockdim_x = op->value.as<IntImm>()->value;
      } else if (IsThreadIdxY(name)) {
        dims.blockdim_y = op->value.as<IntImm>()->value;
      } else if (IsThreadIdxZ(name)) {
        dims.blockdim_z = op->value.as<IntImm>()->value;
      } else if (IsBlockIdxX(name)) {
        dims.griddim_x = op->value.as<IntImm>()->value;
      } else if (IsBlockIdxY(name)) {
        dims.griddim_y = op->value.as<IntImm>()->value;
      } else if (IsBlockIdxZ(name)) {
        dims.griddim_z = op->value.as<IntImm>()->value;
      }
    }
    IRVisitor::Visit(op->body);
  }
  void Visit_(const For *op) { Visit(op->body); }

 public:
  GridBlockDims dims;
};
Map<std::string, NodeRef> BindBlockAndThread(GridBlockDims &dims, bool poly, const Map<std::string, NodeRef> &attrs);
Stmt InsertSync(Stmt &s);
}  // namespace akg

#endif  // COMPOSITE_UTIL_H_
