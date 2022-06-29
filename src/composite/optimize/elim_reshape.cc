/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <unordered_map>
#include "composite/optimize/pass.h"
#include "composite/optimize/elim_reshape.h"
#include "tvm.h"

namespace akg {
bool ElimReshapeAnalysis::Run() {
  // from input to output, try to remove each transform op, when removed op, should change each tensor's shape by
  // elemwise op, and try to add reshape op when unelemwise op's input shape and output shape are changed.
  size_t settled_size;
  do {
    settled_size = g_.visited_funcs.size();
    if (forward_) {
      AnalysisForward();
    } else {
      AnalysisBackkward();
    }
  } while (settled_size != g_.visited_funcs.size());

  bool elim_valid = AnalysisElimValid();
  result_.Dump(elim_valid);
  if (!elim_valid) {
    return false;
  }

  for (const auto &p : result_.to_be_removed) {
    // if output removed, should collect opt.sames
    if (std::find(g_.output_funcs.begin(), g_.output_funcs.end(), p->func) != g_.output_funcs.end()) {
      opt_.sames[p->func] = result_.to_be_replaced[p->func];
    } else {
      for (auto pair : opt_.sames) {
        if (pair.second == p->func) {
          opt_.sames[pair.second] = result_.to_be_replaced[p->func];
        }
      }
    }
  }
  return true;
}

FunctionRef GetFuncFromPos(const Provide *provide, const size_t &pos, const FuncRefList &inputs) {
  if (pos < inputs.size()) {
    return inputs[pos];
  }
  auto call = provide->value.as<Call>();
  CHECK(call);
  CHECK_EQ(pos, call->args.size());
  return provide->func;
}

bool ElimReshapeAnalysis::ForwardHasOtherOp(const FuncRefList &funcs, FuncBoolMap &cache_res) {
  for (auto func : funcs) {
    if (cache_res.count(func)) {
      return cache_res.at(func);
    }
    cache_res[func] = false;
    CHECK(g_.func_stmts.count(func));
    auto provide = g_.func_stmts[func];
    auto op_name = GetOpName(provide);
    if (!(IsTransform(op_name) || IsElemwise(op_name) || IsInplaceAssign(op_name))) {
      cache_res[func] = true;
      return true;
    }
    if (!g_.post_graph.count(func)) continue;
    auto outputs = g_.post_graph[func];
    if (ForwardHasOtherOp(outputs, cache_res)) {
      cache_res[func] = true;
      return true;
    }
  }
  return false;
}

int ElimReshapeAnalysis::ElimForwardEasier() {
  auto &elim_reshapes = result_.to_be_removed;
  FuncRefList elim_funcs;
  for (auto provide : elim_reshapes) {
    elim_funcs.push_back(provide->func);
  }
  FuncRefList insert_funcs;
  auto &insert_reshapes = result_.need_reshape_map;
  for (const auto &kv : insert_reshapes) {
    auto provide = kv.first;
    insert_funcs.push_back(provide->func);
  }
  FuncBoolMap cache_res;
  bool elim_has_other_op = ForwardHasOtherOp(elim_funcs, cache_res);
  bool insert_has_other_op = ForwardHasOtherOp(insert_funcs, cache_res);
  if (elim_has_other_op && !insert_has_other_op) {
    return 1;
  } else if (!elim_has_other_op && insert_has_other_op) {
    return -1;
  }
  return 0;
}

bool ElimReshapeAnalysis::AnalysisElimValid() {
  auto &elim_reshapes = result_.to_be_removed;
  auto &insert_reshapes = result_.need_reshape_map;
  // The less the number of reshape operators, the better
  auto elim_op_count = elim_reshapes.size();
  size_t insert_op_count = 0;
  for (auto kv : insert_reshapes) {
    insert_op_count += kv.second.size();
  }
  if (insert_op_count < elim_op_count) {
    return true;
  } else if (insert_op_count > elim_op_count) {
    return false;
  }

  // After the ElimReshapeBackward, the easier it is to the ElimReshapeForward, the better
  if (!forward_) {
    int elim_forward_easier = ElimForwardEasier();
    if (elim_forward_easier > 0) {
      return true;
    } else if (elim_forward_easier < 0) {
      return false;
    }
  }

  // The less dimensionality increased by reshape operators, the better
  auto elim_dim_inc = 0;
  for (const auto &provide : elim_reshapes) {
    auto out_shape = provide->args;
    auto call = provide->value.as<Call>();
    CHECK(call);
    CHECK_EQ(call->args.size(), 1);
    CHECK(call->args[0].as<Call>());
    auto input_shape = call->args[0].as<Call>()->args;
    elim_dim_inc += static_cast<int>(out_shape.size() - input_shape.size());
  }
  auto insert_dim_inc = 0;
  for (const auto &kv : insert_reshapes) {
    auto provide = kv.first;
    auto output_func = provide->func;
    for (const auto &nr : kv.second) {
      auto func = GetFuncFromPos(provide, nr.pos, g_.pre_graph[output_func]);
      auto shape_change = result_.changed_shapes.count(func) ? result_.changed_shapes[func] : g_.func_shape[func];
      auto ori_shape = nr.origin_shape;
      if (func == output_func) {
        // for output: reshape ori_shape to shape_change
        insert_dim_inc += static_cast<int>(shape_change.size() - ori_shape.size());
      } else {
        // for input: reshape shape_change to ori_shape
        insert_dim_inc += static_cast<int>(ori_shape.size() - shape_change.size());
      }
    }
  }
  if (insert_dim_inc < elim_dim_inc) {
    return true;
  }
  return false;
}

void ElimReshapeAnalysis::AnalysisForward() {
  for (const auto &input : g_.input_funcs) {
    if (!g_.post_graph.count(input)) continue;
    for (const auto &output : g_.post_graph[input]) {
      AnalysisInner(output);
    }
  }
}
void ElimReshapeAnalysis::AnalysisBackkward() {
  for (const auto &output : g_.output_funcs) {
    AnalysisInner(output);
    if (opt_.sames.count(output)) {
      AnalysisInner(opt_.sames[output]);
    }
  }
}
void ElimReshapeAnalysis::AnalysisTransform(const FunctionRef &output) {
  auto provide = g_.func_stmts[output];
  auto call = provide->value.as<Call>();
  CHECK(call);
  CHECK(call->args.size() == 1);
  CHECK(call->args[0].as<Call>());
  auto input = call->args[0].as<Call>()->func;
  if (std::find(g_.input_funcs.begin(), g_.input_funcs.end(), input) != g_.input_funcs.end() &&
      std::find(g_.output_funcs.begin(), g_.output_funcs.end(), output) != g_.output_funcs.end()) {
    return;
  }
  // if not visited or input shape and output shape as same, can remove this op, change input shape to output
  // shape, replace output tensor to input tensor
  auto input_shape = result_.ShapeChanged(input) ? result_.changed_shapes[input] : g_.func_shape[input];
  auto output_shape = result_.ShapeChanged(output) ? result_.changed_shapes[output] : g_.func_shape[output];
  if ((forward_ && !g_.visited_funcs.count(output)) || (!forward_ && !g_.visited_funcs.count(input)) ||
      EqualShape(input_shape, output_shape)) {
    auto output_replace = result_.to_be_replaced.count(input) ? result_.to_be_replaced[input] : input;
    result_.to_be_replaced[output] = output_replace;
    // if any tensor replace to input already, it should change to new input
    for (auto &kv : result_.to_be_replaced) {
      if (kv.second == output) {
        kv.second = input;
      }
    }
    if (forward_) {
      result_.changed_shapes[output] = input_shape;
    } else {
      result_.changed_shapes[input] = output_shape;
    }
    result_.to_be_removed.insert(provide);
    g_.visited_funcs.insert(output);
    g_.visited_funcs.insert(input);
  }  // else if visited and input output shape are different, do nothing, if input shape changed, already in set
}

bool ElimReshapeAnalysis::AnalysisElemwise(const FunctionRef &output) {
  // If the boundary between the broadcast axis and the elewise axis is broken by reshape,
  // the AnalysisElemwise process cannot be performed and return false.
  // Otherwise, return true.
  if (forward_) {
    return AnalysisElemwiseForward(output);
  } else {
    return AnalysisElemwiseBackward(output);
  }
}
bool ElimReshapeAnalysis::AnalysisElemwiseBackward(const FunctionRef &output) {
  auto inputs = g_.pre_graph[output];
  auto output_changed = result_.ShapeChanged(output);
  auto output_shape = output_changed ? result_.changed_shapes[output] : g_.func_shape[output];
  auto input_map_shape_change = BroadcastReshapeUtil::GetInputsChangeShape(output, g_, output_shape);
  if (input_map_shape_change.empty()) {
    return false;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    auto input_shape = result_.ShapeChanged(input) ? result_.changed_shapes[input] : g_.func_shape[input];
    CHECK(input_map_shape_change.count(input));
    auto input_shape_change = input_map_shape_change[input];
    if (ShapeIsOne(input_shape) && !ShapeSizeIsOne(output_shape)) continue;
    if (!g_.visited_funcs.count(input)) {
      // if not visited and output changed, change input shape
      if (output_changed) {
        result_.changed_shapes[input] = input_shape_change;
        g_.visited_funcs.insert(input);
      }
    } else {
      // if visited, check input_shape and input_shape_change are same or not, if not, need reshape
      if (!EqualShape(input_shape_change, input_shape)) {
        LOG(INFO) << "[ELEMWISE] RESHAPE: " << input->func_name() << ": " << input_shape_change << "->" << input_shape;
        result_.CollectReshape(g_.func_stmts[output], i, input_shape_change, input_shape);
      } else {
        auto op = g_.func_stmts[output];
        if (!result_.need_reshape_map.count(op)) continue;
        for (auto it = result_.need_reshape_map[op].begin(); it < result_.need_reshape_map[op].end();) {
          if ((*it).pos == i && !EqualShape((*it).origin_shape, input_shape)) {
            it = result_.need_reshape_map[op].erase(it);
          } else {
            ++it;
          }
        }
      }
    }
  }
  return true;
}

bool ElimReshapeAnalysis::AnalysisElemwiseForward(const FunctionRef &output) {
  auto inputs = g_.pre_graph[output];
  bool output_changed = false;
  for (const auto &input : inputs) {
    if (result_.ShapeChanged(input) && !g_.visited_funcs.count(output)) {
      auto input_shape = result_.changed_shapes[input];
      auto output_shape_change = input_shape;
      if (!EqualShape(g_.func_shape[output], g_.func_shape[input])) {
        output_shape_change =
          BroadcastReshapeUtil::GetOutputShapeChange(g_.func_shape[output], g_.func_shape[input], input_shape);
        if (output_shape_change.empty()) {
          return false;
        }
      }
      output_changed = true;
      result_.changed_shapes[output] = output_shape_change;
      g_.visited_funcs.insert(output);
      break;
    }
  }
  if (!AnalysisElemwiseBackward(output)) {
    // If other inputs cannot broadcast to the updated output,
    // the change of the output should be rolled back.
    if (output_changed) {
      result_.changed_shapes.erase(output);
      g_.visited_funcs.erase(output);
    }
    return false;
  }
  return true;
}

void ElimReshapeAnalysis::AnalysisOthers(const FunctionRef &output) {
  auto provide = g_.func_stmts[output];
  auto op_name = GetOpName(provide);
  auto output_shape = result_.ShapeChanged(output) ? result_.changed_shapes[output] : g_.func_shape[output];
  // if output shape changed, output need reshape
  // b = reduce(a) -> t = reduce(a); b = trans(t)
  g_.visited_funcs.insert(output);
  if (result_.ShapeChanged(output)) {
    LOG(INFO) << "[UNELEMWISE] OUTPUT RESHAPE: " << output->func_name() << ": " << g_.func_shape[output] << "->"
              << output_shape;
    // input_size denote the pos of output
    auto call = provide->value.as<Call>();
    CHECK(call);
    auto input_size = call->args.size();
    result_.CollectReshape(provide, input_size, g_.func_shape[output], output_shape);
  }
  if (!(IsReduce(op_name) && ShapeIsOne(output_shape))) {  // we consider that allreduce op's input shape is flexable
    // if input shape changed, input need reshape
    // b = reduce(a) -> t = trans(a); b = reduce(t)
    auto inputs = g_.pre_graph[output];
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto &input = inputs[i];
      g_.visited_funcs.insert(input);
      if (result_.ShapeChanged(input)) {
        LOG(INFO) << "[UNELEMWISE] INPUT RESHAPE: " << input->func_name() << ": " << g_.func_shape[input] << "->"
                  << result_.changed_shapes[input];
        result_.CollectReshape(provide, i, g_.func_shape[input], result_.changed_shapes[input]);
      }
    }
  }
}

void ElimReshapeAnalysis::AnalysisInplaceAssign(const FunctionRef &output) {
  auto inputs = g_.pre_graph[output];
  bool output_changed = result_.ShapeChanged(output);
  auto output_shape = output_changed ? result_.changed_shapes[output] : g_.func_shape[output];
  CHECK(inputs.size() == 3);
  auto input2 = inputs[2];
  if (!g_.visited_funcs.count(input2)) {
    // if not visited and output changed, change input2 shape
    if (output_changed) {
      result_.changed_shapes[input2] = output_shape;
      g_.visited_funcs.insert(input2);
    }
  } else {
    auto input_shape = result_.ShapeChanged(input2) ? result_.changed_shapes[input2] : g_.func_shape[input2];
    if (!EqualShape(output_shape, input_shape)) {
      result_.changed_shapes[output] = input_shape;
      g_.visited_funcs.insert(output);
    }
  }
  auto input0 = inputs[0];
  auto input1 = inputs[1];
  auto input0_shape = result_.ShapeChanged(input0) ? result_.changed_shapes[input0] : g_.func_shape[input0];
  auto input1_shape = result_.ShapeChanged(input1) ? result_.changed_shapes[input1] : g_.func_shape[input1];
  if (g_.visited_funcs.count(input1) && result_.ShapeChanged(input1)) {
    if (!g_.visited_funcs.count(input0)) {
      result_.changed_shapes[input0] = input1_shape;
      g_.visited_funcs.insert(input0);
    } else {
      if (!EqualShape(input0_shape, input1_shape)) {
        LOG(INFO) << "[INPLACEASSIGN] INPUT RESHAPE: " << input1->func_name() << ": " << input0_shape << "->"
                  << input1_shape;
        auto provide = g_.func_stmts[output];
        size_t input1_pos = 1;
        result_.CollectReshape(provide, input1_pos, input0_shape, input1_shape);
      }
    }
  }
}

void ElimReshapeAnalysis::AnalysisInner(const FunctionRef &output) {
  if (!g_.func_stmts.count(output)) return;
  auto provide = g_.func_stmts[output];
  auto op_name = GetOpName(provide);
  auto inputs = g_.pre_graph[output];
  if (op_name == "BroadcastTo" && inputs.empty()) return;
  if (IsTransform(op_name)) {
    AnalysisTransform(output);
  } else if ((IsElemwise(op_name) && g_.CanChangeElem(output)) || op_name == "BroadcastTo") {
    if (!AnalysisElemwise(output)) {
      AnalysisOthers(output);
    }
  } else if (IsInplaceAssign(op_name)) {
    AnalysisInplaceAssign(output);
  } else {
    // the op which can not change shape
    AnalysisOthers(output);
  }
  if (forward_) {
    if (!g_.post_graph.count(output)) return;
    auto outputs = g_.post_graph[output];
    for (const auto &out : outputs) {
      AnalysisInner(out);
    }
  } else {
    auto inputs = g_.pre_graph[output];
    for (const auto &input : inputs) {
      AnalysisInner(input);
    }
  }
}

std::string GetId(const std::string &name, int count) {
  std::stringstream id;
  id << name << "_" << count;
  return id.str();
}

/*B(64,1) = Reshape(A(64))
  C = TensorScatterAdd(par, B(64,1), update)
  ----------------------->
  C = TensorScatterAdd(par, A(64), update)
*/
class TSA : public IRMutator {
 public:
  explicit TSA(AnalysisResult &result) : result_(result){};
  ~TSA() = default;

 private:
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto op_name = GetOpName(op);
    auto call = op->value.as<Call>();
    CHECK(call);
    if (IsTransform(op_name)) {
      CHECK(call->args[0].as<Call>());
      reshape_[op->func] = call->args[0];
      p_.emplace_back(op);
    }
    if (op_name == "TensorScatterAdd") {
      constexpr size_t three_args = 3;
      CHECK(call->args.size() == three_args);
      auto index = call->args[1].as<Call>();
      CHECK(index);
      auto arg1 = index->func;
      if (reshape_.count(arg1) && index->args.size() == 2 && index->args[1].as<IntImm>()->value == 1) {
        for (auto &it : p_) {
          if (it->func == arg1) {
            result_.to_be_removed.insert(it);
          }
        }
        auto new_call = Call::make(call->type, op_name, {call->args[0], reshape_[arg1], call->args[2]}, call->call_type,
                                   call->func, call->value_index);
        return Provide::make(op->func, op->value_index, new_call, op->args);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  AnalysisResult &result_;
  FuncExprMap reshape_;
  std::vector<const Provide *> p_;
};

Stmt ElimReshapeBackward(const Stmt &stmt, BuildInfo *info) {
  auto s = stmt;
  AnalysisResult as;
  s = TSA(as).Mutate(s);
  s = AnalysisResultMutator(as).Mutate(s);

  auto checker = ElimReshapeOpChecker();
  checker.Visit(s);
  if (!checker.can_elim) return s;
  auto count = 0;
  auto max_try_count = 10;
  while (count++ < max_try_count) {
    auto f = StmtToGraph(info->opt.input_funcs, info->opt.output_funcs);
    f.Visit(s);
    AnalysisResult result;
    auto analysis = ElimReshapeAnalysis(f.g_, info->opt, result, false);
    bool elim_valid = analysis.Run();
    if (!elim_valid) {
      return s;
    }
    s = AnalysisResultMutator(result, GetId("b", count)).Mutate(s);
  }
  LOG(WARNING) << "ElimReshapeBackward reach to max_try_count!";
  return s;
}

Stmt ElimReshapeForward(const Stmt &stmt, BuildInfo *info) {
  auto s = stmt;
  auto checker = ElimReshapeOpChecker();
  checker.Visit(s);
  if (!checker.can_elim) return s;
  auto count = 0;
  auto max_try_count = 10;
  while (count++ < max_try_count) {
    auto f = StmtToGraph(info->opt.input_funcs, info->opt.output_funcs);
    f.Visit(s);
    AnalysisResult result;
    auto analysis = ElimReshapeAnalysis(f.g_, info->opt, result, true);
    bool elim_valid = analysis.Run();
    if (!elim_valid) {
      return s;
    }
    s = AnalysisResultMutator(result, GetId("f", count)).Mutate(s);
  }
  LOG(WARNING) << "ElimReshapeForward reach to max_try_count!";
  return s;
}
}  // namespace akg
