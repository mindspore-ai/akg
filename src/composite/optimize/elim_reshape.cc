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
#include "composite/optimize/elim_reshape.h"

namespace akg {
void ElimReshapeAnalysis::Run() {
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
  }
}
void ElimReshapeAnalysis::AnalysisTransform(const FunctionRef &output) {
  auto provide = g_.func_stmts[output];
  auto call = provide->value.as<Call>();
  CHECK(call);
  CHECK(call->args.size() == 1);
  CHECK(call->args[0].as<Call>());
  auto input = call->args[0].as<Call>()->func;
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
  }  // else if visited and input output shape are different, do noting, if input shape changed, already in set
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
  for (const auto &input : inputs) {
    auto input_shape = result_.ShapeChanged(input) ? result_.changed_shapes[input] : g_.func_shape[input];
    CHECK(input_map_shape_change.count(input));
    auto input_shape_change = input_map_shape_change[input];
    if (ShapeIsOne(input_shape)) continue;
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
        result_.CollectReshape(g_.func_stmts[output], input, input_shape_change, input_shape);
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
    result_.CollectReshape(provide, output, g_.func_shape[output], output_shape);
  }
  if (!(IsReduce(op_name) && ShapeIsOne(output_shape))) {  // we consider that allreduce op's input shape is flexable
    // if input shape changed, input need reshape
    // b = reduce(a) -> t = trans(a); b = reduce(t)
    auto inputs = g_.pre_graph[output];
    for (const auto &input : inputs) {
      g_.visited_funcs.insert(input);
      if (result_.ShapeChanged(input)) {
        LOG(INFO) << "[UNELEMWISE] INPUT RESHAPE: " << input->func_name() << ": " << g_.func_shape[input] << "->"
                  << result_.changed_shapes[input];
        result_.CollectReshape(provide, input, g_.func_shape[input], result_.changed_shapes[input]);
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
        result_.CollectReshape(provide, input1, input0_shape, input1_shape);
      }
    }
  }
}

void ElimReshapeAnalysis::AnalysisInner(const FunctionRef &output) {
  if (!g_.func_stmts.count(output)) return;
  auto provide = g_.func_stmts[output];
  auto op_name = GetOpName(provide);
  if (IsTransform(op_name)) {
    AnalysisTransform(output);
  } else if (IsElemwise(op_name) && g_.CanChangeElem(output)) {
    if (!AnalysisElemwise(output)) {
      return AnalysisOthers(output);
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
Stmt ElimReshapeBackward::Run(const Stmt &s) {
  auto checker = ElimReshapeOpChecker();
  checker.Visit(s);
  if (!checker.can_elim) return s;
  auto f = StmtToGraph(info_.opt.input_funcs, info_.opt.output_funcs);
  f.Visit(s);
  AnalysisResult result;
  auto analysis = ElimReshapeAnalysis(f.g_, info_.opt, result, false);
  analysis.Run();
  result.Dump();
  return AnalysisResultMutator(result).Mutate(s);
}
Stmt ElimReshapeForward::Run(const Stmt &s) {
  auto checker = ElimReshapeOpChecker();
  checker.Visit(s);
  if (!checker.can_elim) return s;
  auto f = StmtToGraph(info_.opt.input_funcs, info_.opt.output_funcs);
  f.Visit(s);
  AnalysisResult result;
  auto analysis = ElimReshapeAnalysis(f.g_, info_.opt, result, true);
  analysis.Run();
  result.Dump();
  return AnalysisResultMutator(result).Mutate(s);
}
}  // namespace akg
