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
#include "poly/tiling/hermes/init_graph.h"
#include "poly/tiling/hermes/tensor.h"
#include "poly/tiling/hermes/utils.h"

namespace akg {
namespace ir {
namespace poly {
InitGraph::InitGraph(const std::string &name, const std::vector<std::shared_ptr<Node>> &nodes,
                     const std::vector<std::shared_ptr<Node>> &inputs,
                     const std::vector<std::shared_ptr<Node>> &outputs)
    : name_{name}, nodes_{nodes}, inputs_{inputs}, outputs_{outputs} {}

InitGraph::InitGraph(const std::vector<std::shared_ptr<akg::ir::poly::Node>> &check_visitor_nodes)
    : nodes_{check_visitor_nodes} {
  SetInputNodes();
  SetOutputNodes();
  SetConstantNodes(this->nodes_, inputs_);
  if (!outputs_.empty()) {
    this->name_ = outputs_.back()->name_;
  }
}

void InitGraph::SetConstantNodes(const std::vector<std::shared_ptr<Node>> &nodes,
                                 const std::vector<std::shared_ptr<Node>> &inputs) {
  for (auto const &node : nodes) {
    if (node->op_.IsInput() && (std::find(inputs.begin(), inputs.end(), node) == inputs.end())) {
      node->op_.op_type_ = Op::OpType::Constant;
    }
  }
}

void InitGraph::SetInputNodes() {
  for (auto const &node : nodes_) {
    if (node->op_.IsInput()) {
      inputs_.push_back(node);
    }
  }
}

void InitGraph::SetOutputNodes() {
  std::vector<std::string> predecessor_names;
  for (auto const &node : nodes_) {
    for (auto const &pred : node->pred_) {
      predecessor_names.push_back(pred->name_);
    }
  }
  for (auto const &node : nodes_) {
    for (auto const &succ : node->succ_) {
      if (std::find(outputs_.begin(), outputs_.end(), succ) != outputs_.end()) {
        continue;
      }
      if (std::find(predecessor_names.begin(), predecessor_names.end(), succ->name_) != predecessor_names.end()) {
        continue;
      }
      outputs_.push_back(succ);
    }
  }
  if (outputs_.empty()) {
    outputs_.push_back(nodes_.back());
  }
}

void InitGraph::RemoveNameless() {
  std::set<int> to_remove;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (!(nodes_[i]->HasName())) {
      if (nodes_[i]->op_.RemoveUselessInput()) {  // InplaceAssign
        to_remove = UselessInput(nodes_[i]->pred_, nodes_, to_remove);
      } else {
        FixGraph(nodes_, i);
      }
      to_remove.insert(static_cast<int>(i));
    }
  }

  for (auto nid = to_remove.rbegin(); nid != to_remove.rend(); ++nid) {
    nodes_.erase(nodes_.begin() + (*nid));
  }
}

std::set<int> InitGraph::UselessInput(const std::vector<std::shared_ptr<Node>> &inputs,
                                      const std::vector<std::shared_ptr<Node>> &nodes, std::set<int> to_remove) {
  for (auto const &node : inputs) {
    if ((node->succ_.size() == 1) && node->pred_.empty()) {
      to_remove.insert(IdOfNodeName(node->name_, nodes));
    }
  }
  return to_remove;
}

void InitGraph::FixGraph(std::vector<std::shared_ptr<Node>> nodes, size_t zombie_id) {
  for (auto const &input : nodes[zombie_id]->pred_) {
    for (auto const &output : nodes[zombie_id]->succ_) {
      input->succ_.push_back(output);
      output->pred_.push_back(input);
      for (auto const &out_tensor : nodes[zombie_id]->output_tensors_) {
        auto ipt_tsr = std::find(output->input_tensors_.begin(), output->input_tensors_.end(), out_tensor);
        if (ipt_tsr != output->input_tensors_.end()) {
          output->input_tensors_.erase(ipt_tsr);
        }
      }
      output->input_tensors_.insert(output->input_tensors_.end(), input->output_tensors_.begin(),
                                    input->output_tensors_.end());
    }
  }
}

// Buffers Naming
int InitGraph::IdOfNodeName(const std::string &name, const std::vector<std::shared_ptr<Node>> &nodes) {
  for (size_t i = 0; i < nodes.size(); i++) {
    if (nodes[i]->name_ == name) {
      return static_cast<int>(i);
    }
  }
  LOG(FATAL) << "No node with the name" + name;
  return -1;
}

// True if all inputs of a node were assigned a name
bool AreAllInputsAssigned(std::set<std::shared_ptr<Node>> assigned, std::vector<std::shared_ptr<Node>> inputs) {
  bool is_assigned = true;
  std::for_each(std::begin(inputs), std::end(inputs), [&is_assigned, &assigned](const std::shared_ptr<Node> &node) {
    is_assigned = is_assigned && ((assigned.find(node) != assigned.end()) || node->op_.IsConstant());
  });
  return is_assigned;
}

void InitGraph::AddNodesName(const std::vector<std::string> &names) {
  std::set<std::shared_ptr<Node>> nexts = std::set<std::shared_ptr<Node>>();  // yet to assign
  std::set<std::shared_ptr<Node>> assigned;
  std::string found;
  bool is_buffer_stitch = false;

  if (names.empty()) {
    LOG(FATAL) << "No buffer names given";
  }

  LOG(INFO) << "AddNodesName: " << nodes_.size() << " nodes; " << outputs_.size() << " outputs; " << inputs_.size()
            << " inputs";

  std::set<std::string> intermed_output_names = GetIntermediateOutputsNames(names);
  if (!intermed_output_names.empty()) {
    is_buffer_stitch = true;
    inputs_ = GetInputs(intermed_output_names, nodes_);
  }
  // give tensor name as name for input node
  for (auto const &input_node : inputs_) {
    input_node->name_ = input_node->output_tensors_[0]->name_;
    assigned.insert(input_node);

    for (auto const &node : input_node->succ_) {
      if (std::find(inputs_.begin(), inputs_.end(), node) == inputs_.end()) {
        nexts.insert(node);
      }
    }

    FilterNames(names, input_node->output_tensors_[0]->name_);
  }

  for (auto const &node : nodes_) {
    if (node->op_.IsConstant()) {
      nexts.insert(node->succ_.begin(), node->succ_.end());
    }
  }

  std::shared_ptr<Node> node;

  // Breadth-First Search
  while (!nexts.empty()) {
    auto it1 = std::find_if(nexts.begin(), nexts.end(), [&assigned](const std::shared_ptr<Node> &node) {
      return (AreAllInputsAssigned(assigned, node->pred_));
    });
    if (it1 == nexts.end()) {
      auto it2 = std::find_if(nodes_.begin(), nodes_.end(), [&assigned](const std::shared_ptr<Node> &node) {
        return ((assigned.find(node) == assigned.end()) && (!node->op_.IsInput()) &&
                (AreAllInputsAssigned(assigned, node->pred_)));
      });
      if (it2 == nodes_.end()) {  // raise an error
        LOG(WARNING) << "some nodes are left unassigned\n";
        break;
      }
      node = *it2;
    } else {
      node = *it1;
    }

    nexts.erase(node);

    found = FindName(names, node);
    node->name_ = found;
    assigned.insert(node);

    // add to next only if not assigned already
    std::for_each(node->succ_.begin(), node->succ_.end(), [&assigned, &nexts](const std::shared_ptr<Node> &node) {
      if (assigned.find(node) == assigned.end()) {
        nexts.insert(node);
      }
    });

    FilterNames(names, found);
  }

  if (is_buffer_stitch) {
    for (auto const &input : inputs_) {
      input->name_ = "";
    }
  }
}

std::set<std::string> InitGraph::GetIntermediateOutputsNames(const std::vector<std::string> &names) {
  std::set<std::string> intermed_output_names;
  size_t pos = 0;
  size_t end = 0;
  std::string output;
  for (std::string const &name : names) {
    pos = name.find("output");
    while (pos != std::string::npos) {
      end = name.find('_', pos + 1);
      end = name.find('_', end + 1);
      end = name.find('_', end + 1);
      if (end == std::string::npos) {
        output = name.substr(pos);
      } else {
        output = name.substr(pos, end - pos);
      }
      intermed_output_names.insert(output);
      pos = name.find("output", end);
    }
  }
  return intermed_output_names;
}

std::vector<std::shared_ptr<Node>> InitGraph::GetInputs(std::set<std::string> intermed_output_names,
                                                        const std::vector<std::shared_ptr<Node>> &nodes) {
  std::vector<std::shared_ptr<Node>> n_inputs;
  for (auto const &node : nodes) {
    auto name = intermed_output_names.find(node->output_tensors_[0]->name_);
    if (name != intermed_output_names.end()) {
      n_inputs.push_back(node);
      node->op_.op_type_ = Op::OpType::Input;
      intermed_output_names.erase(name);
    }
  }
  return n_inputs;
}

bool InputsAreInside(std::vector<std::shared_ptr<Node>> inputs, const std::string &name) {
  size_t pos = 0;

  for (size_t i = 0; i < inputs.size(); i++) {
    std::string input = StripRename(inputs[i]->name_);
    if (i > 1) {
      return true;
    }

    if ((name.find(input, pos) == std::string::npos) &&
        (name.find(inputs[i]->output_tensors_[0]->name_, pos) == std::string::npos)) {
      return false;
    }
    pos++;
  }
  return true;
}

// find the name of a node
std::string InitGraph::FindName(std::vector<std::string> names, const std::shared_ptr<Node> &node) {
  std::vector<std::string> possibles;
  std::vector<std::shared_ptr<Node>> inputs;
  if (!node->op_.IsLonely()) {  // ie Exp does't have its input written
    inputs = node->pred_;
  }
  bool cst_input = HasConstantInput(node);
  std::copy_if(names.begin(), names.end(), std::back_inserter(possibles),
               [&inputs, &node, &cst_input](const std::string &name) {
                 bool is_inside = InputsAreInside(inputs, name);
                 bool reduceName = !(name.find("red", name.size() - 4) == std::string::npos);
                 reduceName = (node->op_.IsReduce()) ? reduceName : !reduceName;
                 return (is_inside && reduceName && node->op_.FitBufferName(name, cst_input));
               });

  std::sort(possibles.begin(), possibles.end(),  // avoid intermediate buffer
            [](const std::string &str_a, const std::string &str_b) { return (str_a.size() < str_b.size()); });

  if (possibles.empty()) {
    LOG(INFO) << "Careful: No suitable name left for node " << node->op_.ToString();
    return "";  // Accepted because nodes may be simplified (e.g. RealDiv(x, 1))
  }
  return possibles[0];
}

bool InitGraph::HasConstantInput(const std::shared_ptr<Node> &node) {
  bool all_consts = true;
  for (size_t i = 0; i < node->pred_.size(); ++i) {
    all_consts = all_consts && (node->pred_[i]->op_.IsConstant());
  }
  return all_consts;
}

// remove name assigned & similar ones (with local_UB.*)
void InitGraph::FilterNames(std::vector<std::string> names, const std::string &out) {
  std::vector<int> to_remove;
  int size = static_cast<int>(names.size());
  for (int i = size - 1; i >= 0; --i) {
    if (names[i] == out) {
      to_remove.push_back(i);
    }
  }

  std::for_each(std::begin(to_remove), std::end(to_remove), [&names](int i) { names.erase(names.begin() + i); });
}

std::string InitGraph::ToString() {
  std::stringstream buf;
  buf << "{ name = " << this->name_;
  for (size_t i = 0; i < this->nodes_.size(); i++) {
    buf << "{node:" << nodes_[i] << "}" << std::endl;
  }
  return buf.str();
}

Op::OpCategory InitGraph::OperatorCategory() {
  Op::OpCategory result = Op::OpCategory::Input;
  for (auto const &node : nodes_) {
    result = Op::DominantCategory(result, node->op_.Category());
  }
  return result;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
