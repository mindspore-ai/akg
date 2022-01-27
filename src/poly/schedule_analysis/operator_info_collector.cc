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
#include "operator_info_collector.h"
#include "poly/schedule_tree_util.h"
#include "poly/scop_builder.h"

namespace akg {
namespace ir {
namespace poly {

class CollectToTTensor : public IRVisitor {
 public:
  void Visit_(const Variable *op) final {
    loop_vars_.insert(op->name_hint);
  }

  void Visit_(const IntImm *op) final {
    if (scope_halide_) {
      indices_.push_back(op->value);
    }
  }

  void Visit_(const Call *op) final {
    if (op->call_type == Call::PureIntrinsic) {
      if (op->name == "with") {
        int size = static_cast<int>(op->args.size());
        CHECK(size);
        if (auto sub_op = op->args[0].as<Call>()) {
          if (sub_op->name == "lhs" && !scope_collect_) {
            CHECK(sub_op->args.size() == 3);
            RemoveRewriteIdx_(sub_op->args[1]);
            CollectLHS_(sub_op->args);
            for (int ii = 1; ii < size; ++ii) {
              IRVisitor::Visit(op->args[ii]);
            }
            return;
          } else if (sub_op->name == "rhs") {
            CHECK(sub_op->args.size() == 3);
            if (auto idx = sub_op->args[1].as<IntImm>()) {
              rhs_idx_.insert(idx->value);
            }
            if (!scope_collect_) {
              Array<Expr> arr;
              // orig call containing tensor name is visited first
              arr.push_back(op->args[size - 1]);
              for (int ii = 0; ii < size - 1; ++ii) {
                arr.push_back(op->args[ii]);
              }
              CollectRHS_(arr, "");
              return;
            }
          }
        }
      } else if (op->name == "orig") {
        CHECK(static_cast<int>(op->args.size()));
        auto sub_op = op->args[0].as<Call>();
        if (sub_op != nullptr && sub_op->call_type == Call::Halide && scope_collect_ && scope_rhs_) {
          if (tensor_name_.empty()) {
            tensor_name_ = sub_op->func->func_name();
          }
          scope_halide_ = true;
          for (int ii = 0; ii < static_cast<int>(sub_op->args.size()); ++ii) {
            if (rhs_idx_.count(ii) == 0) {
              IRVisitor::Visit(sub_op->args[ii]);
            }
          }
          scope_halide_ = false;
          return;
        }
      }
    } else if (op->call_type == Call::Halide) {
      scope_halide_ = true;
      if (!scope_collect_) {
        CollectRHS_(op->args, op->func->func_name());
        scope_halide_ = false;
        return;
      }
    }
    for (auto e: op->args) {
      IRVisitor::Visit(e);
    }
    scope_halide_ = false;
  }

  void Visit_(const Provide *op) final {
    lhs.name = op->func->func_name();
    for (int ii = 0; ii < static_cast<int>(op->args.size()); ++ii) {
      if (auto idx = op->args[ii].as<IntImm>()) {
        lhs_idx_[idx->value] = ii;
      }
    }
    CollectLHS_(op->args);
    IRVisitor::Visit(op->value);
  }

  void Visit_(const Select *op) final {
    IRVisitor::Visit(op->true_value);
    IRVisitor::Visit(op->false_value);
  }

  ToTTensor lhs;
  std::vector<ToTTensor> rhs;

 private:
  void RemoveRewriteIdx_(const Expr &e) {
    if (auto idx = e.as<IntImm>()) {
      if (lhs_idx_.count(idx->value) > 0) {
        int erase_idx = lhs_idx_[idx->value];
        if (static_cast<int>(lhs.indices.size()) > erase_idx) {
          lhs.indices.erase(lhs.indices.begin() + erase_idx);
        }
      }
    }
  }

  void CollectLoopVars_(Array<Expr> args) {
    loop_vars_.clear();
    indices_.clear();
    tensor_name_ = "";
    scope_collect_ = true;
    for (auto e: args) {
      IRVisitor::Visit(e);
    }
    scope_collect_ = false;
  }

  void CollectLHS_(Array<Expr> args) {
    CollectLoopVars_(args);
    lhs.loop_vars.insert(loop_vars_.begin(), loop_vars_.end());
    std::copy(indices_.begin(), indices_.end(), std::back_inserter(lhs.indices));
  }

  void CollectRHS_(Array<Expr> args, std::string func_name) {
    ToTTensor rhs_tensor;
    scope_rhs_ = true;
    CollectLoopVars_(args);
    scope_rhs_ = false;
    rhs_idx_.clear();
    rhs_tensor.name = func_name.empty()? tensor_name_ : func_name;
    rhs_tensor.loop_vars.insert(loop_vars_.begin(), loop_vars_.end());
    rhs_tensor.indices.assign(indices_.begin(), indices_.end());
    rhs.push_back(rhs_tensor);
  }

  bool scope_collect_{false};
  bool scope_halide_{false};
  bool scope_rhs_{false};
  std::string tensor_name_;
  std::set<std::string> loop_vars_;
  std::vector<int64_t> indices_;
  std::unordered_map<int, int64_t> lhs_idx_;
  std::unordered_set<int> rhs_idx_;
};

class CheckContainsConst : public IRVisitor {
  void Visit_(const Call *op) final {
    return;
  }

  void Visit_(const Select *op) final {
    IRVisitor::Visit(op->true_value);
    IRVisitor::Visit(op->false_value);
  }

  void Visit_(const IntImm *op) final {
    contains_const_ = true;
  }

 public:
  bool contains_const_{false};
};

void OpTypeCollector::Collect() { this->Visit(stmt_); }

void OpTypeCollector::WriteToScopInfo() {
  for (auto it : provides_ana_) {
    auto cur_loop = it.first;
    auto provs = it.second;
    for (auto prov : provs) {
      scop_info_.analysis_result_.RecordProvideAnalysis(cur_loop, prov);
    }
  }
}

void OpTypeCollector::Dump() {
  LOG(INFO) << "OP TEMPLATE = "
            << scop_info_.analysis_result_.ShowOpTemplate();
  for (auto it : provides_ana_) {
    auto loop = it.first;
    auto provs = it.second;
    LOG(INFO) << "Under loop " << loop->loop_var->name_hint;
    for (auto prov : provs) {
      LOG(INFO) << "[DST] " << prov.dst.name << " [OPTYPE] " << prov.basic_op_type;
    }
  }
}

void OpTypeCollector::Visit_(const AttrStmt *op) {
  cur_attr_ = op;
  IRVisitor::Visit_(op);
}

void OpTypeCollector::Visit_(const Realize *op) {
  local_buf_.insert(op->func->func_name());
  IRVisitor::Visit_(op);
}

void OpTypeCollector::Visit_(const Provide *op) {
  AnalyzeProvide(op);
  IRVisitor::Visit_(op);
}

void OpTypeCollector::Visit_(const For *op) {
  loop_count_ += 1;
  cur_loop_ = op;
  cur_band_.emplace_back(cur_loop_);
  IRVisitor::Visit_(op);
  cur_loop_ = op;
  loop_count_ -= 1;
  // end of an outer band
  if (loop_count_ == 0) {
    band_count_ += 1;
    cur_band_.clear();
  }
}

void OpTypeCollector::Visit_(const IfThenElse *op) {
  cur_if_ = op;
  IRVisitor::Visit_(op);
  cur_if_ = op;
}

TensorEntry OpTypeCollector::MakeTensorEntry(const ToTTensor &tot) {
  TensorEntry tensor;
  tensor.name = tot.name;
  for (std::string var: tot.loop_vars) {
    tensor.args.push_back(Expr(var));
    VarNames vname;
    vname.push_back(var);
    tensor.var_names.push_back(vname);
  }
  for (int idx: tot.indices) {
    tensor.args.push_back(Expr(idx));
  }
  tensor = MatchLoopByName(tensor);
  return tensor;
}

const Call *GetAtomicRhs(const Array<Expr> &args) {
  for (auto e: args) {
    auto ref_call = e.as<Call>();
    if (ref_call != nullptr && ref_call->name == "&" && static_cast<int>(ref_call->args.size()) == 1) {
      auto atomic_rhs = ref_call->args[0].as<Call>();
      if (atomic_rhs != nullptr && atomic_rhs->call_type == Call::CallType::Halide) {
        return atomic_rhs;
      }
    }
  }
  return nullptr;
}

void OpTypeCollector::Visit_(const Evaluate *op) {
  if (scop_info_.analysis_result_.GetOpTemplate() != Template::COUNT_OP) return;
  if (auto call = op->value.as<Call>()) {
    if (call->name.find(AKG_REDUCE_RETURN_NAME) != std::string::npos) {
      auto atomic_rhs = GetAtomicRhs(call->args);
      CHECK(atomic_rhs);
      count_op_tensor_.name = call->name;
      for (auto arg: atomic_rhs->args) {
        VarNames vname;
        vname = VisitVarNames(arg, vname);
        count_op_tensor_.var_names.emplace_back(vname);
      }
      count_op_tensor_ = MatchLoopByName(count_op_tensor_);
      count_op_tensor_.args = call->args;
      count_op_tensor_.band_index = band_count_;
      count_op_tensor_.type_byte = call->type.bytes();
    }
  }
}

void OpTypeCollector::AnalyzeProvide(const Provide *op) {
  if (cur_loop_ == nullptr) return;
  ProvideEntry prov;
  std::string basic_op_type = "";
  std::vector<TensorEntry> src_tensor;
  TensorEntry dst_tensor;
  std::vector<const Call *> src_call;
  auto GetSrc = [&, this](const NodeRef &op) {
    if (const auto call = op.as<Call>()) {
      if (call->call_type == Call::Extern && scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
        basic_op_type = AT_CALL;
      } else {
        src_call.emplace_back(call);
      }
    } else if (op.as<Select>() && scop_info_.user_config_.GetTarget() == TARGET_CCE) {
      basic_op_type = AT_CALL;
    }
  };
  air::ir::PostOrderVisit(op->value, GetSrc);

  for (auto call : src_call) {
    TensorEntry tensor;
    tensor.name = call->name;
    // get variable names
    for (auto arg : call->args) {
      VarNames vname;
      vname = VisitVarNames(arg, vname);
      tensor.var_names.emplace_back(vname);
    }
    tensor = MatchLoopByName(tensor);
    tensor.args = call->args;
    tensor.band_index = band_count_;
    tensor.type_byte = call->type.bytes();
    src_tensor.emplace_back(tensor);
  }

  auto src_length = static_cast<int>(src_tensor.size());
  for (auto st : src_tensor) {
    if (st.name == "mad" || st.name == LOAD_IM2COL) {
      basic_op_type = "SP_CALL";
    }
  }

  if (scop_info_.user_config_.GetTarget() == TARGET_CCE && src_length > 2 && basic_op_type != "SP_CALL") {
    LOG(WARNING) << "Detect provide has " << src_tensor.size() << " source tensors.";
    LOG(WARNING) << "After ToThreeAddress pass, number of ops in src should be less than 2.";
  }
  dst_tensor.name = op->func->func_name();
  for (auto arg : op->args) {
    VarNames vname;
    vname = VisitVarNames(arg, vname);
    dst_tensor.var_names.emplace_back(vname);
  }
  dst_tensor = MatchLoopByName(dst_tensor);
  dst_tensor.args = op->args;
  dst_tensor.band_index = band_count_;
  dst_tensor.type_byte = scop_info_.user_config_.GetDataBytes(dst_tensor.name);

  if (scop_info_.analysis_result_.GetTensorOfTensor()) {
    auto tot_tensors = CollectToTTensor();
    tot_tensors.Visit_(op);
    TensorEntry lhs = MakeTensorEntry(tot_tensors.lhs);
    std::vector<TensorEntry> rhs;
    for (auto rhs_tensor: tot_tensors.rhs) {
      rhs.push_back(MakeTensorEntry(rhs_tensor));
    }
    basic_op_type = GetBasicOpType(lhs, rhs);
  }

  prov.basic_op_type = basic_op_type.empty() ? GetBasicOpType(dst_tensor, src_tensor) : basic_op_type;

  auto check_contains_const = CheckContainsConst();
  check_contains_const.Visit(op->value);
  if (prov.basic_op_type.find(AT_ELEMWISE) != std::string::npos && check_contains_const.contains_const_ &&
      dst_tensor.loops.size() > 0) {
    prov.basic_op_type += AT_COUNT;
  }

  prov.band_index = band_count_;
  prov.src = src_tensor;
  prov.dst = dst_tensor;
  prov.op = op;
  prov.cond = cur_if_;
  provides_ana_[cur_loop_].emplace_back(prov);
}

// Match variable to loop by name since the name in current band is unique.
// If name is not unique, it means the axis is separated into different chunks
// and they will need same alignment rule.
TensorEntry OpTypeCollector::MatchLoopByName(TensorEntry tensor) {
  std::unordered_map<size_t, std::vector<const For *>> loop_pos;
  for (size_t p = 0; p < tensor.var_names.size(); ++p) {
    for (auto name : tensor.var_names[p]) {
      for (auto i = static_cast<int>(cur_band_.size()) - 1; i >= 0; i--) {
        const For *loop = cur_band_[i];
        if (loop != nullptr && loop->loop_var.get()->name_hint == name) {
          loop_pos[p].emplace_back(loop);
          break;
        }
      }
    }
  }
  tensor.loops = loop_pos;
  return tensor;
}

std::string OpTypeCollector::GetBasicOpType(const TensorEntry dst, const std::vector<TensorEntry> &srcs) {
  auto CountUniqueLoopName = [](std::vector<VarNames> var_names) -> size_t {
    std::unordered_set<std::string> uni_name;
    for (auto names : var_names) {
      for (auto n : names) {
        if (IsNum(n)) {
          continue;
        }
        uni_name.insert(n);
      }
    }
    return uni_name.size();
  };

  auto GetSingleOpType = [&CountUniqueLoopName, this](const TensorEntry d, const TensorEntry s) -> std::string {
    auto dst_vars = d.var_names;
    auto src_vars = s.var_names;
    auto dst_vars_size = CountUniqueLoopName(dst_vars);
    auto src_vars_size = CountUniqueLoopName(src_vars);
    std::string type = "";
    if (scop_info_.user_config_.GetTarget() == TARGET_CCE) {
      if (this->local_buf_.find(s.name) == this->local_buf_.end()) type += "DMA2_";
      if (this->local_buf_.find(d.name) == this->local_buf_.end()) type += "DMA3_";
    }
    if (src_vars_size == 0 && scop_info_.user_config_.GetTarget() != TARGET_CUDA) {
      return type + "SP_CALL";
    }
    if (dst_vars_size < src_vars_size) {
      if (d.loops.size() < s.loops.size() && d.name != s.name) {
        // A[i,0] = B[i,j]
        return type + AT_REDUCE;
      } else {
        return type + "UNKNOWN";
      }
    } else if (dst_vars_size > src_vars_size) {
      // A[i,j] = B[i,0]
      return type + AT_BROADCAST;
    } else {
      // Index size is the same.
      while (!dst_vars.empty() && !src_vars.empty()) {
        // detect transpose first
        VarNames dst_name = dst_vars.back();
        VarNames src_name = src_vars.back();
        dst_vars.pop_back();
        src_vars.pop_back();
        VarNames dst_pure_name;
        VarNames src_pure_name;
        for (auto n : dst_name) {
          if (!IsNum(n)) {
            dst_pure_name.emplace_back(n);
          }
        }
        for (auto n : src_name) {
          if (!IsNum(n)) {
            src_pure_name.emplace_back(n);
          }
        }
        if (dst_pure_name.size() == src_pure_name.size()) {
          for (size_t j = 0; j < dst_pure_name.size(); ++j) {
            if (dst_pure_name[j] != src_pure_name[j]) {
              return type + AT_TRANSPOSE;
            }
          }
        }
      }
      if (d.loops.size() == s.loops.size()) {
        // A[i,j] = B[i,j]
        return type + AT_ELEMWISE;
      } else {
        // AutoFused case in cuda
        if (scop_info_.user_config_.GetTarget() == TARGET_CUDA && d.args.size() == s.args.size()) {
          for (size_t i = 0; i < d.args.size(); ++i) {
            if (Equal(d.args[i], s.args[i])) {
              continue;
            }
            if (arith_ana_.CanProve(s.args[i] == 0)) {
              // A[floordiv(i, 128), floormod(i, 128)] = B[0, floormod(i, 128)]
              type += AT_BROADCAST;
            } else if (arith_ana_.CanProve(d.args[i] == 0)) {
              // A[0, floormod(i, 128)] = B[floordiv(i, 128), floormod(i, 128)]
              type += AT_REDUCE;
            } else {
              type += AT_TRANSFORM;
            }
            type += ("|" + std::to_string(i) + "_");
          }
        } else {
          // A[0,i] = B[i,i]
          type += AT_TRANSFORM;
        }
        return type;
      }
    }
    return type;
  };

  std::string basic_op_type = "";
  bool is_unpad = (dst.name.find("unpad") != std::string::npos) || (dst.name.find("Unpad") != std::string::npos);
  bool is_pad = (dst.name.find("pad") != std::string::npos || dst.name.find("Pad") != std::string::npos);
  if (!is_unpad && is_pad) {
    basic_op_type += AT_PAD;
    basic_op_type += "_";
  }
  if (srcs.empty()) {
    // Dst = const
    if (this->local_buf_.find(dst.name) == this->local_buf_.end()) basic_op_type += "DMA3_";
    basic_op_type += "INIT";
  } else {
    for (auto s : srcs) {
      basic_op_type += GetSingleOpType(dst, s);
      basic_op_type += "_";
    }
  }
  return basic_op_type;
}

void OpTypeCollector::Run() {
  Collect();
  WriteToScopInfo();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg