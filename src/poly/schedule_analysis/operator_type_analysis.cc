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
#include "operator_type_analysis.h"
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
        CHECK(static_cast<int>(lhs.indices.size()) > erase_idx);
        lhs.indices.erase(lhs.indices.begin() + erase_idx);
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

void OpTypeCollector::Collect() { this->Visit(stmt_); }

void OpTypeCollector::AnalyzeOpTemplate() {
  std::string concated_op_type;
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      concated_op_type += pe.basic_op_type + ",";
      if (scop_info_.user_config_.GetTarget() == TARGET_CUDA &&
          pe.basic_op_type.find(AT_TRANSPOSE) != std::string::npos &&
          pe.basic_op_type.find(AT_ELEMWISE) != std::string::npos) {
        AnalyzeGemmAxes(pe);
      }
    }
  }
  if (scop_info_.analysis_result_.GetOpTemplate() != Template::DEFAULT) {
    return;
  }

  if (concated_op_type.find(AT_REDUCE) != std::string::npos) {
    scop_info_.analysis_result_.SetOpTemplate(Template::REDUCTION);
  } else if (concated_op_type.find(AT_TRANSPOSE) != std::string::npos) {
    scop_info_.analysis_result_.SetOpTemplate(Template::TRANSPOSE_OP);
  } else if (concated_op_type.find(AT_PAD) != std::string::npos) {
    scop_info_.analysis_result_.SetOpTemplate(Template::PAD_OP);
  } else if (concated_op_type.find(AT_BROADCAST) != std::string::npos ||
             concated_op_type.find(AT_TRANSFORM) != std::string::npos) {
    scop_info_.analysis_result_.SetOpTemplate(Template::BROADCAST_OP);
  } else if (concated_op_type.find(AT_CALL) != std::string::npos) {
    scop_info_.analysis_result_.SetOpTemplate(Template::EXTERN_CALL);
  } else {
    scop_info_.analysis_result_.SetOpTemplate(Template::PURE_ELEM);
  }
}

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
            << scop_info_.analysis_result_.ShowOpTemplate(scop_info_.analysis_result_.GetOpTemplate());
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
  prov.band_index = band_count_;
  prov.src = src_tensor;
  prov.dst = dst_tensor;
  prov.op = op;
  prov.cond = cur_if_;
  provides_ana_[cur_loop_].emplace_back(prov);
}

void OpTypeCollector::AnalyzeGemmAxes(const ProvideEntry &pe) {
  VarNames mx_c, mx_a, mx_b;
  int index_a = -1;
  int index_b = -1;
  auto EmplaceVarsInTensor = [](TensorEntry tensor, VarNames &var_list) -> void {
    for (const auto &vars_i : tensor.var_names) {
      for (const auto &name : vars_i) {
        if (IsNum(name)) {
          continue;
        }
        var_list.emplace_back(name);
      }
    }
  };

  // Visit source tensors to fill mx_a and mx_b. Also, we need to check whether this provide stmt
  // is in `C = C + A * B` form and directly return if the form is broken.
  if (pe.src.size() != 3U) {
    return;
  }
  EmplaceVarsInTensor(pe.dst, mx_c);
  bool found_c = false;
  for (size_t i = 0; i < pe.src.size(); ++i) {
    auto src = pe.src[i];
    if (src.name == pe.dst.name) {
      VarNames src_c;
      EmplaceVarsInTensor(src, src_c);
      if (src_c.size() != mx_c.size()) {
        return;
      }
      for (size_t i = 0; i < src_c.size(); ++i) {
        if (src_c[i] != mx_c[i]) {
          return;
        }
      }
      found_c = true;
    } else if (index_a == -1) {
      EmplaceVarsInTensor(src, mx_a);
      index_a = i;
    } else if (index_b == -1) {
      EmplaceVarsInTensor(src, mx_b);
      index_b = i;
    } else {
      return;
    }
  }
  if (!found_c || mx_a.empty()) {
    return;
  }

  // construct relationship between loop indices and loop type(b/m/n/k) and mark axis with corresponding attribute
  std::string attr_key = "";
  if (scop_info_.user_config_.GetEnableConvTensorCore()) {
    scop_info_.analysis_result_.SetOpTemplate(Template::CONV);
  } else {
    scop_info_.analysis_result_.SetOpTemplate(Template::MATMUL);
  }
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
        if (g_csr.empty()) {
          return type + AT_REDUCE;
        } else {
          return type + AT_BROADCAST;
        }
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

class ExtractAxisUsed : public IRVisitor {
 public:
  ExtractAxisUsed(std::vector<const Variable *> &vec, std::unordered_set<std::string> &red) : vec_(vec), red_(red) {}
  ~ExtractAxisUsed() override = default;
  void Visit_(const Variable *op) final {
    if (red_.count(op->name_hint) == 1) {
      vec_.push_back(op);
    }
    IRVisitor::Visit_(op);
  }

  void Run(const NodeRef &ref) { IRVisitor::Visit(ref); }

 private:
  std::vector<const Variable *> &vec_;
  std::unordered_set<std::string> &red_;
};

class ExtractAxisNotUsed : public IRVisitor {
 public:
  ExtractAxisNotUsed(std::vector<const Variable *> &vec, std::unordered_set<std::string> &red) : vec_(vec), red_(red) {}
  ~ExtractAxisNotUsed() override = default;
  void Visit_(const Variable *op) final {
    if (red_.count(op->name_hint) == 0) {
      vec_.push_back(op);
    }
    IRVisitor::Visit_(op);
  }

  void Run(const NodeRef &ref) { IRVisitor::Visit(ref); }

 private:
  std::vector<const Variable *> &vec_;
  std::unordered_set<std::string> &red_;
};

class GetBatchNum final : public IRVisitor {
 public:
  GetBatchNum(const NodeRef node) { IRVisitor::Visit(node); }
  ~GetBatchNum() override = default;

  void Visit_(const Call *op) final {
    if (visited_axis.size() == 0) {
      batch_axis_num = op->args.size();
      for (size_t i = 0; i < op->args.size(); i++) {
        visited_axis.push_back(op->args[i]);
      }
    } else {
      unsigned int same_axis_num = 0;
      for (size_t i = 0; (i < op->args.size()) && (i < visited_axis.size()); i++) {
        if (Equal(op->args[i], visited_axis[i])) {
          same_axis_num++;
        } else {
          break;
        }
      }
      if (batch_axis_num > same_axis_num) batch_axis_num = same_axis_num;
    }
    IRVisitor::Visit_(op);
  }

 public:
  std::vector<Expr> visited_axis;
  unsigned int batch_axis_num{0};
};

class OperatorInfoCollector {
 public:
  OperatorInfoCollector(ScopInfo &scop_info) : scop_info_(scop_info) {}
  ~OperatorInfoCollector() = default;
  void Run() {
    auto op_type = scop_info_.analysis_result_.GetOpTemplate();
    if (op_type == Template::REDUCTION) {
      RecordReduceInfo();
    } else if (op_type == Template::MATMUL || op_type == Template::CONV) {
      RecordMatmulInfo();
    }
  }

  void RecordReduceInfo() {
    for (auto &rm : scop_info_.analysis_result_.GetReduceMap()) {
      auto op = rm.first;
      auto reduce_iter_var = rm.second;
      isl::id red_id;
      for (auto &c : scop_info_.analysis_result_.GetStatementMap()) {
        if (op == c.second) {
          red_id = c.first;
          break;
        }
      }

      OperatorDomainSpace op_domain;
      auto dm = scop_info_.analysis_result_.GetOperatorDomainMap();
      if (dm.count(red_id)) {
        op_domain = dm[red_id];
      }

      std::unordered_set<std::string> reduce_attrs;
      for (auto &r : reduce_iter_var) {
        reduce_attrs.insert(r->var->name_hint);
      }

      auto args = op->args;
      std::vector<const Variable *> vars_not_reduce;
      for (auto &a : args) {
        ExtractAxisNotUsed(vars_not_reduce, reduce_attrs).Run(a);
      }

      bool is_all_reduce = vars_not_reduce.size() == 0;
      scop_info_.user_config_.SetTileCheckCoincident(!is_all_reduce);
      if (is_all_reduce) {
        scop_info_.analysis_result_.SetOpTemplate(Template::ALL_REDUCTION);
      }
      isl::ctx ctx = op_domain.tuple.ctx();

      isl::aff_list aff_list = isl::aff_list(ctx, 0);
      for (auto id : op_domain.tuple.get_id_list()) {
        if (reduce_attrs.count(id.get_name()) == 1) {
          continue;
        }
        isl::aff aff = isl::aff::param_on_domain(op_domain.param_space, id);
        aff = aff.unbind_params_insert_domain(op_domain.tuple);
        aff_list = aff_list.add(aff);
      }
      isl::space op_domain_space = op_domain.tuple.get_space();
      isl::space space = op_domain_space.params().add_named_tuple_id_ui(red_id, aff_list.size());
      space = op_domain_space.product(space).unwrap();
      isl::union_map upa = isl::union_map(isl::map(isl::multi_aff(space, aff_list)));

      ReduceTensorInfo reduce_tensor_info;
      reduce_tensor_info.stmt_node = op;
      reduce_tensor_info.stmt_map = upa;
      scop_info_.analysis_result_.RecordReduceTensorInfoMap(red_id, reduce_tensor_info);
      auto type = scop_info_.analysis_result_.GetReduceOpType(red_id);
      if (type == AKG_REDUCE_AND || type == AKG_REDUCE_OR) {
        scop_info_.analysis_result_.SetOpTemplate(Template::BITWISE_REDUCTION);
      }
      if (AkgSupportedReduceOp.count(type) != 0) {
        reduce_tensor_info.write_tensor_name = op->func->func_name();
        SetReduceInitValue(reduce_tensor_info);
        SetReduceWriteDataType(reduce_tensor_info);
        scop_info_.analysis_result_.UpdateReduceTensorInfoMap(red_id, reduce_tensor_info);

        std::string reduce_direction;
        if (scop_info_.analysis_result_.GetCsr()) {
          reduce_direction = X_DIRECTION;
        } else {
          PostOrderVisit(op->value, [&reduce_direction, &reduce_attrs, op](const NodeRef &node) -> void {
            if (reduce_direction == Y_DIRECTION) {
              return;
            }
            auto call = node.as<Call>();
            if (call == nullptr || call->call_type != Call::CallType::Halide ||
                call->func->func_name() == op->func->func_name() || call->args.empty()) {
              return;
            }
            int call_size = static_cast<int>(call->args.size());
            int reduce_position = -1;
            int non_variable_count = 0;
            bool is_continuous = true;
            for (int i = call_size - 1; i >= 0; --i) {
              auto last_axis = call->args[i];
              auto mod = last_axis.as<FloorMod>();
              auto var = mod != nullptr ? mod->a.as<Variable>() : last_axis.as<Variable>();
              if (var != nullptr) {
                reduce_position = reduce_attrs.count(var->name_hint) ? i : reduce_position;
                is_continuous = false;
              } else if (var == nullptr && is_continuous) {
                ++non_variable_count;
              }
            }
            if (reduce_position == -1) {
              return;
            }

            bool is_all_reduce = true;
            for (int i = 0; i < static_cast<int>(op->args.size()); ++i) {
              if (op->args[i].as<IntImm>() == nullptr || op->args[i].as<IntImm>()->value != 0) {
                is_all_reduce = false;
                break;
              }
            }

            if (is_all_reduce) {
              reduce_direction = ALL_DIRECTION;
              return;
            }

            if (reduce_position == call_size - non_variable_count - 1) {
              reduce_direction = X_DIRECTION;
            } else {
              reduce_direction = Y_DIRECTION;
            }
          });
        }
        if (reduce_direction.empty()) {
          LOG(WARNING) << "Cannot identify reduce direction for stmt " << red_id;
        }
        scop_info_.analysis_result_.RecordReduceDirection(reduce_direction);
        if (scop_info_.user_config_.GetEnableAkgReduceLib()) {
          scop_info_.analysis_result_.SetUseGpuReduceLib(true);
        }
      }
    }
  }

  void RecordMatmulInfo() {
    for (auto &rm : scop_info_.analysis_result_.GetReduceMap()) {
      auto op = rm.first;
      auto reduce_iter_var = rm.second;
      isl::id red_id;
      for (auto &c : scop_info_.analysis_result_.GetStatementMap()) {
        if (op == c.second) {
          red_id = c.first;
          break;
        }
      }

      OperatorDomainSpace op_domain;
      auto dm = scop_info_.analysis_result_.GetOperatorDomainMap();
      if (dm.count(red_id)) {
        op_domain = dm[red_id];
      }

      std::unordered_set<std::string> reduce_attrs;
      for (auto &r : reduce_iter_var) {
        reduce_attrs.insert(r->var->name_hint);
      }

      auto args = op->args;
      std::vector<const Variable *> vars_not_reduce;
      std::vector<const Variable *> vars_reduce;
      for (auto &a : args) {
        ExtractAxisNotUsed(vars_not_reduce, reduce_attrs).Run(a);
      }

      ExtractAxisUsed(vars_reduce, reduce_attrs).Run(op->value);

      GetBatchNum get_batch_num(op->value);
      scop_info_.analysis_result_.RecordNotReduceAxisForMatmul(vars_not_reduce);
      scop_info_.analysis_result_.RecordReduceAxisForMatmul(vars_reduce);
      scop_info_.analysis_result_.RecordBatchAxisNumForMatmul(get_batch_num.batch_axis_num);

      isl::ctx ctx = op_domain.tuple.ctx();

      isl::aff_list aff_list = isl::aff_list(ctx, 0);
      for (auto id : op_domain.tuple.get_id_list()) {
        if (reduce_attrs.count(id.get_name()) == 1) {
          continue;
        }
        isl::aff aff = isl::aff::param_on_domain(op_domain.param_space, id);
        aff = aff.unbind_params_insert_domain(op_domain.tuple);
        aff_list = aff_list.add(aff);
      }
      isl::space op_domain_space = op_domain.tuple.get_space();
      isl::space space = op_domain_space.params().add_named_tuple_id_ui(red_id, aff_list.size());
      space = op_domain_space.product(space).unwrap();
      isl::union_map upa = isl::union_map(isl::map(isl::multi_aff(space, aff_list)));
      bool use_tensor_core = false;
      if (CheckMatmul(op)) {
        // Default vectorization access mode (128 bits).
        if (scop_info_.user_config_.GetVectorLength() == 0 && scop_info_.user_config_.GetTarget() != TARGET_CPU) {
          scop_info_.user_config_.SetVectorLength(PROMOTE_VECTORIZATION_BIT);
        }
        RecordMatrixInfoForFuse(op);
        use_tensor_core = true;
      }
      scop_info_.user_config_.SetEnableMatmul(use_tensor_core);
      scop_info_.user_config_.SetEnableTensorCore(use_tensor_core);
      scop_info_.user_config_.SetEnableTensorCoreUsePoly(use_tensor_core);
    }
  }

  void RecordMatrixInfoForFuse(const Provide *op) {
    auto matmul_map = scop_info_.analysis_result_.GetMatrixMatmulMap();
    if (!matmul_map.empty()) {
      std::string accumulator = "";
      auto mp = GetMatmulTensorsName(scop_info_);
      if (mp.find(MATRIX_C) != mp.end()) {
        accumulator = mp[MATRIX_C];
      }
      CHECK(accumulator != "") << "MatMul info not enough!";
      Array<Expr> elem_tensors = GetBinaryOpExprChildren(op->value);
      if (!elem_tensors.empty()) {
        auto left = elem_tensors[0].as<Call>();
        auto right = elem_tensors[1].as<Call>();
        if ((left || right) &&
            (matmul_map.find(left->name) != matmul_map.end() || matmul_map.find(right->name) != matmul_map.end())) {
          if (op->func->func_name() != accumulator) {
            scop_info_.analysis_result_.RecordMatrixMatmulMap(op->func->func_name(), MATRIX_ELSE);
            scop_info_.analysis_result_.RecordMatrixMatmulMajor(op->func->func_name(), ROW_MAJOR);
          }
          if (left && left->name != accumulator) {
            scop_info_.analysis_result_.RecordMatrixMatmulMap(left->name, MATRIX_ELSE);
            scop_info_.analysis_result_.RecordMatrixMatmulMajor(left->name, ROW_MAJOR);
          }
          if (right && right->name != accumulator) {
            scop_info_.analysis_result_.RecordMatrixMatmulMap(right->name, MATRIX_ELSE);
            scop_info_.analysis_result_.RecordMatrixMatmulMajor(right->name, ROW_MAJOR);
          }
        }
      }
    }
  }
  void SetReduceWriteDataType(ReduceTensorInfo &reduce_tensor_info) {
    auto init_value = reduce_tensor_info.init_value;
    if (!init_value.defined()) {
      return;
    }
    reduce_tensor_info.write_dtype = init_value.type();
  }

  void SetReduceInitValue(ReduceTensorInfo &reduce_tensor_info) {
    Expr init_value;
    if (!reduce_tensor_info.stmt_node->IsInstance<Provide>()) {
      return;
    }
    auto provide = static_cast<const Provide *>(reduce_tensor_info.stmt_node);
    if (provide == nullptr) {
      return;
    }
    auto red_tensor_name = provide->func->func_name();
    for (auto it : scop_info_.analysis_result_.GetStatementMap()) {
      if (it.second->IsInstance<Provide>()) {
        auto prev_provide = static_cast<const Provide *>(it.second);
        if (prev_provide == nullptr || prev_provide == provide || prev_provide->func->func_name() != red_tensor_name) {
          continue;
        }
        init_value = prev_provide->value;
        scop_info_.analysis_result_.RecordReduceInitIds(it.first);
        break;
      }
    }
    if (!init_value.defined()) {
      return;
    }
    reduce_tensor_info.init_value = init_value;
  }

  bool GetRowColInfo(const Provide *op) {
    auto axis = scop_info_.analysis_result_.GetNotReduceAxisForMatmul();
    auto reduce_axis = scop_info_.analysis_result_.GetReduceAxisForMatmul();
    auto batch_num_axis = scop_info_.analysis_result_.GetBatchAxisNumForMatmul();
    if (axis.size() < 2 || reduce_axis.size() < 1 || axis.size() <= batch_num_axis) return false;

    const Variable *axis_var[2];
    const Variable *reduce_axis_var;
    axis_var[0] = axis[batch_num_axis];
    axis_var[1] = axis.back();
    reduce_axis_var = reduce_axis.back();

    class CollectInfoOfBody : public IRVisitor {
     public:
      CollectInfoOfBody() {}
      using IRVisitor::Visit_;

      void Visit_(const Call *op) final {
        IRVisitor::Visit_(op);
        args_.insert(std::make_pair(op->name, op->args));
      }

      std::unordered_map<std::string, Array<Expr>> GetArgs() { return args_; }

     private:
      std::unordered_map<std::string, Array<Expr>> args_;
    } collect_info_of_body;

    auto right = op->value;
    auto add_op = right.as<Add>();
    CHECK(add_op);
    auto tensor_c = add_op->a.as<Call>();
    if (tensor_c == nullptr) return false;

    Type tensor_c_type;
    if (!IsExistTensor(tensor_c->name, tensor_c_type)) return false;

    collect_info_of_body.Visit(add_op->b);

    for (auto iter : collect_info_of_body.GetArgs()) {
      auto name = iter.first;
      auto args = iter.second;
      if (args.size() < 2) continue;

      const Variable *var0 = args[batch_num_axis].as<Variable>();
      const Variable *var1 = args[args.size() - 1].as<Variable>();
      if (var0 == nullptr || var1 == nullptr) continue;

      std::string major;
      if ((var0 == reduce_axis_var) && (var1 == axis_var[0])) {
        major = COL_MAJOR;
      } else if ((var0 == reduce_axis_var) && (var1 == axis_var[1])) {
        major = ROW_MAJOR;
      } else if ((var0 == axis_var[0]) && (var1 == reduce_axis_var)) {
        major = ROW_MAJOR;
      } else if ((var0 == axis_var[1]) && (var1 == reduce_axis_var)) {
        major = COL_MAJOR;
      } else {
        return false;
      }
      scop_info_.analysis_result_.RecordMatrixMatmulMajor(name, major);
    }
    scop_info_.analysis_result_.RecordMatrixMatmulMajor(op->func->func_name(), ROW_MAJOR);
    return true;
  }

  bool IsExistTensor(const std::string &tensor_name, Type &tensor_type) {
    auto all_tensors = scop_info_.user_config_.GetRealizeTensors();
    for (auto it : all_tensors) {
      if (it->op->name == tensor_name) {
        tensor_type = it->dtype;
        return true;
      }
    }
    auto orig_binds = scop_info_.user_config_.GetOriginBind();
    for (auto it : orig_binds) {
      if (it.first->op->name == tensor_name) {
        tensor_type = it.first->dtype;
        return true;
      }
    }
    return false;
  }

  std::string GetTensorName(Expr tensor_data, bool &enable_tensor_core) {
    std::string tensor_name = "";
    if (tensor_data.as<Call>()) {
      auto tensor_data_p = tensor_data.as<Call>();
      Type tensor_type;
      if (!IsExistTensor(tensor_data_p->name, tensor_type)) {
        return tensor_name;
      }
      if ((tensor_type != Float(16)) && (tensor_type != Int(8))) {
        enable_tensor_core = false;
      }
      tensor_name = tensor_data_p->name;
    } else if (tensor_data.as<Cast>() &&
               ((tensor_data.as<Cast>()->type == Float(16)) || (tensor_data.as<Cast>()->type == Int(8)))) {
      auto tensor_data_p = tensor_data.as<Cast>();
      auto value = tensor_data_p->value;
      tensor_name = value.as<Call>()->name;
      scop_info_.analysis_result_.RecordCastTensors(tensor_name);
    }
    return tensor_name;
  }

  bool CheckMatmul(const Provide *op) {
    if (!scop_info_.user_config_.GetEnableMatmul()) {
      return false;
    }

    // C + A * B
    auto add_op = op->value.as<Add>();
    if (add_op == nullptr) {
      return false;
    }

    auto tensor_c = add_op->a.as<Call>();
    if (tensor_c == nullptr) {
      return false;
    }
    Type tensor_c_type;
    if (!IsExistTensor(tensor_c->name, tensor_c_type)) {
      return false;
    }
    if (tensor_c_type != Float(16) && tensor_c_type != Float(32) && tensor_c_type != Int(32)) {
      return false;
    }

    auto mul_op = akg::common::SplitCast(add_op->b, tensor_c_type).as<Mul>();
    if (mul_op == nullptr) {
      return false;
    }

    auto tensor_a = akg::common::SplitCast(mul_op->a, tensor_c_type);
    auto tensor_b = akg::common::SplitCast(mul_op->b, tensor_c_type);
    bool enable_tensor_core = true;
    std::string tensor_a_name = GetTensorName(tensor_a, enable_tensor_core);
    std::string tensor_b_name = GetTensorName(tensor_b, enable_tensor_core);
    if (!enable_tensor_core) {
      return false;
    }

    if (tensor_a_name.empty() || tensor_b_name.empty()) {
      return false;
    }

    scop_info_.analysis_result_.RecordMatrixMatmulMap(tensor_a_name, MATRIX_A);
    scop_info_.analysis_result_.RecordMatrixMatmulMap(tensor_b_name, MATRIX_B);
    scop_info_.analysis_result_.RecordMatrixMatmulMap(tensor_c->name, MATRIX_C);

    bool ret = GetRowColInfo(op);
    if (!ret) {
      return false;
    }

    SetMmaModeForTensor(tensor_a_name, tensor_b_name);

    if (tensor_c_type == Float(16)) {
      std::string shared_tensors = tensor_a_name + " " + tensor_b_name + " " + tensor_c->name;
      scop_info_.user_config_.SetSharedTensors(shared_tensors);
    }

    return true;
  }

  void SetMmaModeForTensor(const std::string &tensor_a_name, const std::string &tensor_b_name) {
    std::string custom_dim = scop_info_.user_config_.GetBDim();
    if (!custom_dim.empty() && !scop_info_.user_config_.GetEnableConvTensorCore()) {
      const int each_axis_size = 4;
      const int m_axis_pos = 1;
      const int n_axis_pos = 2;
      const int k_axis_pos = 3;

      Mma mma;
      std::vector<std::string> dim_str = Split(custom_dim, " ");
      int batch_number = static_cast<int>(scop_info_.analysis_result_.GetBatchAxisNumForMatmul()) > 0 ? 1 : 0;
      int real_m_axis_pos = (m_axis_pos + batch_number) * each_axis_size - 1;
      int real_n_axis_pos = (n_axis_pos + batch_number) * each_axis_size - 1;
      int real_k_axis_pos = (k_axis_pos + batch_number) * each_axis_size - 1;
      mma.m = static_cast<int>(WrappedStrtol(dim_str[real_m_axis_pos]));
      mma.n = static_cast<int>(WrappedStrtol(dim_str[real_n_axis_pos]));
      mma.k = static_cast<int>(WrappedStrtol(dim_str[real_k_axis_pos]));

      scop_info_.analysis_result_.SetMmaMode(mma);
      return;
    }

    Mma mma;
    auto matrix_a_major = scop_info_.analysis_result_.GetMatrixMatmulMajor()[tensor_a_name];
    auto matrix_b_major = scop_info_.analysis_result_.GetMatrixMatmulMajor()[tensor_b_name];
    if (matrix_a_major == COL_MAJOR && matrix_b_major == ROW_MAJOR) {
      mma = {32, 32, 4};
    } else {
      mma = {16, 16, 8};
    }
    scop_info_.analysis_result_.SetMmaMode(mma);
  }

 private:
  ScopInfo &scop_info_;
};

void OpTypeCollector::Run() {
  Collect();
  AnalyzeOpTemplate();
  WriteToScopInfo();
  Dump();
  if (target_ == TARGET_CUDA || target_ == TARGET_CPU) {
    OperatorInfoCollector op_info_coll(scop_info_);
    op_info_coll.Run();
  }
}

}  // namespace poly
}  // namespace ir
}  // namespace akg