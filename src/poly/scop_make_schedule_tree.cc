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

#include "pass/utils.h"
#include "construct_poly_accesses.h"
#include "poly/scop_builder.h"
#include "poly/schedule_tree_util.h"

namespace akg {
namespace ir {
namespace poly {
class ExtractCond : protected IRVisitor {
 public:
  ExtractCond() {}
  ~ExtractCond() override = default;
  std::vector<Expr> run(const Expr expr) {
    IRVisitor::Visit(Simplify_cce(expr));
    return result;
  }
  bool hasBothOrAndAnd() const { return (or_num && and_num); }
  bool IsOr() {
    if (!or_num && !and_num && result.size() > 1) {
      LOG(INFO) << "  result.size() > 1 and or(and)_num = 0";
    }
    return (or_num > 0);
  }
  std::vector<Expr> result;

 protected:
#define COMOP_VISIT_(OP)                    \
  void Visit_(const OP *op) final {         \
    has_tensor = false;                     \
    this->Visit(op->a);                     \
    this->Visit(op->b);                     \
    if (!has_tensor) {                      \
      Expr expr(GetRef<Expr>(op));          \
      result.push_back(Simplify_cce(expr)); \
    }                                       \
  }
  COMOP_VISIT_(EQ)
  COMOP_VISIT_(NE)
  COMOP_VISIT_(LT)
  COMOP_VISIT_(LE)
  COMOP_VISIT_(GT)
  COMOP_VISIT_(GE)

  void Visit_(const Call *op) final {
    IRVisitor::Visit_(op);
    if (op->call_type == Call::Halide) {
      has_tensor = true;
    }
  }
  void Visit_(const And *op) final {
    and_num++;
    this->Visit(op->a);
    this->Visit(op->b);
  }

  void Visit_(const Or *op) final {
    or_num++;
    this->Visit(op->a);
    this->Visit(op->b);
  }

  void Visit_(const Not *op) final {
    Expr expr(GetRef<Expr>(op));
    LOG(FATAL) << expr << " so far NOT is handled, please modify DSL";
  }

 private:
  int or_num{0};
  int and_num{0};
  bool has_tensor{false};
};

class CutSetTopDown final : protected IRVisitor {
 public:
  CutSetTopDown() {}
  ~CutSetTopDown() override = default;

  const isl::union_map Run(const Expr &expr, const isl::multi_id &tuple_, const isl::union_map &accesses_,
                           const isl::set &read_set_) {
    accesses = accesses_;
    read_set = read_set_;
    tuple = tuple_;
    Visit(expr);
    return accesses;
  }

 private:
  static std::unordered_set<std::string> GatherCallTensors(const Expr &e) {
    std::unordered_set<std::string> tensor_names;
    PostOrderVisit(e, [&](const NodeRef &node) -> void {
      if (auto op = node.as<Call>()) {
        if (op->call_type == Call::CallType::Halide) {
          tensor_names.insert(op->func->func_name());
        }
      }
    });
    return tensor_names;
  }

  void CutAccesses(const Expr &value, const std::vector<Expr> &conds, bool is_else, bool is_or) {
    auto may_access_tensors = GatherCallTensors(value);
    isl::union_map must_access = isl::union_map::empty(accesses.space());
    isl::union_map may_access = isl::union_map::empty(accesses.space());
    accesses.foreach_map([&](const isl::map &map) {
      auto tensor = map.get_tuple_id(isl_dim_out).get_name();
      if (may_access_tensors.count(tensor) == 0) {
        must_access = must_access.add_map(map);
      } else {
        may_access = may_access.add_map(map);
      }
    });
    read_set = CutSet(conds, read_set, is_else, is_or);
    auto cut_may_access = may_access.curry().intersect_domain(read_set.unbind_params(tuple)).uncurry();
    accesses = must_access.unite(cut_may_access);
  }

  void Visit_(const Select *sel) final {
    auto ec = ExtractCond();
    std::vector<Expr> conds = ec.run(sel->condition);
    if (!ec.hasBothOrAndAnd()) {
      if (isImm(sel->true_value)) {
        CutAccesses(sel->false_value, conds, true, !ec.IsOr());
      } else if (isImm(sel->false_value)) {
        CutAccesses(sel->true_value, conds, false, ec.IsOr());
      }
    }
  }

  isl::union_map accesses;
  isl::set read_set;
  isl::multi_id tuple;
};

class ScopMakeScheduleTree final : protected IRVisitor {
 public:
  ScopMakeScheduleTree(const NodeRef s, ScopInfo &scop_info, const isl::set set, const isl::id_list outer,
                       ssize_t macro_stmt)
      : s(s), scop_info_(scop_info), set(set), outer(outer), macro_stmt(macro_stmt) {
    IRVisitor::Visit(s);
  }
  ~ScopMakeScheduleTree() override = default;

  /// Visitor implementation
  void Visit_(const Provide *op) final {
    size_t stmt_index = scop_info_.analysis_result_.GetStatementMap().size();
    isl::id id(set.ctx(), macro_stmt >= 0 ? kStatementLabel + std::to_string(macro_stmt)
                                          : kStatementLabel + std::to_string(stmt_index));
    scop_info_.analysis_result_.RecordStatement(id, op);

    auto tuple_space = isl::space(set.ctx(), 0);
    tuple_space = tuple_space.add_named_tuple_id_ui(id, static_cast<unsigned int>(outer.size()));
    OperatorDomainSpace op_domain;
    op_domain.param_space = set.get_space();
    op_domain.tuple = isl::multi_id(tuple_space, outer);
    scop_info_.analysis_result_.RecordOperatorDomain(id, op_domain);
    auto domain = set.unbind_params(op_domain.tuple);
    sch = isl::schedule::from_domain(domain);

    if (scop_info_.user_config_.GetTarget() == TARGET_CUDA &&
        (scop_info_.user_config_.GetEnableAkgReduceLib() || scop_info_.user_config_.GetEnableMatmul())) {
      RecordReduceInfo(op, op_domain, id);
    }
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
        if ((left || right) && (matmul_map.find(left->name) != matmul_map.end() || matmul_map.find(right->name) != matmul_map.end())) {
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

    isl::union_map new_reads, new_writes, new_to_inner;
    isl::union_map new_reads_with_conds, new_writes_with_conds;
    isl::set read_set = set;
    isl::set write_set = set;
    isl::set reduction_set = set;
    Stmt stmt = Downcast<Stmt>(s);
    std::tie(new_reads, new_writes, new_to_inner) =
      ConstructPolyAccesses(op_domain, stmt, scop_info_.analysis_result_.GetAccessMap());

    new_reads_with_conds = new_reads.curry().intersect_domain(read_set.unbind_params(op_domain.tuple)).uncurry();
    /// has Select
#if (SELECT_DOMAIN_OPT)
    new_reads_with_conds = CutSetTopDown().Run(op->value, op_domain.tuple, new_reads_with_conds, read_set);
#endif
    new_writes_with_conds = new_writes.curry().intersect_domain(write_set.unbind_params(op_domain.tuple)).uncurry();

    ParseStmtOps(id, op, scop_info_.analysis_result_, new_reads, new_writes);

    // The parameters should be added as constraints of the reads/writes sets
    // otherwise isl may not be able to obtain a fixed box.
    if (macro_stmt >= 0) {
      auto params = domain.params();
      new_reads = new_reads.curry().intersect_domain(params).uncurry();
      new_writes = new_writes.curry().intersect_domain(params).uncurry();

      new_reads_with_conds = new_reads_with_conds.curry().intersect_domain(params).uncurry();
      new_writes_with_conds = new_writes_with_conds.curry().intersect_domain(params).uncurry();
    }
    scop_info_.analysis_result_.RecordReads(scop_info_.analysis_result_.GetReads().unite(new_reads_with_conds));
    scop_info_.analysis_result_.RecordWrites(scop_info_.analysis_result_.GetWrites().unite(new_writes_with_conds));
    found = true;
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
    auto provide = static_cast<const Provide *>(reduce_tensor_info.stmt_node);
    if (provide == nullptr) {
      return;
    }
    auto red_tensor_name = provide->func->func_name();
    for (auto it : scop_info_.analysis_result_.GetStatementMap()) {
      auto prev_provide = static_cast<const Provide *>(it.second);
      if (prev_provide == nullptr || prev_provide == provide || prev_provide->func->func_name() != red_tensor_name) {
        continue;
      }
      init_value = prev_provide->value;
      scop_info_.analysis_result_.RecordReduceInitIds(it.first);
    }
    if (!init_value.defined()) {
      return;
    }
    reduce_tensor_info.init_value = init_value;
  }

  void RecordReduceInfo(const Provide *op, OperatorDomainSpace op_domain, isl::id red_id) {
    auto reduce_attrs = scop_info_.analysis_result_.GetReduceAttrs();
    if (reduce_attrs.empty()) {
      return;
    }
    bool is_all_reduce = scop_info_.analysis_result_.GetNotReduceAttrs().size() == 0;
    scop_info_.user_config_.SetTileCheckCoincident(!is_all_reduce);
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

    bool is_matmul = false;
    if (AkgSupportedReduceOp.count(type) == 0) {
      is_matmul = CheckMatmul(op);
      if (!is_matmul) {
        return;
      }
    } else {
      scop_info_.user_config_.SetEnableMatmul(false);
      scop_info_.user_config_.SetEnableTensorCore(false);
      scop_info_.user_config_.SetEnableTensorCoreUsePoly(false);
    }

    reduce_tensor_info.write_tensor_name = op->func->func_name();
    SetReduceInitValue(reduce_tensor_info);
    SetReduceWriteDataType(reduce_tensor_info);
    scop_info_.analysis_result_.UpdateReduceTensorInfoMap(red_id, reduce_tensor_info);

    std::string reduce_direction;
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
      if (reduce_position == call_size - non_variable_count - 1) {
        reduce_direction = X_DIRECTION;
      } else {
        reduce_direction = Y_DIRECTION;
      }
    });
    if (reduce_direction.empty()) {
      LOG(WARNING) << "Cannot identify reduce direction for stmt " << red_id;
    }
    if (is_matmul) {
      reduce_direction = X_DIRECTION;
    }
    scop_info_.analysis_result_.RecordReduceDirection(reduce_direction);
  }

  bool GetRowColInfo(const Provide *op) {
    auto axis = scop_info_.analysis_result_.GetNotReduceAxisForMatmul();
    auto reduce_axis = scop_info_.analysis_result_.GetReduceAxisForMatmul();
    auto batch_num_axis = scop_info_.analysis_result_.GetBatchAxisNumForMatmul();
    if (axis.size() < 2 || reduce_axis.size() < 1 || axis.size() <= batch_num_axis) return false;

    const Variable *axis_var[2];
    const Variable *reduce_axis_var;
    axis_var[0] = axis[batch_num_axis].as<Variable>();
    axis_var[1] = axis.back().as<Variable>();
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

  bool IsExistTensor(const std::string tensor_name, Type &tensor_type) {
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
    bool enable_tensor_core = scop_info_.user_config_.GetEnableTensorCore();
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
      enable_tensor_core = false;
    }

    auto mul_op = akg::common::SplitCast(add_op->b, tensor_c_type).as<Mul>();
    if (mul_op == nullptr) {
      return false;
    }

    auto tensor_a = akg::common::SplitCast(mul_op->a, tensor_c_type);
    auto tensor_b = akg::common::SplitCast(mul_op->b, tensor_c_type);
    std::string tensor_a_name = GetTensorName(tensor_a, enable_tensor_core);
    std::string tensor_b_name = GetTensorName(tensor_b, enable_tensor_core);

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
    
    scop_info_.user_config_.SetEnableMatmul(true);
    scop_info_.user_config_.SetEnableTensorCore(true);
    scop_info_.user_config_.SetEnableTensorCoreUsePoly(true);
    scop_info_.user_config_.SetVectorLoadType(128);   // Default vectorization access mode (128 bits).
    scop_info_.user_config_.SetEnableAkgReduceLib(false);

    if (tensor_c_type == Float(16)) {
      std::string shared_tensors = tensor_a_name + " " + tensor_b_name + " " + tensor_c->name;
      scop_info_.user_config_.SetSharedTensors(shared_tensors);
    }

    return true;
  }

  void SetMmaModeForTensor(const std::string tensor_a_name, const std::string tensor_b_name) {
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

  void Visit_(const Block *op) final {
    auto sch_first = MakeScheduleTreeHelper(op->first, scop_info_, set, outer, macro_stmt);
    auto sch_rest = MakeScheduleTreeHelper(op->rest, scop_info_, set, outer, macro_stmt);
    if (macro_stmt >= 0)
      sch = sch_first;
    else
      sch = sch_first.sequence(sch_rest);
    found = true;
  }

  void Visit_(const IfThenElse *op) final {
    Expr cond = op->condition;

    size_t stmt_index = scop_info_.analysis_result_.GetStatementMap().size();
    isl::id id(set.ctx(), macro_stmt >= 0 ? kStatementLabel + std::to_string(macro_stmt)
                                          : kStatementLabel + std::to_string(stmt_index));
    scop_info_.analysis_result_.RecordStatement(id, op);
    auto tuple_space = isl::space(set.ctx(), 0);
    tuple_space = tuple_space.add_named_tuple_id_ui(id, static_cast<unsigned int>(outer.size()));
    OperatorDomainSpace op_domain;
    op_domain.param_space = set.get_space();
    op_domain.tuple = isl::multi_id(tuple_space, outer);
    scop_info_.analysis_result_.RecordOperatorDomain(id, op_domain);
    auto domain = set.unbind_params(op_domain.tuple);
    sch = isl::schedule::from_domain(domain);

    isl::union_map new_reads, new_writes, new_to_inner;

    // Update the reads/writes sets of scop_info by analyzing the condition
    Stmt condition = Stmt(GetObjPtr<Object>(cond.get()));
    std::tie(new_reads, new_writes, new_to_inner) =
      ConstructPolyAccesses(op_domain, condition, scop_info_.analysis_result_.GetAccessMap());
    StmtOpInfo stmt_op_Info;
    for (auto a : new_reads.get_map_list()) {
      auto tensor_id = a.get_tuple_id(isl_dim_out);
      stmt_op_Info.readtensors.push_back(tensor_id);
    }
    scop_info_.analysis_result_.RecordStmtOpInfo(id, stmt_op_Info);
    ParseStmtOps(id, cond, scop_info_.analysis_result_, FunctionRef(GetObjPtr(cond.get())));
    scop_info_.analysis_result_.RecordReads(scop_info_.analysis_result_.GetReads().unite(new_reads));
    scop_info_.analysis_result_.RecordWrites(scop_info_.analysis_result_.GetWrites().unite(new_writes));

    // Update the flag for recording a macro statement
    if (macro_stmt < 0) macro_stmt = static_cast<int64_t>(stmt_index);
    // Build schedule for the then case without updating the schedule

    isl::set cut_set = set;

#if (SELECT_DOMAIN_OPT)
    auto ec = ExtractCond();
    std::vector<Expr> cond_vec = ec.run(cond);
    if (!ec.hasBothOrAndAnd()) {
      cut_set = CutSet(cond_vec, set, false, ec.IsOr());
    }
#endif
    static_cast<void>(MakeScheduleTreeHelper(op->then_case, scop_info_, cut_set, outer, macro_stmt));

    // Build schedule for the else case without updating the schedule if defined
    if (op->else_case.defined()) {
#if (SELECT_DOMAIN_OPT)
      if (!ec.hasBothOrAndAnd()) {
        cut_set = CutSet(cond_vec, set, true, !ec.IsOr());
      }
#endif
      static_cast<void>(MakeScheduleTreeHelper(op->else_case, scop_info_, cut_set, outer, macro_stmt));
    }

    found = true;
  }

  void Visit_(const Evaluate *op) final {
    const Call *call_op = op->value.as<Call>();
    if (call_op && call_op->name == CALL_IM2COL_UB) {
      size_t stmt_index = scop_info_.analysis_result_.GetStatementMap().size();
      isl::id id(set.ctx(), macro_stmt >= 0 ? kStatementLabel + std::to_string(macro_stmt)
                                            : kStatementLabel + std::to_string(stmt_index));
      scop_info_.analysis_result_.RecordStatement(id, op);
      auto tuple_space = isl::space(set.ctx(), 0);
      tuple_space = tuple_space.add_named_tuple_id_ui(id, static_cast<unsigned int>(outer.size()));

      OperatorDomainSpace op_domain;
      op_domain.param_space = set.get_space();
      op_domain.tuple = isl::multi_id(tuple_space, outer);
      scop_info_.analysis_result_.RecordOperatorDomain(id, op_domain);

      auto domain = set.unbind_params(op_domain.tuple);
      sch = isl::schedule::from_domain(domain);

      isl::union_map new_reads, new_writes, new_to_inner;
      Stmt stmt = Downcast<Stmt>(s);
      for (auto item : scop_info_.analysis_result_.GetAttrStmt()) {
        if (item->attr_key == ATTR_IM2COL_KEY) {
          stmt = AttrStmt::make(item->node, item->attr_key, item->value, stmt);
        }
      }
      std::tie(new_reads, new_writes, new_to_inner) =
        ConstructPolyAccesses(op_domain, stmt, scop_info_.analysis_result_.GetAccessMap());

      ParseStmtOps(id, op, scop_info_.analysis_result_, new_reads, new_writes);

      if (macro_stmt >= 0) {
        auto params = domain.params();
        new_reads = new_reads.curry().intersect_domain(params).uncurry();
        new_writes = new_writes.curry().intersect_domain(params).uncurry();
        new_to_inner = new_to_inner.curry().intersect_domain(params).uncurry();
      }
      scop_info_.analysis_result_.RecordReads(scop_info_.analysis_result_.GetReads().unite(new_reads));
      scop_info_.analysis_result_.RecordWrites(scop_info_.analysis_result_.GetWrites().unite(new_writes));
      found = true;
    }
  }

  void AddLoopBoundConstraints(const isl::aff &loop_var, const isl::space &space, const Expr &expr, bool permit_min,
                               bool permit_max) {
    auto constraint_bounds = Expr2AffChecked(space, expr, permit_min, permit_max);
    if (constraint_bounds.size() == 0u) LOG(INFO) << "could not obtain polyhedral lower / upper bounds from " << expr;
    for (const auto &item : constraint_bounds) {
      if (!permit_min && permit_max) {
        set = set.intersect(loop_var.ge_set(item));
      } else if (permit_min && !permit_max) {
        set = set.intersect(item.ge_set(loop_var));
      }
    }
  }

  isl::union_pw_aff GetUnionPwAffAtDomain(const isl::aff &f, const isl::union_set &domain,
                                          const OperatorDomainMap &map) {
    auto upa = isl::union_pw_aff::empty(domain.space());
    for (auto set : domain.get_set_list()) {
      upa = upa.union_add(isl::union_pw_aff(f.unbind_params_insert_domain(map.at(set.tuple_id()).tuple)));
    }
    return upa;
  }

  void Visit_(const For *op) final {
    auto loop_var_id = isl::id(set.ctx(), op->loop_var->name_hint);
    auto space = set.get_space().add_param(loop_var_id);

    auto loop_var = isl::aff::param_on_domain(space, loop_var_id);

    // Add lower/upper loop bound constraints.
    AddLoopBoundConstraints(loop_var, space, op->min, false, true);
    Expr max = Simplify_cce(op->min + op->extent - 1);
    AddLoopBoundConstraints(loop_var, space, max, true, false);

    auto outer_add = outer.add(loop_var_id);
    auto outer_list = macro_stmt >= 0 ? outer : outer_add;
    auto body_schedule = MakeScheduleTreeHelper(op->body, scop_info_, set, outer_list, macro_stmt);

    auto multi_union_pw_aff_func = isl::multi_union_pw_aff(
      GetUnionPwAffAtDomain(isl::aff::param_on_domain(space, loop_var_id), body_schedule.get_domain(),
                            scop_info_.analysis_result_.GetOperatorDomainMap()));

    sch = body_schedule.insert_partial_schedule(multi_union_pw_aff_func);
    found = true;
  }

  void Visit_(const Realize *op) final {
    auto name = op->func->func_name();
    CHECK_EQ(op->func->num_outputs(), 1);
    auto type = op->type;

    Array<Expr> shapes;
    for (auto i : op->bounds) {
      shapes.push_back(i->extent);
    }
    Tensor tensor = placeholder(shapes, type, name);
    const Buffer buffer = decl_buffer(shapes, type, name);

    scop_info_.user_config_.RecordRealizeTensors(tensor);
    IRVisitor::Visit_(op);

    /// add old realize
    scop_info_.user_config_.InsertRealizeFromInput(isl::id(scop_info_.GetCtx(), name));

    auto binds = scop_info_.user_config_.GetBind();
    for (auto i : binds) {
      if (i.first->op->name == name) return;
    }

    // add Realize's buf into binds
    scop_info_.user_config_.SetBind(tensor, buffer);
  }

  void Visit_(const ProducerConsumer *op) final {
    sch = MakeScheduleTreeHelper(op->body, scop_info_, set, outer, macro_stmt);
    found = true;
  }

  void Op_buffer_bind_scope(const AttrStmt *op) {
    /* ******************************************
     * parse attr like below
     * // attr [[buffer(Abuf, 0x29ff3b0), Tensor(shape=[1, 32, 7, 7, 16], op.name=fmap)]] buffer_bind_scope =
     * tvm_tuple(0, 1, floordiv(k, 9), 1, ((j*16)/5), 3, 0, 7, 0, 16):handle:I
     * *******************************************/
    Array<NodeRef> array = Downcast<Array<NodeRef>>(op->node);
    Buffer buffer = Downcast<Buffer>(array[0]);
    Tensor tensor = Downcast<Tensor>(array[1]);
    Array<NodeRef> update_array;
    std::string update_name = tensor->op->name;
    std::string update_scope;
    if (tensor->op.as<PlaceholderOpNode>()) {
      update_name += LOCAL_C1;
      update_scope = DOT_LOCAL_C1;
    } else {
      update_name += LOCAL_BUF;
      update_scope = DOT_LOCAL_BUF;
    }
    Buffer update_buffer =
      BufferNode::make(buffer->data, buffer->dtype, buffer->shape, buffer->strides, buffer->elem_offset, buffer->name,
                       update_scope, buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
    Tensor update_tensor = placeholder(tensor->shape, tensor->dtype, update_name);
    update_array.push_back(update_buffer);
    update_array.push_back(update_tensor);
    scop_info_.analysis_result_.RecordUpdateTensor(update_tensor);
    scop_info_.analysis_result_.RecordBufferBindVec(std::make_pair(update_array, op->value));
    scop_info_.analysis_result_.RecordAccess(op, isl::id(set.ctx(), tensor->op->name));
  }

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == air::ir::attr::reduce_update) {
      Array<IterVar> red = Downcast<Array<IterVar>>(op->node);
      const auto pro = op->body.as<Provide>();
      if (pro) {
        scop_info_.analysis_result_.RecordReduce(pro, red);
      } else {
        auto blo = op->body.as<Block>();
        if (blo) {
          while (blo->rest.defined() && blo->rest.as<Block>()) {
            blo = blo->rest.as<Block>();
          }
          const auto pro_first = blo->first.as<Provide>();
          const auto pro_rest = blo->rest.as<Provide>();
          if (pro_rest) {
            scop_info_.analysis_result_.RecordReduce(pro_rest, red);
          } else if (pro_first) {
            scop_info_.analysis_result_.RecordReduce(pro_first, red);
          }
        }
      }
      if (scop_info_.user_config_.GetTarget() == TARGET_CUDA &&
          (scop_info_.user_config_.GetEnableAkgReduceLib() || scop_info_.user_config_.GetEnableMatmul())) {
        class ExtractReductionAttrs final : public IRVisitor {
         public:
          ExtractReductionAttrs(const Stmt stmt, std::unordered_set<std::string> left_args)
              : extract_left_args(left_args) {
            IRVisitor::Visit(stmt);
          }
          ~ExtractReductionAttrs() override = default;

          void Visit_(const Variable *op) final {
            if (!extract_left_args.count(op->name_hint)) {
              extract_reduce_attrs.insert(op->name_hint);
              for (auto &i : extract_reduce_axis) {
                if (i == op) return;
              }
              extract_reduce_axis.push_back(op);
            }
          }

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
          std::unordered_set<std::string> extract_reduce_attrs;
          std::unordered_set<std::string> extract_left_args;
          std::vector<const Variable *> extract_reduce_axis;
          std::vector<Expr> visited_axis;
          unsigned int batch_axis_num;
        };

        const auto pro = op->body.as<Provide>();
        CHECK(pro);
        for (auto i = 0u; i < pro->args.size(); ++i) {
          auto args_i = pro->args[i];
          auto mod = args_i.as<FloorMod>();
          if (mod != nullptr && mod->a.as<Variable>()) {
            left_args.insert(mod->a.as<Variable>()->name_hint);
            left_axis_for_matmul.push_back(Downcast<Var>(mod->a));
          }
          auto div = args_i.as<FloorDiv>();
          if (div != nullptr && div->a.as<Variable>()) {
            left_args.insert(div->a.as<Variable>()->name_hint);
            left_axis_for_matmul.push_back(Downcast<Var>(div->a));
          }
          if (mod == nullptr && div == nullptr && args_i.as<Variable>()) {
            left_args.insert(args_i.as<Variable>()->name_hint);
            left_axis_for_matmul.push_back(Downcast<Var>(args_i));
          }
        }

        ExtractReductionAttrs extract_reduce_attr(op->body, left_args);
        scop_info_.analysis_result_.RecordReduceAttrs(extract_reduce_attr.extract_reduce_attrs);
        scop_info_.analysis_result_.RecordNotReduceAttrs(left_args);
        if (scop_info_.user_config_.GetEnableMatmul()) {
          scop_info_.analysis_result_.RecordReduceAxisForMatmul(extract_reduce_attr.extract_reduce_axis);
          scop_info_.analysis_result_.RecordNotReduceAxisForMatmul(left_axis_for_matmul);
          scop_info_.analysis_result_.RecordBatchAxisNumForMatmul(extract_reduce_attr.batch_axis_num);
        }
        sch = MakeScheduleTreeHelper(op->body, scop_info_, set, outer, macro_stmt);
        scop_info_.analysis_result_.ClearReduceAttrs();
        scop_info_.analysis_result_.ClearNotReduceAttrs();
        left_args.clear();
      } else {
        sch = MakeScheduleTreeHelper(op->body, scop_info_, set, outer, macro_stmt);
      }
    } else if (op->attr_key == air::ir::attr::buffer_bind_scope) {
      Op_buffer_bind_scope(op);
      sch = MakeScheduleTreeHelper(op->body, scop_info_, set, outer, macro_stmt);
    } else if (op->attr_key == ATTR_IM2COL_KEY) {
      scop_info_.analysis_result_.RecordAttrStmt(op);
      sch = MakeScheduleTreeHelper(op->body, scop_info_, set, outer, macro_stmt);
    } else {
      sch = MakeScheduleTreeHelper(op->body, scop_info_, set, outer, macro_stmt);
    }

    found = true;
  }

  isl::schedule sch;
  bool found{false};

 private:
  const NodeRef s;
  ScopInfo &scop_info_;
  isl::set set;
  isl::id_list outer;
  ssize_t macro_stmt{-1};
  std::unordered_set<std::string> left_args;
  std::vector<Var> left_axis_for_matmul;
};

isl::schedule MakeScheduleTreeHelper(const NodeRef &s, ScopInfo &scop_info, const isl::set &set,
                                     const isl::id_list &outer, ssize_t macro_stmt) {
  ScopMakeScheduleTree schedule_tree(s, scop_info, set, outer, macro_stmt);
  if (!schedule_tree.found) {
    LOG(FATAL) << "Unhandled " << s.get()->GetTypeKey() << " : " << s;
  }
  return schedule_tree.sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
