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
#include "band_node_analysis.h"
#include "poly/schedule_tree_util.h"
#include "poly/schedule_pass.h"
#include "poly/scop_builder.h"

namespace akg {
namespace ir {
namespace poly {

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
    if (visited_axis_.size() == 0) {
      batch_axis_num_ = op->args.size();
      batch_axis_var_num_ = batch_axis_num_;
      for (size_t i = 0; i < op->args.size(); i++) {
        visited_axis_.push_back(op->args[i]);
      }
    } else {
      unsigned int same_axis_num = 0;
      unsigned int same_axis_var_num = 0;
      for (size_t i = 0; (i < op->args.size()) && (i < visited_axis_.size()); i++) {
        if (Equal(op->args[i], visited_axis_[i])) {
          if (op->args[i].as<IntImm>() == nullptr) {
            same_axis_var_num++;
          }
          same_axis_num++;
        } else {
          break;
        }
      }
      batch_axis_num_ = batch_axis_num_ > same_axis_num ? same_axis_num : batch_axis_num_;
      batch_axis_var_num_ = batch_axis_var_num_ > same_axis_var_num ? same_axis_var_num : batch_axis_var_num_;
    }
    IRVisitor::Visit_(op);
  }

 public:
  std::vector<Expr> visited_axis_;
  unsigned int batch_axis_num_{0};
  unsigned int batch_axis_var_num_{0};
};

class OperatorInfoCollector {
 public:
  explicit OperatorInfoCollector(ScopInfo &scop_info) : scop_info_(scop_info), const_batch_axis_num_(0) {}
  ~OperatorInfoCollector() = default;
  void Run() {
    auto op_type = scop_info_.analysis_result_.GetOpTemplate();
    if (op_type == Template::REDUCTION) {
      RecordReduceInfo(true);
    } else if (op_type == Template::MATMUL || op_type == Template::CONV) {
      if (scop_info_.user_config_.GetTarget() == TARGET_CPU) {
        RecordReduceInfo(false);
      }
      RecordMatmulInfo();
    }
  }

  void RecordReduceDirection(std::unordered_set<std::string> &reduce_attrs, const Provide *op, isl::id red_id) {
    ReduceDirection reduce_direction = ReduceDirection::UNKNOWN;
    if (scop_info_.analysis_result_.GetCsr()) {
      // If CSR has feature dimension, threadIdx.x should map to the feature
      // dimension instead, to keep continuous memory load/save
      if (scop_info_.analysis_result_.GetCsrFeatLen() > 1) {
        reduce_direction = ReduceDirection::Y;
      } else {
        reduce_direction = ReduceDirection::X;
      }
    } else {
      PostOrderVisit(op->value, [&reduce_direction, &reduce_attrs, op, this](const NodeRef &node) -> void {
        if (reduce_direction == ReduceDirection::Y) {
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
        scop_info_.user_config_.SetTileCheckCoincident(!is_all_reduce);

        if (is_all_reduce) {
          reduce_direction = ReduceDirection::ALL;
          return;
        }

        if (reduce_position == call_size - non_variable_count - 1) {
          reduce_direction = ReduceDirection::X;
        } else {
          reduce_direction = ReduceDirection::Y;
        }
      });
    }
    if (reduce_direction == ReduceDirection::UNKNOWN) {
      LOG(WARNING) << "Cannot identify reduce direction for stmt " << red_id;
    }
    scop_info_.analysis_result_.RecordReduceDirection(red_id, reduce_direction);
  }

  void RecordReduceInfo(const bool is_reduce_operator) {
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
      if (red_id.is_null()) {
        LOG(WARNING) << "Not find corresponding reduce identifier statement (red_id) for statement: "
          << op->func;
        continue;
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

      isl::aff_list aff_list = isl::aff_list(op_domain.tuple.ctx(), 0);
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
      SetReduceInitValue(reduce_tensor_info);
      auto type = scop_info_.analysis_result_.GetReduceOpType(red_id);
      if (type == AKG_REDUCE_AND || type == AKG_REDUCE_OR) {
        scop_info_.analysis_result_.SetOpTemplate(Template::BITWISE_REDUCTION);
      }

      if (is_reduce_operator) {
        reduce_tensor_info.write_tensor_name = op->func->func_name();
        SetReduceWriteDataType(reduce_tensor_info);
        scop_info_.analysis_result_.UpdateReduceTensorInfoMap(red_id, reduce_tensor_info);
      }

      if (AkgSupportedReduceOp.count(type) == 0) {
        if (is_reduce_operator) {
          LOG(WARNING) << red_id << " does not detect the reduce operator type!";
          scop_info_.analysis_result_.SetUseGpuReduceLib(false);
        }
        return;
      }

      RecordReduceDirection(reduce_attrs, op, red_id);
      if (scop_info_.user_config_.GetEnableAkgReduceLib()) {
        scop_info_.analysis_result_.SetUseGpuReduceLib(true);
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
      scop_info_.analysis_result_.RecordBatchAxisNumForMatmul(get_batch_num.batch_axis_num_);
      const_batch_axis_num_ = get_batch_num.batch_axis_num_ - get_batch_num.batch_axis_var_num_;

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
      bool enable_tensor_core = false;
      bool enable_matmul = CheckMatmul(op, enable_tensor_core);
      enable_tensor_core &= enable_matmul;
      if (enable_matmul) {
        RecordMatrixInfoForFuse(op);
      }
      if (enable_tensor_core) {
        // Default vectorization access mode (128 bits).
        if (scop_info_.user_config_.GetVectorLength() == 0 && scop_info_.user_config_.GetTarget() != TARGET_CPU) {
          scop_info_.analysis_result_.SetVectorizedLength(VECTORIZED_128_BIT);
        }
      }
      scop_info_.user_config_.SetEnableMatmul(enable_matmul);
      scop_info_.user_config_.SetEnableTensorCore(enable_tensor_core);
      scop_info_.user_config_.SetEnableTensorCoreUsePoly(enable_tensor_core);
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
  };

  bool SetMatmulRowColInfo(const Provide *op) {
    auto axis = scop_info_.analysis_result_.GetNotReduceAxisForMatmul();
    auto reduce_axis = scop_info_.analysis_result_.GetReduceAxisForMatmul();
    auto batch_num_axis = scop_info_.analysis_result_.GetBatchAxisNumForMatmul();
    auto nonzero_batch_axis_num = batch_num_axis - const_batch_axis_num_;
    const unsigned int not_reduce_axis_num = 2;
    if (reduce_axis.size() < 1 || axis.size() <= nonzero_batch_axis_num) {
      return false;
    }

    if (scop_info_.user_config_.GetTarget() == TARGET_GPU && axis.size() < not_reduce_axis_num) {
      return false;
    }

    auto right = op->value;
    auto add_op = right.as<Add>();
    CHECK(add_op);

    CollectInfoOfBody collect_info_of_body;
    collect_info_of_body.Visit(add_op->b);
    const Variable *reduce_axis_var = reduce_axis.back();
    for (auto iter : collect_info_of_body.GetArgs()) {
      auto name = iter.first;
      auto args = iter.second;

      auto matmul_map = scop_info_.analysis_result_.GetMatrixMatmulMap();
      auto matmul_name = matmul_map.find(name);

      const size_t mn_k_num = 2;
      if (matmul_name == matmul_map.end() || args.size() < mn_k_num) {
        continue;
      }

      const Variable *var0 = args[batch_num_axis].as<Variable>();
      const Variable *var1 = args[args.size() - 1].as<Variable>();
      if (scop_info_.user_config_.GetTarget() == TARGET_GPU && (var0 == nullptr || var1 == nullptr)) {
        continue;
      }

      std::string major;
      bool is_matrix_a = matmul_name->second == MATRIX_A;
      if (is_matrix_a || matmul_name->second == MATRIX_B) {
        if (var0 == reduce_axis_var) {
          major = is_matrix_a ? COL_MAJOR : ROW_MAJOR;
        } else if (var1 == reduce_axis_var) {
          major = is_matrix_a ? ROW_MAJOR : COL_MAJOR;
        } else {
          return false;
        }
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

  bool GetTensorNameAndType(Expr tensor_data, std::string &tensor_name, Type &tensor_type) {
    if (tensor_data.as<Call>()) {
      auto tensor_data_call = tensor_data.as<Call>();
      tensor_name = tensor_data_call->name;
      if (!IsExistTensor(tensor_name, tensor_type)) {
        return false;
      }
      return true;
    } else if (tensor_data.as<Cast>()) {
      auto tensor_data_cast = tensor_data.as<Cast>();
      auto value = tensor_data_cast->value;
      tensor_name = value.as<Call>()->name;
      tensor_type = tensor_data_cast->type;
      scop_info_.analysis_result_.RecordCastTensors(tensor_name);
      return true;
    }
    return false;
  }

  bool IfEnableTensorCore(const Type type_a, const Type type_b, const Type type_c) {
    if (type_c != Float(16) && type_c != Float(32) && type_c != Int(32)) {
      return false;
    }
    if ((type_a != Float(16)) && (type_a != Int(8)) && (type_b != Float(16)) && (type_b != Int(8))) {
      return false;
    }
    return true;
  }

  bool CheckMatmul(const Provide *op, bool &enable_tensor_core) {
    if (!scop_info_.user_config_.GetEnableMatmul() && !scop_info_.user_config_.GetEnableConv2dDirect()) {
      return false;
    }
    if (scop_info_.user_config_.GetTarget() == TARGET_CPU) {
      SetPackBlockSize();
    }

    Type tensor_c_type, tensor_a_type, tensor_b_type;
    std::string tensor_c_name, tensor_a_name, tensor_b_name;

    // C + A * B
    auto add_op = op->value.as<Add>();
    if (add_op == nullptr) {
      return false;
    }

    if (add_op->a.as<Call>() == nullptr) return false;

    if (!GetTensorNameAndType(add_op->a, tensor_c_name, tensor_c_type)) {
      return false;
    }
    auto mul_op = akg::common::SplitCast(add_op->b, tensor_c_type).as<Mul>();
    if (mul_op == nullptr) {
      return false;
    }

    auto tensor_a = akg::common::SplitCast(mul_op->a, tensor_c_type);
    auto tensor_b = akg::common::SplitCast(mul_op->b, tensor_c_type);

    if (!GetTensorNameAndType(tensor_a, tensor_a_name, tensor_a_type) ||
        !GetTensorNameAndType(tensor_b, tensor_b_name, tensor_b_type)) {
      return false;
    }

    enable_tensor_core = IfEnableTensorCore(tensor_a_type, tensor_b_type, tensor_c_type);

    scop_info_.analysis_result_.RecordMatrixMatmulMap(tensor_a_name, MATRIX_A);
    scop_info_.analysis_result_.RecordMatrixMatmulMap(tensor_b_name, MATRIX_B);
    scop_info_.analysis_result_.RecordMatrixMatmulMap(tensor_c_name, MATRIX_C);

    bool ret = SetMatmulRowColInfo(op);
    if (!ret) {
      return false;
    }

    SetMmaModeForTensor(tensor_a_name, tensor_b_name);

    if (tensor_c_type == Float(16) && enable_tensor_core) {
      std::string shared_tensors = tensor_a_name + " " + tensor_b_name + " " + tensor_c_name;
      scop_info_.user_config_.RecordSharedTensors(shared_tensors);
    }

    return true;
  }

  void SetMmaModeForTensor(const std::string &tensor_a_name, const std::string &tensor_b_name) {
    const unsigned int m_axis_pos = 1;
    const unsigned int n_axis_pos = 2;
    const unsigned int k_axis_pos = 3;
    Mma mma;
    if (scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
      std::string custom_dim = scop_info_.user_config_.GetBDim();
      if (!custom_dim.empty() && !scop_info_.user_config_.GetEnableConvTensorCore()) {
        const unsigned int each_axis_size_with_mapping = 6;
        const unsigned int each_axis_size_without_mapping = 4;
        const unsigned int interval_len = 3;

        auto each_axis_size =
          custom_dim.find(T0) != std::string::npos ? each_axis_size_with_mapping : each_axis_size_without_mapping;
        std::vector<std::string> dim_str = Split(custom_dim, SPACE_PATTERN);
        auto batch_number =
          (scop_info_.analysis_result_.GetBatchAxisNumForMatmul() - const_batch_axis_num_) > 0 ? 1 : 0;
        auto real_m_axis_pos = (m_axis_pos + batch_number - 1) * each_axis_size + interval_len;
        auto real_n_axis_pos = (n_axis_pos + batch_number - 1) * each_axis_size + interval_len;
        auto real_k_axis_pos = (k_axis_pos + batch_number - 1) * each_axis_size + interval_len;
        mma.m = static_cast<int>(WrappedStrtol(dim_str[real_m_axis_pos]));
        mma.n = static_cast<int>(WrappedStrtol(dim_str[real_n_axis_pos]));
        mma.k = static_cast<int>(WrappedStrtol(dim_str[real_k_axis_pos]));
      } else {
        const int64_t row_major_m = 16;
        const int64_t row_major_n = 16;
        const int64_t row_major_k = 8;
        const int64_t col_major_m = 32;
        const int64_t col_major_n = 32;
        const int64_t col_major_k = 4;
        auto matrix_a_major = scop_info_.analysis_result_.GetMatrixMatmulMajor()[tensor_a_name];
        auto matrix_b_major = scop_info_.analysis_result_.GetMatrixMatmulMajor()[tensor_b_name];
        if (matrix_a_major == COL_MAJOR && matrix_b_major == ROW_MAJOR) {
          mma = {col_major_m, col_major_n, col_major_k};
        } else {
          mma = {row_major_m, row_major_n, row_major_k};
        }
      }
    } else {
      std::string custom_mnk = scop_info_.user_config_.GetGemmKernelMNK();
      if (!custom_mnk.empty()) {
        std::vector<std::string> tile_size_kernel = Split(custom_mnk, SPACE_PATTERN);
        mma.m = static_cast<int>(WrappedStrtol(tile_size_kernel[m_axis_pos - 1]));
        mma.n = static_cast<int>(WrappedStrtol(tile_size_kernel[n_axis_pos - 1]));
        mma.k = static_cast<int>(WrappedStrtol(tile_size_kernel[k_axis_pos - 1]));
      } else {
        auto block_size = scop_info_.analysis_result_.GetPackBlockSize();
        const int64_t pack_a = block_size.pack_a_size;
        const int64_t pack_b = block_size.pack_b_size;
        const int64_t mma_k = 8;
        mma = {pack_a, pack_b, mma_k};
      }
    }
    scop_info_.analysis_result_.SetMmaMode(mma);
  }

  void SetPackBlockSize() {
    std::string feature = scop_info_.user_config_.GetFeature();
    if (feature.empty()) {
      LOG(WARNING) << "The acquired feature is empty and will be set to the default value: sse.";
      feature = SSE_INSTRUCTION_SET;
    }
    auto it = CpuPackABBlockSize.find(feature);
    CHECK(it != CpuPackABBlockSize.end())
      << "The instruction set supported by the cpu only includes sse, avx, avx2, avx512 and neon.";

    PackBlockSize pack_block_size;
    pack_block_size.pack_a_size = it->second[0];
    pack_block_size.pack_b_size = it->second[1];
    scop_info_.analysis_result_.SetPackBlockSize(pack_block_size);
  }

 private:
  ScopInfo &scop_info_;
  // The batch axis is the number of one.
  unsigned int const_batch_axis_num_;
};

void AnalyzeBandNode::Run() {
  if (target_ == TARGET_CCE) {
    return;
  }
  AnalyzeScheduleTreeTemplate();
  // Collect information about MatMul and Conv operators
  OperatorInfoCollector op_info_coll(scop_info_);
  op_info_coll.Run();

  CollectStmtInfo();
  auto &bands = scop_info_.analysis_result_.GetAllOuterBandNode();
  for (auto &bn : bands) {
    AnalyzeOuterBandTemplate(bn);
    AnalyzeOuterBandAccessInfo(bn);
    AnalyzeAxisPosition(bn);
    bn->use_register_memory = scop_info_.user_config_.GetUseRegisterMemory();

    if (target_ == TARGET_CUDA) {
      CheckVectorization(bn);
      bn->use_shared_memory = scop_info_.user_config_.GetUseSharedMemory();
    } else if (target_ == TARGET_CPU) {
      if (bn->template_type == Template::TRANSPOSE_OP) {
        bn->enable_transpose = scop_info_.user_config_.GetEnableTranspose();
      }
    }
  }
  ShowBandInfo();
}

void AnalyzeBandNode::AnalyzeOuterBandAccessInfo(std::unique_ptr<OuterBandNode> &bn) {
  auto GetCurrentBandAccessInfo = [&bn](const isl::union_map &access_map) -> isl::union_map {
    isl::map_list access_list = access_map.get_map_list();
    isl::union_map current_access_map = isl::union_map::empty(access_map.ctx());
    for (auto access : access_list) {
      auto access_id = access.domain_factor_domain().get_tuple_id(isl_dim_in);
      for (auto stmt_id : bn->stmts) {
        if (stmt_id == access_id) {
          current_access_map = current_access_map.add_map(access);
        }
      }
    }
    return current_access_map;
  };
  bn->reads = GetCurrentBandAccessInfo(scop_info_.analysis_result_.GetReads());
  bn->writes = GetCurrentBandAccessInfo(scop_info_.analysis_result_.GetWrites());
}

void AnalyzeBandNode::AnalyzeAxisPosition(std::unique_ptr<OuterBandNode> &bn) {
  if (!bn->node.isa<isl::schedule_node_band>()) {
    return;
  }

  int last_axis = -1;
  if (target_ == TARGET_CPU) {
    last_axis = GetVectorizationAxisForCpu(bn);
  } else {
    last_axis = GetCoalescedAccessAxisForCuda(bn);
  }
  bn->last_axis = last_axis;
}

void AnalyzeBandNode::CheckVectorization(std::unique_ptr<OuterBandNode> &bn) {
  if (bn->template_type != Template::PURE_ELEM && bn->template_type != Template::BROADCAST_OP &&
      bn->template_type != Template::PAD_OP) {
    return;
  }

  if (!bn->node.get_permutable() || !scop_info_.user_config_.GetEnableVectorization() ||
      scop_info_.user_config_.GetHasTotOps() || scop_info_.analysis_result_.GetTensorOfTensor()) {
    return;
  }

  if (!bn->node.get_permutable()) {
    return;
  }
  return CheckVectorizationFromTensorSize(bn);
}

void AnalyzeBandNode::CheckVectorizationFromTensorSize(std::unique_ptr<OuterBandNode> &bn) {
  GetVectorizationTileSize(scop_info_);
  auto vectorized_loop_size = scop_info_.analysis_result_.GetVectorizedLoopSize();
  if (vectorized_loop_size == 0) {
    return;
  }

  auto reads_access = bn->reads.domain_factor_domain();
  auto write_access = bn->writes.domain_factor_domain();
  isl::map_list original_access_list = reads_access.unite(write_access).get_map_list();

  bool is_first_tensor = true;
  int64_t first_value = 0;
  int not_divisible_count = 0;
  for (auto access : original_access_list) {
    auto access_id = access.get_tuple_id(isl_dim_out).to_str();
    Array<Expr> tensor = scop_info_.FindTensor(access_id)->shape;
    CHECK(tensor[tensor.size() - 1].as<IntImm>());
    int64_t tensor_value = tensor[tensor.size() - 1].as<IntImm>()->value;
    // Non-elementwise operator: tensor does not need to be vectorized if its last axis is not divisible by the
    // vectorized size.
    // Elementwise operators: tensor does not need to be vectorized if its last axis is not divisible by the
    // vectorized size and the number of axes is greater than 1.
    if (tensor_value % vectorized_loop_size != 0 && (bn->template_type != Template::PURE_ELEM || tensor.size() > 1)) {
      ++not_divisible_count;
    }

    if (is_first_tensor) {
      first_value = tensor_value;
      is_first_tensor = false;
      continue;
    }

    // Determine whether the axis filled by the pad operator is the last axis. If it is the last axis, it indicates
    // that the length of the vectorized axis is not uniform and cannot be vectorized.
    if ((bn->template_type == Template::PAD_OP || is_unpad_op_) && tensor_value != first_value) {
      return;
    }
  }

  if (not_divisible_count == static_cast<int>(original_access_list.size())) {
    return;
  }

  bn->enable_vectorization = true;
}

int AnalyzeBandNode::GetVectorizationAxisForCpu(std::unique_ptr<OuterBandNode> &bn) {
  auto bn_schedule_node = bn->node;
  GetVectorizationTileSize(scop_info_);

  auto n_member = static_cast<int>(bn_schedule_node.n_member());
  bool is_reduce_op = (bn->template_type == Template::REDUCTION || bn->template_type == Template::BITWISE_REDUCTION);

  int vectorization_axis = -1;
  if (is_reduce_op) {
    if (bn->reduce_direction == ReduceDirection::Y) {
      vectorization_axis = 0;
    } else {
      vectorization_axis = n_member - 1;
    }
  } else if (bn->template_type == Template::BROADCAST_OP) {
    vectorization_axis = n_member - 1;
  } else {
    vectorization_axis = GetLastAxisPos(bn_schedule_node);
  }
  return vectorization_axis;
}

// For the tensor of tensor operator, confirm whether coalesced access is required in the calculation phase.
int AnalyzeBandNode::GetCoalescedAccessAxisForCuda(std::unique_ptr<OuterBandNode> &bn) {
  int coalesced_access_axis = -1;
  if (scop_info_.user_config_.GetEnableMatmul()) {
    return coalesced_access_axis;
  }
  std::unordered_set<std::string> skip_tensors = scop_info_.analysis_result_.GetTensorsNotPromote();
  for (auto inner_tensor : scop_info_.analysis_result_.GetInnerTensor()) {
    skip_tensors.emplace(inner_tensor);
  }
  coalesced_access_axis = GetLastAxisPos(bn->node, skip_tensors);
  RecordAllCoalescedAccessTensors(bn, skip_tensors);
  return coalesced_access_axis;
}

int AnalyzeBandNode::GetLastAxisPos(const isl::schedule_node &orig_node, std::unordered_set<std::string> skip_tensors) {
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return -1;
  }

  auto node = orig_node;
  auto band_node = node.as<isl::schedule_node_band>();
  auto n_parallel_axis = CountConsecutiveCoincident(band_node);
  node = band_node.split(n_parallel_axis);

  // Get read and write tensor information.
  auto reads_access = scop_info_.analysis_result_.GetReads().domain_factor_domain();
  int last_axis = GetLastAxis(node, reads_access, skip_tensors);
  if (last_axis != -1) {
    return last_axis;
  }

  auto write_access = scop_info_.analysis_result_.GetWrites().domain_factor_domain();
  last_axis = GetLastAxis(node, write_access, skip_tensors);
  if (last_axis != -1) {
    return last_axis;
  }
  return -1;
}

void AnalyzeBandNode::RecordAllCoalescedAccessTensors(std::unique_ptr<OuterBandNode> &bn,
                                                      std::unordered_set<std::string> skip_tensors) {
  if (!bn->node.isa<isl::schedule_node_band>()) {
    return;
  }

  // Get read and write tensor information.
  auto reads_access = bn->reads.domain_factor_domain();
  reads_access.foreach_map([this, &bn, skip_tensors](const isl::map &map) -> void {
    auto node = bn->node;
    int band_number = static_cast<int>(node.as<isl::schedule_node_band>().n_member());
    std::string id_name = map.range().get_tuple_name();
    auto access = isl::union_map(map);
    int last_axis = GetLastAxis(node, access, skip_tensors);
    if (last_axis != -1 && last_axis < band_number - 1) {
      bn->coalesced_access_tensors.emplace(id_name);
    }
  });
}

void AnalyzeBandNode::CollectStmtInfo() {
  auto prov_entry = scop_info_.analysis_result_.GetProvideAnalysis();
  auto provides = scop_info_.analysis_result_.GetStatementMap();
  if (prov_entry.empty() || provides.empty()) {
    return;
  }
  std::vector<ProvideEntry> entries;
  for (auto &provs : prov_entry) {
    for (auto &p : provs.second) {
      entries.emplace_back(p);
    }
  }
  auto direct_map = scop_info_.analysis_result_.GetReduceDirectionMap();
  for (auto &pro : provides) {
    const Node *op;
    if (pro.second->IsInstance<IfThenElse>()) {
      const IfThenElse *if_stmt = static_cast<const IfThenElse *>(pro.second);
      auto body = if_stmt->then_case;
      while (auto attr_stmt = body.as<AttrStmt>()) {
        body = attr_stmt->body;
      }
      op = body.as<Provide>();
    } else {
      op = pro.second;
    }
    if (op != nullptr && !op->IsInstance<Provide>()) {
      continue;
    }
    auto stmt = pro.first;
    for (auto &entry : entries) {
      if (entry.op != op) {
        continue;
      }
      std::string s_type = entry.basic_op_type;
      ReduceDirection direct{ReduceDirection::UNKNOWN};
      if (direct_map.find(stmt) != direct_map.end()) {
        direct = direct_map[stmt];
      }
      stmt_info_[stmt] = std::make_pair(s_type, direct);
    }
  }
}

bool AnalyzeBandNode::IsGemmTempleteInBand(std::unique_ptr<OuterBandNode> &bn) {
  if (!bn || bn->stmts.empty()) {
    return false;
  }
  auto stmts = scop_info_.analysis_result_.GetStatementMap();
  isl::id gemm_stmt;
  for (auto &p : gemm_provides_) {
    for (auto &s : stmts) {
      if (s.second == p) {
        gemm_stmt = s.first;
        break;
      }
    }
  }
  if (gemm_stmt.is_null()) {
    return false;
  }
  for (auto &item : bn->stmts) {
    if (item.get_name() == gemm_stmt.get_name()) {
      return true;
    }
  }
  return false;
}

void AnalyzeBandNode::DetermineTemplateOfBand(std::unique_ptr<OuterBandNode> &bn) {
  if (!bn || bn->stmts.empty()) {
    return;
  }
  std::string concated_op_type;
  ReduceDirection direct{ReduceDirection::UNKNOWN};
  isl::id red_stmt;
  auto schedule_tree_op = scop_info_.analysis_result_.GetOpTemplate();
  if (schedule_tree_op == Template::CONV || schedule_tree_op == Template::MATMUL) {
    if (IsGemmTempleteInBand(bn)) {
      bn->template_type = schedule_tree_op;
      return;
    }
  }
  for (auto &st : bn->stmts) {
    if (stmt_info_.find(st) == stmt_info_.end()) {
      continue;
    }
    concated_op_type += stmt_info_[st].first + ",";
    if (stmt_info_[st].first.find(AT_REDUCE) != std::string::npos) {
      direct = stmt_info_[st].second;
      red_stmt = st;
    }
  }
  if (concated_op_type.find(AT_REDUCE) != std::string::npos) {
    auto type = scop_info_.analysis_result_.GetReduceOpType(red_stmt);
    if (type == AKG_REDUCE_AND || type == AKG_REDUCE_OR) {
      bn->template_type = Template::BITWISE_REDUCTION;
    } else {
      bn->template_type = Template::REDUCTION;
    }
    bn->reduce_direction = direct;
    scop_info_.analysis_result_.SetReduceDirection(direct);
  } else if (concated_op_type.find(AT_TRANSPOSE) != std::string::npos) {
    bn->template_type = Template::TRANSPOSE_OP;
  } else if (concated_op_type.find(AT_PAD) != std::string::npos) {
    bn->template_type = Template::PAD_OP;
  } else if (concated_op_type.find(AT_BROADCAST) != std::string::npos ||
             concated_op_type.find(AT_TRANSFORM) != std::string::npos) {
    bn->template_type = Template::BROADCAST_OP;
  } else if (concated_op_type.find(AT_CALL) != std::string::npos) {
    bn->template_type = Template::EXTERN_CALL;
  } else if (concated_op_type.find(AT_COUNT) != std::string::npos) {
    bn->template_type = Template::COUNT_OP;
  } else if (concated_op_type.find(AT_PARTIAL_ELEM) != std::string::npos) {
    bn->template_type = Template::PARTIAL_ELEM;
  } else {
    bn->template_type = Template::PURE_ELEM;
    if (concated_op_type.find(AT_UNPAD) != std::string::npos) {
      is_unpad_op_ = true;
    }
  }
}

void AnalyzeBandNode::AnalyzeOuterBandTemplate(std::unique_ptr<OuterBandNode> &bn) {
  if (!bn->node || bn->node.get_partial_schedule().is_null()) {
    return;
  }
  isl::union_pw_aff_list aff_list = bn->node.get_partial_schedule().get_union_pw_aff_list();
  for (unsigned int i = 0; i < aff_list.size(); ++i) {
    isl::pw_aff_list pw_list = aff_list.get_at(i).get_pw_aff_list();
    for (unsigned int j = 0; j < pw_list.size(); ++j) {
      isl::pw_aff pw = pw_list.get_at(j);
      std::string stmt_id = pw.domain().get_tuple_name();
      isl::ctx ctx = bn->node.ctx();
      isl::id id(ctx, stmt_id);
      bn->stmts.emplace(id);
    }
  }
  DetermineTemplateOfBand(bn);
}

void AnalyzeBandNode::AnalyzeConvAndMatmulOp(const ProvideEntry &pe) {
  bool is_conv_or_gemm = (pe.basic_op_type.find(AT_TRANSPOSE) != std::string::npos) &&
                         (pe.basic_op_type.find(AT_ELEMWISE) != std::string::npos);
  if (!is_conv_or_gemm && !scop_info_.user_config_.GetEnableConv2dDirect()) {
    return;
  }

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
  if (scop_info_.user_config_.GetEnableConvTensorCore() || scop_info_.user_config_.GetEnableConv2dDirect()) {
    scop_info_.analysis_result_.SetOpTemplate(Template::CONV);
  } else {
    scop_info_.analysis_result_.SetOpTemplate(Template::MATMUL);
  }
  gemm_provides_.emplace_back(pe.op);
}

void AnalyzeBandNode::AnalyzeScheduleTreeTemplate() {
  std::string concated_op_type;
  auto provides = scop_info_.analysis_result_.GetProvideAnalysis();
  for (auto it : provides) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      concated_op_type += pe.basic_op_type + ",";
      AnalyzeConvAndMatmulOp(pe);
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
  } else if (concated_op_type.find(AT_COUNT) != std::string::npos) {
    scop_info_.analysis_result_.SetOpTemplate(Template::COUNT_OP);
  } else {
    scop_info_.analysis_result_.SetOpTemplate(Template::PURE_ELEM);
    if (concated_op_type.find(AT_UNPAD) != std::string::npos) {
      is_unpad_op_ = true;
    }
  }
}

void AnalyzeBandNode::ShowBandInfo() {
  auto &bands = scop_info_.analysis_result_.GetAllOuterBandNode();
  std::stringstream s;
  s << "Outer bands template: {";
  for (size_t i = 0; i < bands.size(); ++i) {
    auto *bn = bands[i].get();
    s << scop_info_.analysis_result_.ShowOpTemplate(bn->template_type) << ", ";
  }
  s << "}";
  LOG(INFO) << s.str();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
