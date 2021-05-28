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

#ifndef PASS_POST_FUSION_UTILS_H_
#define PASS_POST_FUSION_UTILS_H_
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include "pass/convolution_model.h"
#include "pass/utils.h"
#include "poly/poly_util.h"

namespace akg {
namespace ir {
enum AxisName { NN = 0, C1, HH, WW, C0 };
enum FilterL1AxisName { BATCH = 0, NO, KO, KI, NI };
enum GemmMNK { M = 0, N, K, INVALID = 99 };
struct MadAxis {
  Expr base{0};
  VarExpr var;
  Range oo{0, 1};
  Range oi{0, 1};
  Range ii{0, 1};
};
constexpr auto WHOLE_REDUCE_UB = "whole_red_local_UB_opt_";

#define GET_OUTER_AXIS(axis_, idx_)                    \
  do {                                                 \
    if (auto axis = op->args[(idx_)].as<Variable>()) { \
      axis_ = axis;                                    \
    } else {                                           \
      CHECK(is_zero(op->args[(idx_)]));                \
      axis_ = nullptr;                                 \
    }                                                  \
  } while (0)

inline bool IsInBinds(const std::string &name, const Map<Tensor, Buffer> &extern_buffer) {
  return std::any_of(extern_buffer.begin(), extern_buffer.end(),
                     [=](const std::pair<Tensor, Buffer> &i) { return (name == i.first->op->name); });
}

class FindMNKValue : public IRVisitor {
 public:
  explicit FindMNKValue(GemmMNK mnk)
      : update_map_value_(false),
        mnk_(mnk),
        o_axis_(Expr(1)),
        i_axis_(Expr(16)),
        ov_axis_(Expr(0)),
        iv_axis_(Expr(0)) {}
  ~FindMNKValue() override = default;

  void Find(const Stmt &stmt);
  Expr GetOAxis() { return o_axis_; }
  Expr GetIAxis() { return i_axis_; }
  Expr GetOVAxis() { return ov_axis_; }
  Expr GetIVAxis() { return iv_axis_; }

 private:
  void Visit_(const Call *op) final;
  void Visit_(const For *op) final;
  bool IsSame(const Expr &left, const Expr &right);

  bool update_map_value_{false};
  GemmMNK mnk_{INVALID};
  std::unordered_map<std::string, Expr> maps_;
  Expr o_axis_;
  Expr i_axis_;
  Expr ov_axis_;
  Expr iv_axis_;
};

class FindMadAttrVar : public IRVisitor {
 public:
  explicit FindMadAttrVar(bool use_all) : use_all_(use_all) {}
  ~FindMadAttrVar() override = default;

  Range FindNameRange(const std::string &name);
  Range FindRange(const std::string &preName);
  std::string FindAxisName(const std::string &preName) const;
  std::string FindOldAxisName(const std::string &newName);

  bool is_var_substitute_{false};

 private:
  void Visit_(const AttrStmt *op) final;
  void Visit_(const For *op) final;

  bool use_all_{false};
  Map<std::string, Range> ranges_;
  Map<std::string, VarExpr> old_axis_map_;
};

class FindCUBCall : public IRVisitor {
 public:
  explicit FindCUBCall(const std::string &name) : name_(name) {}
  ~FindCUBCall() override = default;
  const Call *c_ub_{nullptr};

 private:
  void Visit_(const Call *op) final {
    if (op->name == name_) {
      c_ub_ = op;
    }
    IRVisitor::Visit_(op);
  }

  std::string name_;
};

class SubstituteArgs : public IRMutator {
 public:
  SubstituteArgs(const Array<Expr> &args, const Array<Expr> &bias_args, const Array<Expr> &reduce_args,
                 const std::string &bias, const Array<Expr> &bias_offset, bool is_reduce,
                 const std::unordered_set<const Provide *> &reduce_tensor_set)
      : args_(args),
        bias_args_(bias_args),
        reduce_args_(reduce_args),
        bias_(bias),
        bias_offset_(bias_offset),
        is_reduce_(is_reduce),
        reduce_tensor_set_(reduce_tensor_set) {}
  ~SubstituteArgs() override = default;

 private:
  Expr Mutate_(const Call *op, const Expr &e) final;
  Stmt Mutate_(const Provide *op, const Stmt &s) final;

  Array<Expr> args_;
  Array<Expr> bias_args_;
  Array<Expr> reduce_args_;
  std::string bias_{""};
  Array<Expr> bias_offset_;
  bool is_reduce_{false};
  const std::unordered_set<const Provide *> &reduce_tensor_set_;
};

class RealizeNewFunc : public IRMutator {
 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == air::ir::attr::realize_scope) {
      if (auto r = op->body.as<Realize>()) {
        Array<Expr> shape;
        auto t = placeholder(shape, r->type, r->func->func_name());
        auto stmt = IRMutator::Mutate_(op, s);
        return TensorSubstitute(stmt, r->func, t->op, t->value_index);
      }
    }
    return IRMutator::Mutate_(op, s);
  }
};

class RealizeNewShape : public IRMutator {
 public:
  explicit RealizeNewShape(const std::string &bias) : bias_(bias) {}
  ~RealizeNewShape() override = default;

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final;
  Stmt Mutate_(const For *op, const Stmt &s) final;
  Stmt Mutate_(const Provide *op, const Stmt &s) final;
  Stmt Mutate_(const Realize *op, const Stmt &s) final;

  bool mutate_{false};
  bool is_l0write_{true};
  std::string bias_;
  Array<Expr> c_ub_l0idx_;
  std::unordered_map<std::string, Range> l0write_region_;
};

class FractalInfoExtractor : public IRVisitor {
 public:
  explicit FractalInfoExtractor(bool is_dynamic) {
    is_dynamic_ = is_dynamic;
    axis_map_["m"] = MadAxis();
    axis_map_["k"] = MadAxis();
    axis_map_["n"] = MadAxis();
  }
  ~FractalInfoExtractor() override = default;

  std::vector<Expr> gemmFormula_;

 private:
  void UpdateMNKAxis(GemmMNK mnk, const Stmt &s);
  void ComputeMadAxis(MadAxis &curAxis, const std::string &name, const Range &range);
  void ComputeMFormula(const Stmt &smt, const Expr &base, const Expr &baseN);
  bool IsConvGemmKIsolate();
  void Visit_(const For *op) final;
  void Visit_(const AttrStmt *op) final;

  bool pragma_gemm_l0_{false};
  bool is_dynamic_{false};
  std::unordered_map<std::string, MadAxis> axis_map_;
  std::unordered_map<std::string, VarExpr> lv_map_;
  std::unordered_map<std::string, Range> r_map_;
};

// correct axles' name
class ExtractIterfromExpr : public air::ir::IRVisitor {
 public:
  // extract Variable* from expr ((mo_11*16) + mi_12)/7 + ((mo_11*16) + mi_12) % 7 into idx_vec_
  ExtractIterfromExpr() = default;
  ~ExtractIterfromExpr() override = default;

  std::set<const Variable *> GetIdxVar() { return idx_vec_; }
  const Variable *GetMivar() const { return mi_var_; }

 private:
  void Visit_(const Variable *op) final;
  void Visit_(const Block *op) final;
  void Visit_(const Add *op) final;

  std::set<const Variable *> idx_vec_;
  const Variable *mi_var_{nullptr};
};

class GemmAxisMap : public IRMutator {
 public:
  explicit GemmAxisMap(bool is_conv_backprop_filter) : is_conv_backprop_filter_(is_conv_backprop_filter) {}
  ~GemmAxisMap() override = default;
  std::map<std::string, Expr> axis_map_info_;

 private:
  Stmt Mutate_(const Block *op, const Stmt &s) final;
  Stmt Mutate_(const Provide *op, const Stmt &s) final;
  void UpdateAxisMap(const Expr &e, const std::string &v);

  bool is_conv_backprop_filter_{false};
};

class FindOutC1HW : public IRVisitor {
 public:
  explicit FindOutC1HW(const Map<Tensor, Buffer> &extern_buffer) : binds_orig_(extern_buffer) {}
  ~FindOutC1HW() override = default;

  const Variable *OutH_{nullptr};
  const Variable *OutW_{nullptr};
  const Variable *OutC1_{nullptr};
  Expr OutHExpr_{0};
  Expr OutWExpr_{0};

 private:
  void Visit_(const Provide *op) final;
  void Visit_(const Variable *op) final;
  void Visit_(const For *op) final;

  bool check_h_{false};
  bool check_w_{false};
  bool check_c1_{false};
  std::unordered_set<const Variable *> loopvars_;
  Map<Tensor, Buffer> binds_orig_;
};

class RegionExtract : public IRVisitor {
 public:
  explicit RegionExtract(const std::string &name) : name_(name) {}
  ~RegionExtract() override = default;
  Region bounds_;

 private:
  void Visit_(const For *op) final;
  void Visit_(const Provide *op) final;

  std::string name_;
  std::unordered_map<const Variable *, Range> region_map_;
};

class OutAxisExtract : public IRVisitor {
 public:
  OutAxisExtract(const Variable *outaxis, const std::set<const Variable *> &iters) : axis_o_(outaxis), iters_(iters) {}
  ~OutAxisExtract() override = default;
  const Variable *axis_oo_{nullptr};

 private:
  void Visit_(const Variable *op) final {
    if (axis_o_ && axis_o_->name_hint == op->name_hint) {
      return;
    }
    CHECK(axis_oo_ == nullptr);
    axis_oo_ = op;
  }

  const Variable *axis_o_{nullptr};
  std::set<const Variable *> iters_;
};

class ProvideExtract : public IRVisitor {
 public:
  ProvideExtract() = default;
  ~ProvideExtract() override = default;
  std::vector<const Provide *> op_;

 private:
  void Visit_(const Provide *op) final { op_.push_back(op); }
};

class TensorReplace : public IRMutator {
 public:
  TensorReplace(FunctionRef func, const Array<Expr> &lhs_args, const Array<Expr> &rhs_args,
                const std::unordered_map<std::string, const For *> &outer_loopvar_map)
      : lhs_args_(lhs_args), rhs_args_(rhs_args), outer_loopvar_map_(outer_loopvar_map) {
    func_map_.emplace(std::pair<std::string, FunctionRef>(func->func_name(), func));
  }
  ~TensorReplace() override = default;

 private:
  Expr Mutate_(const Call *op, const Expr &e) final;
  Stmt Mutate_(const For *op, const Stmt &s) final;
  Stmt Mutate_(const Realize *op, const Stmt &s) final;
  Stmt Mutate_(const Provide *op, const Stmt &s) final;

  std::unordered_map<std::string, FunctionRef> func_map_;
  Array<Expr> lhs_args_;
  Array<Expr> rhs_args_;
  std::unordered_map<std::string, const For *> outer_loopvar_map_;
  std::unordered_map<std::string, const For *> inner_loopvar_map_;
  std::set<std::string> drop_loop_;
};

class InnerAxisCollect : public IRVisitor {
 public:
  std::unordered_map<std::string, const For *> loopvar_map_;

 private:
  void Visit_(const For *op) final {
    VarExpr var = op->loop_var;
    std::string name = var->name_hint;
    loopvar_map_.emplace(std::pair<std::string, const For *>(name, op));

    IRVisitor::Visit_(op);
  }
};

class InnerRealize : public IRVisitor {
 public:
  const Realize *realize_op_{nullptr};

 private:
  void Visit_(const Realize *op) final { realize_op_ = op; }
};

class GetOuterAxisRHS : public IRVisitor {
 public:
  GetOuterAxisRHS(const std::unordered_map<std::string, VarExpr> &lv_map, const std::string &name, int idx)
      : outer_loopvar_map_(lv_map), tensor_name_(name), idx_(idx) {}
  ~GetOuterAxisRHS() override = default;
  VarExpr var_{VarExpr("")};

 private:
  void Visit_(const Provide *op) final;
  void Visit_(const Call *op) final;
  void Visit_(const Variable *op) final;

  std::unordered_map<std::string, VarExpr> outer_loopvar_map_;
  std::string tensor_name_;
  int idx_{0};
  bool is_provide_{false};
  bool is_idx_{false};
};

class GetOuterAxisLHS : public IRVisitor {
 public:
  GetOuterAxisLHS(const std::unordered_map<std::string, VarExpr> &lv_map, const std::string &name, int idx)
      : outer_loopvar_map_(lv_map), tensor_name_(name), idx_(idx) {}
  ~GetOuterAxisLHS() override = default;
  VarExpr var_{VarExpr("")};

 private:
  void Visit_(const Provide *op) final;
  void Visit_(const Variable *op) final;

  std::unordered_map<std::string, VarExpr> outer_loopvar_map_;
  std::string tensor_name_;
  int idx_{0};
  bool is_idx_{false};
};

class RemoveNullRealize : public IRMutator {
 public:
  RemoveNullRealize() = default;
  ~RemoveNullRealize() override = default;

 private:
  Stmt Mutate_(const Realize *op, const Stmt &s) final;
  Stmt Mutate_(const Provide *op, const Stmt &s) final;
  Expr Mutate_(const Call *op, const Expr &e) final;

  std::set<FunctionRef> funcs_;
};

class RemoveNullRealizeScope : public IRMutator {
 public:
  explicit RemoveNullRealizeScope(ConvolutionBackpropFilterModel &conv) : conv_(conv) {
    static_cast<void>(conv_.infer_CA1_tile());
  }
  ~RemoveNullRealizeScope() override = default;

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final;

  ConvolutionBackpropFilterModel conv_;
  int l0write_idx_{0};
  int isolate_idx_{0};
  int gemm_num_{0};
  int gemm_idx_{0};
  bool allocC_{false};
};

class FindPartialDmaCond : public IRVisitor {
 public:
  FindPartialDmaCond() = default;
  ~FindPartialDmaCond() override = default;
  const IfThenElse *cond_{nullptr};

 private:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "pragma_partial_dma_condition") {
      if (auto cond = op->body.as<IfThenElse>()) {
        cond_ = cond;
      }
    }
    IRVisitor::Visit_(op);
  }
};

class MarkAxis : public IRMutator {
 public:
  explicit MarkAxis(const std::string &name) : output_name_(name) {}
  ~MarkAxis() override = default;
  VarExpr kh_var_;
  VarExpr kw_var_;

 private:
  Stmt Mutate_(const For *op, const Stmt &s) final;
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final;

  std::string output_name_;
  std::unordered_map<std::string, VarExpr> outerlv_map_;
};

class ElseCaseSplit : public IRMutator {
 public:
  ElseCaseSplit() = default;
  ~ElseCaseSplit() override = default;

 private:
  Stmt Mutate_(const For *op, const Stmt &s) final;
};

class GetBatchAxis : public IRVisitor {
 public:
  explicit GetBatchAxis(const std::string &feature) : feature_(feature) {}
  ~GetBatchAxis() override = default;
  const Variable *batch_axis_{nullptr};

 private:
  void Visit_(const Call *op) final {
    if (op->name == feature_) {
      CHECK_EQ(op->args.size(), 5);
      batch_axis_ = op->args[0].as<Variable>();
    }
    IRVisitor::Visit_(op);
  }
  std::string feature_;
};

class GatherReduceUB : public IRVisitor {
 public:
  GatherReduceUB() = default;
  ~GatherReduceUB() override = default;

  void Visit_(const Provide *op) final {
    auto name = op->func->func_name();
    if (IsReduceUB(name)) {
      if (!reduce_ub_set_.count(name)) {
        reduce_ub_set_.insert(name);
        reduce_ubs_.emplace_back(name);
      }
    }
    IRVisitor::Visit_(op);
  }
  std::vector<std::string> reduce_ubs_;

 private:
  inline bool IsReduceUB(const std::string &name) { return name.find("red_local_UB") != std::string::npos; }
  std::unordered_set<std::string> reduce_ub_set_;
};

class GatherOpAfterReduce : public IRVisitor {
 public:
  explicit GatherOpAfterReduce(const std::string &name) : name_(name) {}
  ~GatherOpAfterReduce() override = default;

  std::vector<Stmt> op_after_reduce_;
  std::unordered_set<const Provide *> miss_realize_;

 private:
  void Visit_(const AttrStmt *op) final;
  void Visit_(const Provide *op) final;
  void Visit_(const Call *op) final;

  std::string name_;
  bool visit_provide_{false};
  bool relate_{false};
  bool already_in_{false};
  std::unordered_set<std::string> provides_;
};

class GatherC1Offset : public IRVisitor {
 public:
  explicit GatherC1Offset(const Map<Tensor, Buffer> &binds) : binds_(binds) {}
  ~GatherC1Offset() override = default;

  std::vector<Expr> c1_offset_;

 private:
  void Visit_(const AttrStmt *op) final;
  void Visit_(const Provide *op) final;
  void Visit_(const Call *op) final;

  bool in_fuse_vector_{false};
  bool found_{false};
  Expr gm_c1_{0};
  Map<Tensor, Buffer> binds_;
};

class ReduceFusionCheck : public IRVisitor {
 public:
  bool is_reduce_fusion_{false};

 private:
  void Visit_(const AttrStmt *op) override {
    if (op->attr_key == "pragma_reduce_init") {
      is_reduce_fusion_ = true;
    }
    IRVisitor::Visit_(op);
  }
};

class FindKL1 : public IRVisitor {
 public:
  FindKL1() = default;
  ~FindKL1() override = default;
  Expr k_L1_;

 private:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "pragma_gemm_l0") {
      auto attrs = Downcast<Map<std::string, Range>>(op->node);
      CHECK_GT(attrs.count("k_l1"), 0);
      k_L1_ = attrs["k_l1"]->extent;
    }
    IRVisitor::Visit_(op);
  }
};
}  // namespace ir
}  // namespace akg

#endif  // PASS_POST_FUSION_UTILS_H_
