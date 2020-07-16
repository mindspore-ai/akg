/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef PASS_ANALYZE_ALIGN_H_
#define PASS_ANALYZE_ALIGN_H_

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>

#include <string>
#include <set>
#include <list>
#include <algorithm>

#include "pass/utils.h"
#include "arith_expr_simplify.h"
#include "expr_alg_simplify.h"
#include "emit_insn/cce_params.h"
#include "common/array_api.h"

namespace akg {
namespace ir {
const std::set<std::string> exclude_align_analyze_list = {
  "mad",
  "scatter",
  "vec_binary_proposal_sort",
  "vec_binary_topk_sort",
  "vec_binary_nms",
  "vec_binary_iou",
  "vec_binary_dropout",
  "vec_single_four2five_nchw",
  "opt_broadcast",
  "reduce_reorder",
  "dma_atomic_add",
  "dma_copy_transpose",
};

const std::set<std::string> exclude_index_fix_list = {
  "mad", "vec_binary_proposal_sort", "vec_binary_topk_sort", "vec_binary_nms", "vec_binary_iou", "vec_binary_dropout",
};

class IndexOptimizer : public IRMutator {
 public:
  explicit IndexOptimizer(bool rm = false) : var2expr(), rm_load_(rm) {}
  ~IndexOptimizer() override = default;

#define MUTATE_OP(OP)                               \
  Expr Mutate_(const OP *op, const Expr &e) final { \
    Var v("tmp");                                   \
    var2expr.Set(v, e);                             \
    return v;                                       \
  }
  MUTATE_OP(Div)
  MUTATE_OP(Mod)
  MUTATE_OP(FloorDiv)
  MUTATE_OP(FloorMod)
#undef MUTATE_OP

  Expr Mutate_(const Load *op, const Expr &e) final {
    if (rm_load_) {
      Var v("tmp");
      var2expr.Set(v, e);
      return v;
    }
    return e;
  }

  Map<Var, Expr> var2expr;

 private:
  bool rm_load_;
};

int GetCommonDivisor(std::vector<int> numbers);

class IndexInfo {
 public:
  Array<Var> vars;
  Array<Expr> coefs;
  Array<Expr> extents;
  int divisor;
  int vec_len{-1};
  Var vec_var{};
  Expr offset;
  Expr index;
  bool is_serial{true};
  bool is_scalar{true};
};

class DstInfo : public IndexInfo {
 public:
  bool IsGlobal() { return (GetBufScope(p_store->buffer_var->name_hint) == DMA_COPY_GLOBAL); }
  bool IsUB() { return (GetBufScope(p_store->buffer_var->name_hint) == SCOPE_UBUF); }
  const Store *p_store;
};

class SrcInfo : public IndexInfo {
 public:
  bool IsGlobal() { return (GetBufScope(p_load->buffer_var->name_hint) == DMA_COPY_GLOBAL); }
  bool IsUB() { return (GetBufScope(p_load->buffer_var->name_hint) == SCOPE_UBUF); }
  const Load *p_load;
  bool is_imm;
  Expr imm;
};

class ArithInfo {
 public:
  Stmt GenIR() { return store; }

  void GetIntrinsicType(Array<Var> &for_vars, Array<Var> &if_vars) {
    if (for_vars.empty()) {
      if (TryScalarType()) {
        insn_type = "scalar";
      } else {
        insn_type = "discrete";
      }
      return;
    }
    if (TryScalarAssignType(if_vars)) {
      insn_type = "scalar";
      return;
    }
    if (TryReduceType()) {
      insn_type = "reduce";
      return;
    }
    auto simd_t = TrySIMDType();
    if (simd_t == 1) {
      insn_type = "simd";
      return;
    } else if (simd_t == 2) {
      insn_type = "simd_split";
      return;
    }
    if (TryVectorScalarType()) {
      insn_type = "vector_scalar";
      return;
    }
    if (TryVectorDumpType()) {
      insn_type = "vector_dump";
      return;
    }
    if (TryCrossingType()) {
      insn_type = "crossing";
      return;
    }
    if (TryDiscrete()) {
      insn_type = "discrete";
      return;
    }
    if (insn_type == "unknown") {
      CHECK(0) << "\nUnknown Intrinsic Type";
    }
  }

  // A[0] = B[1] + C[2]
  bool TryScalarType() {
    if (dst_info.IsUB() && dst_info.p_store->value.as<Load>() && src_info[0].IsGlobal()) {
      return true;
    }
    if (!is_const(dst_info.index)) {
      return false;
    }
    for (auto info : src_info) {
      if (!is_const(info.index)) {
        return false;
      }
    }
    return true;
  }

  // for i { for j { A[i] = reduce(C[X*i + j]) } }
  bool TryReduceType() {
    if (dst_info.p_store->value.as<Call>()) {
      auto t_call = dst_info.p_store->value.as<Call>();
      if (t_call->name.find("reduce_") != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  // for i { for j { A[X*i + j] = B[X*i + j] + C[j] } }
  int TrySIMDType() {
    Var cur_var = dst_info.vec_var;
    int cur_len = dst_info.vec_len;
    Expr cur_offset = dst_info.offset;
    int block_size = GetUbBlkSize(dst_info.p_store->value.type());
    bool is_simd = (cur_len >= 1) ? true : false;
    for (auto info : src_info) {
      if (info.vec_len != cur_len || !Equal(info.vec_var, cur_var)) {
        is_simd = false;
        break;
      }
    }

    bool need_split = false;
    if (is_simd) {
      for (auto info : src_info) {
        int info_block_size = GetUbBlkSize(info.p_load->type);
        if (dst_info.IsUB() && info.IsUB()) {
          if (is_const(cur_offset) && is_const(info.offset) &&
              cur_offset.as<IntImm>()->value % block_size != info.offset.as<IntImm>()->value % info_block_size) {
            need_split = true;
            break;
          }
        }
      }
    }

    if (is_simd && need_split) {
      if (src_info.size() == 1) {
        if (dst_info.divisor != 0 && src_info[0].divisor != 0 &&
            dst_info.offset.as<IntImm>()->value % dst_info.divisor !=
              src_info[0].offset.as<IntImm>()->value % src_info[0].divisor) {
          dst_info.divisor = air::ir::gcd(dst_info.divisor, dst_info.offset.as<IntImm>()->value);
          src_info[0].divisor = air::ir::gcd(src_info[0].divisor, src_info[0].offset.as<IntImm>()->value);
          auto min_dst_src = std::min(dst_info.divisor, src_info[0].divisor);
          dst_info.divisor = min_dst_src;
          src_info[0].divisor = min_dst_src;
        }
      } else {
        CHECK(0) << "\nNeed to split the vector var to make the offset equal or scalar computing\n";
      }
    }

    bool unaligned_divisor = false;
    if (is_simd) {
      if (dst_info.IsUB()) {
        if (dst_info.divisor != 0 && dst_info.divisor < cur_len) {
          dst_info.divisor = air::ir::gcd(dst_info.divisor, cur_len);
          unaligned_divisor = true;
        }
      }
      for (auto info : src_info) {
        if (info.IsUB()) {
          if (info.divisor != 0 && info.divisor < cur_len) {
            unaligned_divisor = true;
            int temp_divisor = air::ir::gcd(info.divisor, cur_len);
            dst_info.divisor = air::ir::gcd(dst_info.divisor, temp_divisor);
          }
        }
      }
    }
    if (is_simd && !need_split && !unaligned_divisor) {
      return 1;
    }
    if (is_simd && (need_split || unaligned_divisor)) {
      return 2;
    }
    return 0;
  }

  // for i { for j { A[X*i + j] = B[X*i + j] + C[Z*i] } }
  bool TryVectorScalarType() {
    if (src_info.size() != 2) {
      return false;
    }
    if (dst_info.is_serial && Equal(dst_info.vec_var, src_info[0].vec_var) &&
        !HasVars(src_info[1].index, dst_info.vec_var) &&
        (!src_info[1].is_serial || !Equal(dst_info.vec_var, src_info[1].vec_var))) {
      scalar_load = src_info[1];
      src_info.pop_back();
      return true;
    }
    if (dst_info.is_serial && Equal(dst_info.vec_var, src_info[1].vec_var) &&
        !HasVars(src_info[0].index, dst_info.vec_var) &&
        (!src_info[0].is_serial || !Equal(dst_info.vec_var, src_info[0].vec_var))) {
      scalar_load = src_info[0];
      src_info.erase(src_info.begin());
      return true;
    }
    return false;
  }

  // for i { for j { A[X*i + j] = C[Z*i] } }
  bool TryVectorDumpType() {
    if (src_info.size() != 1) {
      return false;
    }
    if (GetBufScope(dst_info.p_store->buffer_var->name_hint) == SCOPE_UBUF &&
        GetBufScope(src_info[0].p_load->buffer_var->name_hint) == SCOPE_UBUF && dst_info.is_serial &&
        !HasVars(src_info[0].index, dst_info.vec_var) &&
        (!src_info[0].is_serial || !Equal(dst_info.vec_var, src_info[0].vec_var))) {
      scalar_load = src_info[0];
      src_info.pop_back();
      return true;
    }
    return false;
  }

  bool TryScalarAssignType(Array<Var> &if_vars) {
    if (dst_info.IsUB() && dst_info.is_serial && src_info.size() == 1 && dst_info.p_store->value.as<Load>() &&
        src_info[0].IsUB()) {
      bool not_simd_or_dump = HasVars(src_info[0].index, dst_info.vec_var) &&
                              (!src_info[0].is_serial || !Equal(dst_info.vec_var, src_info[0].vec_var));
      bool in_if_vars = !if_vars.empty() && IsInArray(if_vars, dst_info.vec_var);
      if (not_simd_or_dump || in_if_vars) {
        return true;
      }
    }
    return false;
  }

  // for i { for j { A[X*i + j] = C[Y*j + i] } }
  // for i { for j { A[X*i + j] = C[Y*j] } }
  bool TryCrossingType() {
    if (dst_info.is_serial && src_info.size() == 1 && HasVars(src_info[0].index, dst_info.vec_var) &&
        (!src_info[0].is_serial || !Equal(dst_info.vec_var, src_info[0].vec_var))) {
      return true;
    }
    return false;
  }

  // for i {for j { A[X*i + Y*j] = ....} }
  bool TryDiscrete() { return !(dst_info.is_serial); }

  void GetVectorizedInfo() {
    if (insn_type == "scalar") {
      is_scalar = true;
      return;
    }
    if (insn_type == "simd" || insn_type == "vector_scalar" || insn_type == "vector_dump") {
      vec_len = dst_info.vec_len;
      vec_var = dst_info.vec_var;
      offset = dst_info.offset;
      return;
    }
    if (insn_type == "simd_split") {
      vec_len = dst_info.divisor;
      offset = 0;
      return;
    }
    if (insn_type == "reduce") {
      vec_len = src_info[0].vec_len;
      vec_var = src_info[0].vec_var;
      offset = src_info[0].offset;
      return;
    }
    if (insn_type == "crossing" || insn_type == "discrete") {
      vec_len = 1;
      if (dst_info.is_serial) {
        dst_info.vec_len = 1;
        dst_info.divisor = 1;
      }
      for (size_t i = 0; i < src_info.size(); i++) {
        if (src_info[i].is_serial) {
          src_info[i].divisor = 1;
          src_info[i].vec_len = 1;
        }
      }
      return;
    }
    CHECK(0) << "\ninsn_type is unknown\n";
  }

  DstInfo dst_info;
  std::vector<SrcInfo> src_info;
  int vec_len;
  Var vec_var;
  Expr offset;
  bool is_scalar{false};
  Stmt store;
  std::string op_type;
  std::string insn_type{"unknown"};
  SrcInfo scalar_load;
  Expr scalar_imm{Expr()};
  int scalar_imm_num{0};
};

class IRIfInfo {
 public:
  Array<Expr> conds;
  Array<Var> vars;
  Array<Stmt> ops;
};

class IRForInfo {
 public:
  Array<Var> vars;
  std::vector<int> exts;
  Array<Stmt> ops;
};

class IRInfo {
 public:
  Stmt GenStmt() {
    auto ret = GenIfAndFor();
    return ret;
  }

  Stmt GenIfAndFor() {
    auto core = arith_info.store;
    if (for_info.vars.empty()) {
      return core;
    }
    Stmt ret = core;
    for (int i = static_cast<int>(for_info.vars.size()) - 1; i >= 0; --i) {
      ret = For::make(for_info.vars[i], 0, for_info.exts[i], ForType::Serial, DeviceAPI::None, ret);
    }
    return ret;
  }

  bool ChangeLastDimReduce() {
    if (arith_info.src_info.size() != 2) {
      return false;
    }

    size_t i = 0;
    for (i = 0; i < arith_info.src_info.size(); ++i) {
      if (Equal(arith_info.src_info[i].p_load->buffer_var, arith_info.dst_info.p_store->buffer_var) &&
          Equal(arith_info.src_info[i].p_load->index, arith_info.dst_info.p_store->index)) {
        break;
      }
    }

    if (i >= 2) {
      return false;
    }

    size_t index = 0;
    if (!Equal(arith_info.src_info[1 - i].vec_var, arith_info.src_info[i].vec_var) &&
        GetIndexOfElement(for_info.vars, arith_info.src_info[1 - i].vec_var, index) &&
        !HasVars(arith_info.src_info[i].p_load->index, {arith_info.src_info[1 - i].vec_var})) {
      SrcInfo t_src = arith_info.src_info[1 - i];
      arith_info.src_info.clear();
      arith_info.src_info.push_back(t_src);
      arith_info.insn_type = "reduce_" + GetReduceType();
      Expr pack_value =
        Call::make(t_src.p_load->type, arith_info.insn_type, {GetRef<Expr>(t_src.p_load)}, Call::Extern);
      arith_info.store = Store::make(arith_info.store.as<Store>()->buffer_var, pack_value,
                                     arith_info.store.as<Store>()->index, arith_info.store.as<Store>()->predicate);
      return true;
    }

    return false;
  }

  std::string GetReduceType() {
    std::string ret = GetOpType(arith_info.dst_info.p_store->value);
    std::transform(ret.begin(), ret.end(), ret.begin(), ::tolower);
    return ret;
  }

  IRIfInfo if_info;
  IRForInfo for_info;
  ArithInfo arith_info;
};

class ImmOffsetVisitor : public IRVisitor {
 public:
  int Run(const Expr &e) {
    auto temp_index = Simplify(e);
    IRVisitor::Visit(temp_index);
    return imm_offset_;
  }

  void Visit_(const Add *op) {
    if (op->a.as<IntImm>()) {
      imm_offset_ = op->a.as<IntImm>()->value;
    } else if (op->b.as<IntImm>()) {
      imm_offset_ = op->b.as<IntImm>()->value;
    } else {
      IRVisitor::Visit(op->b);
    }
  }

  bool in_add_flag_{false};
  int imm_offset_{0};
};

class ParserVisitor : public IRVisitor {
 public:
  ParserVisitor(IRInfo &in, bool flag = false) : info(in), with_align(flag) {}
  ~ParserVisitor() override = default;

  void Run(const Stmt &s) {
    in_store = false;
    IRVisitor::Visit(s);
    if (with_align) {
      GetInsnType();
      info.arith_info.GetVectorizedInfo();
    }
  }

  void Visit_(const For *op) {
    info.for_info.vars.push_back(op->loop_var);
    info.for_info.exts.push_back(op->extent.as<IntImm>()->value);
    info.for_info.ops.push_back(op->body);
    IRVisitor::Visit(op->body);
  }

  void Visit_(const IfThenElse *op) {
    CHECK(!op->else_case.defined());
    info.if_info.conds.push_back(op->condition);
    auto var_list = GetVarsInExpr(op->condition);
    for (auto t_var : var_list) {
      if (!IsInArray(info.if_info.vars, t_var)) {
        info.if_info.vars.push_back(t_var);
      }
    }
    info.if_info.ops.push_back(op->then_case);
    IRVisitor::Visit(op->then_case);
  }

  void Visit_(const Load *op) {
    SrcInfo src_info;
    src_info.index = op->index;
    src_info.p_load = op;
    GetIndexInfo(op->index, src_info);
    info.arith_info.src_info.push_back(src_info);
  }

  void Visit_(const FloatImm *op) {
    if (in_store) {
      info.arith_info.scalar_imm = GetRef<Expr>(op);
      ++info.arith_info.scalar_imm_num;
    }
  }

  void Visit_(const IntImm *op) {
    if (in_store) {
      info.arith_info.scalar_imm = GetRef<Expr>(op);
      ++info.arith_info.scalar_imm_num;
    }
  }

  void Visit_(const Store *op) {
    info.arith_info.store = GetRef<Stmt>(op);
    info.arith_info.op_type = GetOpType(op->value);
    in_store = true;
    IRVisitor::Visit(op->value);
    in_store = false;
    DstInfo dst_info;
    dst_info.p_store = op;
    dst_info.index = op->index;
    GetIndexInfo(op->index, dst_info);
    info.arith_info.dst_info = dst_info;
  }

  void GetInsnType() { info.arith_info.GetIntrinsicType(info.for_info.vars, info.if_info.vars); }

  template <typename T>
  void GetIndexInfo(const Expr &e, T &t) {
    bool is_serial = false;
    int imm_offset = ImmOffsetVisitor().Run(e);
    t.offset = imm_offset;

    std::vector<int> nums;
    bool is_linear_inner_for = true;
    if (info.for_info.vars.empty()) {
      t.is_scalar = true;
      return;
    }
    for (size_t i = 0; i < info.for_info.vars.size(); i++) {
      auto coef = air::arith::DetectLinearEquation(e, {info.for_info.vars[i]});
      if (!coef.empty() && !Equal(coef[0], 0)) {
        t.vars.push_back(info.for_info.vars[i]);
        t.coefs.push_back(coef[0].as<IntImm>()->value);
        t.extents.push_back(info.for_info.exts[i]);
        if (!Equal(coef[0], 1)) {
          nums.push_back(coef[0].as<IntImm>()->value);
        } else {
          is_serial = true;
          t.vec_var = info.for_info.vars[i];
          t.vec_len = info.for_info.exts[i];
        }
      } else if (coef.empty()) {
        is_linear_inner_for = false;
      }
    }

    if (is_linear_inner_for) {
      if (nums.empty()) {
        t.divisor = 0;
      } else {
        t.divisor = GetCommonDivisor(nums);
      }
    } else {
      if (is_serial) {
        Map<Var, Expr> value_map;
        value_map.Set(t.vec_var, 0);
        auto new_e = Simplify(Substitute(e, value_map));
        if (Equal(Simplify(Mod::make(new_e, t.vec_len)), 0)) {
          t.divisor = t.vec_len;
        } else {
          t.divisor = 1;
        }
      } else {
        t.divisor = 1;
      }
    }
    t.is_serial = is_serial;
  }

 private:
  IRInfo &info;
  bool with_align{false};
  bool in_store{false};
};

class InsnTensor {
 public:
  InsnTensor(std::string name, Type type) : m_name(name), m_type(type) {}
  virtual ~InsnTensor() {}

  void SetAlignment(int align) { m_alignment = align; }
  int GetAlignment() { return m_alignment; }
  Type GetType() { return m_type; }

  std::string m_name;
  Type m_type;
  int m_alignment{FREE_ALIGN};
};

class UnifyAlignInfo {
 public:
  bool NeedPadding(int align, int block_size) { return (align > 0 && align % block_size != 0); }

  bool UnifyAlign() {
    bool need_adjust = false;
    int align = observers[0]->m_alignment;
    int align_size = 32 / observers[0]->GetType().bytes();
    for (size_t i = 1; i < observers.size(); ++i) {
      auto temp_align = observers[i]->m_alignment;
      auto temp_block = 32 / observers[i]->GetType().bytes();
      if (align != temp_align && (NeedPadding(align, align_size) || NeedPadding(temp_align, temp_block))) {
        need_adjust = true;
        align = SpreadAlign(align, observers[i]->m_alignment, align_size, temp_block);
      }
    }
    if (need_adjust) {
      for (size_t i = 0; i < observers.size(); ++i) {
        observers[i]->m_alignment = align;
      }
    }
    return need_adjust;
  }

  int SpreadAlign(int left, int right, int left_block, int right_block) {
    if (left < 0 || left % left_block == 0) {
      return right;
    }
    if (right < 0 || right % right_block == 0) {
      return left;
    }
    return GetCommonDivisor({left, right});
  }

  std::vector<InsnTensor *> observers;
  std::vector<int> divisors;
  std::vector<Expr> offsets;
  int vector_len;
};

class AlignAttach : public IRMutator {
 public:
  AlignAttach(std::map<const Variable *, InsnTensor *> &in_map) : m_map_(in_map) {}

  Stmt Mutate_(const Store *op, const Stmt &s) {
    auto value = this->Mutate(op->value);
    int align = 1;
    if (m_map_.count(op->buffer_var.get())) {
      align = m_map_[op->buffer_var.get()]->m_alignment;
    }
    return Store::make(op->buffer_var, value, op->index, align);
  }

  Expr Mutate_(const Load *op, const Expr &e) {
    int align = 1;
    if (m_map_.count(op->buffer_var.get())) {
      align = m_map_[op->buffer_var.get()]->m_alignment;
    }
    return Load::make(op->type, op->buffer_var, op->index, align);
  }

 private:
  std::map<const Variable *, InsnTensor *> &m_map_;
};

class AlignGen : public IRVisitor {
 public:
  Stmt Run(const Stmt stmt, std::unordered_map<const Variable *, Type> &var_info) {
    for (auto &item : var_info) {
      auto ptr = new InsnTensor(item.first->name_hint, item.second);
      observer_dic_[item.first] = ptr;
    }
    IRVisitor::Visit(stmt);
    BroadcastAlign();
    auto ret = AlignAttach(observer_dic_).Mutate(stmt);
    return ret;
  }

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "pragma_emit_insn" && exclude_align_analyze_list.count(op->value.as<StringImm>()->value) == 0) {
      IRInfo info;
      ParserVisitor(info, true).Run(op->body);
      AddAlignInfo(info);
    } else if (op->attr_key == "align_info" && op->node.as<Variable>() && observer_dic_[op->node.as<Variable>()] &&
               op->value.as<IntImm>()) {
      observer_dic_[op->node.as<Variable>()]->m_alignment = op->value.as<IntImm>()->value;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void AddAlignInfo(IRInfo &info) {
    if (info.arith_info.insn_type == "scalar") {
      return;
    }
    bool is_ub_to_gm = (info.arith_info.src_info.size() == 1) &&
                       GetBufScope(info.arith_info.dst_info.p_store->buffer_var->name_hint) == DMA_COPY_GLOBAL;
    bool is_gm_to_ub = (info.arith_info.src_info.size() == 1) &&
                       GetBufScope(info.arith_info.src_info[0].p_load->buffer_var->name_hint) == DMA_COPY_GLOBAL;
    if (!is_ub_to_gm) {
      auto dst_name = info.arith_info.dst_info.p_store->buffer_var.get();
      auto divisor_dst = info.arith_info.dst_info.divisor;
      if (!info.arith_info.is_scalar) {
        HandleAlignment(observer_dic_[dst_name], divisor_dst, info.arith_info.vec_len);
      }
    }

    if (!is_gm_to_ub) {
      for (size_t i = 0; i < info.arith_info.src_info.size(); i++) {
        auto src_name = info.arith_info.src_info[i].p_load->buffer_var.get();
        if (observer_dic_.count(src_name) && !info.arith_info.is_scalar) {
          auto src_observer = observer_dic_[src_name];
          auto divisor_src = info.arith_info.src_info[i].divisor;
          HandleAlignment(src_observer, divisor_src, info.arith_info.vec_len);
        }
      }
    }

    if (!is_ub_to_gm && !is_gm_to_ub && info.arith_info.insn_type != "reduce" &&
        info.arith_info.insn_type != "crossing" && info.arith_info.insn_type != "discrete") {
      UnifyAlignInfo temp_info;
      auto dst_name = info.arith_info.dst_info.p_store->buffer_var.get();
      temp_info.observers.push_back(observer_dic_[dst_name]);
      temp_info.divisors.push_back(info.arith_info.dst_info.divisor);
      temp_info.offsets.push_back(info.arith_info.dst_info.offset);
      temp_info.vector_len = info.arith_info.vec_len;

      for (size_t i = 0; i < info.arith_info.src_info.size(); i++) {
        auto src_name = info.arith_info.src_info[i].p_load->buffer_var.get();
        if (observer_dic_.count(src_name)) {
          temp_info.observers.push_back(observer_dic_[src_name]);
          temp_info.divisors.push_back(info.arith_info.src_info[i].divisor);
          temp_info.offsets.push_back(info.arith_info.src_info[i].offset);
        }
      }
      aligns_info_.push_back(temp_info);
    }
  }

  void HandleAlignment(InsnTensor *observer, int divisor, int vector_len) {
    auto block_size = GetUbBlkSize(observer->GetType());
    CHECK(divisor % block_size == 0 || divisor >= vector_len);
    auto cur_align = observer->GetAlignment();
    int align_temp = 0;
    if (cur_align == FREE_ALIGN && divisor % block_size == 0 && divisor >= vector_len) {
      return;
    }
    if (cur_align == FREE_ALIGN && divisor % block_size == 0 && divisor < vector_len) {
      return;
    }
    if (divisor != 0) {
      if (cur_align == FREE_ALIGN) {
        if (divisor == vector_len) {
          align_temp = vector_len;
          observer->SetAlignment(align_temp);
          return;
        }
        if (divisor >= vector_len) {
          return;
        }
        CHECK(0) << "Conditions not considered";
      }
      if (divisor % cur_align == 0 && vector_len < cur_align) {
        return;
      }
      if (divisor % cur_align != 0) {
        if (cur_align % block_size != 0) {
          align_temp = air::ir::gcd(divisor, cur_align);
        } else {
          align_temp = divisor;
        }
        if (vector_len <= align_temp) {
          observer->SetAlignment(align_temp);
        } else {
          align_temp = air::ir::gcd(vector_len, align_temp);
          observer->SetAlignment(align_temp);
        }
      }
    }
  }

  void BroadcastAlign() {
    bool has_update = true;
    while (has_update) {
      has_update = false;
      for (size_t i = 0; i < aligns_info_.size(); ++i) {
        has_update = aligns_info_[i].UnifyAlign() || has_update;
      }
    }
  }

 private:
  std::map<const Variable *, InsnTensor *> observer_dic_;
  std::vector<UnifyAlignInfo> aligns_info_;
};

}  // namespace ir
}  // namespace akg
#endif  // PASS_ANALYZE_ALIGN_H_
