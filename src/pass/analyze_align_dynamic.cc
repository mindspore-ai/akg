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

#include "pass/analyze_align.h"
#include "pass/ir_util.h"
#include "ir_pass.h"
#include "emit_insn/cce_params.h"
#include "emit_insn/insn_info.h"
#include "emit_insn/insn_pattern.h"

namespace akg {
namespace ir {
namespace {
using Var2Scope = std::map<const Variable *, std::string>;

bool IsInStorageScope(const Var2Scope &table, const Variable *var) { return table.find(var) != table.end(); }
using AlignModifier = std::function<void(Expr &)>;
using std::placeholders::_1;

class AlignInfo {
 public:
  explicit AlignInfo(const Type &t, Expr off, const AlignModifier func = nullptr, bool spread = false)
      : block_size_(GetUbBlkSize(t)), base_offset_(std::move(off)), modifiers_(), need_spread_(spread) {
    if (func) modifiers_.push_back(func);
  }
  explicit AlignInfo(const Type &t) : AlignInfo(t, 0, nullptr, false) {}
  AlignInfo() : AlignInfo(Handle(1), 0, nullptr, false) { block_size_ = 0; }
  ~AlignInfo() = default;

  int64_t block_size_{0};
  Expr base_offset_;
  std::vector<AlignModifier> modifiers_;
  bool need_spread_{false};
};

struct VarComp {
  bool operator()(const Var &v0, const Var &v1) const { return v0.get() < v1.get(); }
};

using AlignDict = std::map<Var, AlignInfo, VarComp>;

void MergeAlignInfo(AlignInfo &a, const AlignInfo &b) {
  CHECK(a.block_size_ != 0 || b.block_size_ != 0);
  CHECK(a.block_size_ == 0 || b.block_size_ == 0 || a.block_size_ == b.block_size_);
  if (a.block_size_ == 0) {
    a.block_size_ = b.block_size_;
  }
  a.need_spread_ = a.need_spread_ || b.need_spread_;
  a.base_offset_ = ExprSimplifier().Gcd(a.base_offset_, b.base_offset_);
  a.modifiers_.insert(a.modifiers_.end(), b.modifiers_.begin(), b.modifiers_.end());
}

AlignDict MergeAlignDict(const AlignDict &a, const AlignDict &b) {
  AlignDict rst = a;
  for (const auto &e : b) {
    auto it = rst.find(e.first);
    if (it != rst.end()) {
      MergeAlignInfo(it->second, e.second);
    } else {
      rst.emplace(e);
    }
  }
  return rst;
}

AlignDict GenFreeAlignDict(const StmtInfoList &com_info_list) {
  AlignDict dict;
  for (const auto &com_info : com_info_list) {
    dict.emplace(com_info->data_, AlignInfo(com_info->dtype_));
  }
  return dict;
}

AlignDict GenSpecAlignDict(const StmtInfoList &com_info_list, const Expr &align, bool is_spread) {
  AlignDict dict;
  for (const auto &com_info : com_info_list) {
    dict.emplace(com_info->data_, AlignInfo(com_info->dtype_, align, nullptr, is_spread));
  }
  return dict;
}

Expr CastInt64ToInt32(Expr e) {
  if (Int(64) == e.type() && e.as<IntImm>()) {
    e = Expr(static_cast<int32_t>(e.as<IntImm>()->value));
  }
  return e;
}
void FixAlignBySize(Expr &align, Expr &size) {
  align = CastInt64ToInt32(align);
  if (!Equal(size, align) && !Equal(ExprSimplifier().Simplify(FloorMod::make(size, align)), 0)) {
    align = ExprSimplifier().Gcd(align, size);
  }
}

class RegExprSub : public IRMutator {
 public:
  RegExprSub() = default;
  ~RegExprSub() override = default;

  Expr run(const Expr &e) { return this->Mutate(e); }

 private:
  Expr Mutate_(const Load *op, const Expr &e) final {
    if (GetBufScope(op->buffer_var->name_hint) == SCOPE_REG && isImm(op->index)) {
      return Variable::make(Int(32), "tmp");
    }
    return IRMutator::Mutate_(op, e);
  }
};

AlignDict GenNormalAlignDict(const StmtInfoList &com_info_list, bool is_spread, bool all_remained_axis = false) {
  AlignDict dict;
  for (const auto &com_info : com_info_list) {
    if (com_info->var_.empty() && !all_remained_axis) {
      MergeAlignInfo(dict[com_info->data_], AlignInfo(com_info->dtype_, 0, nullptr, is_spread));
      continue;
    }

    bool min_stride_eq1 = !com_info->var_.empty() && Equal(GetItem(com_info->strides_, -1), 1);
    auto index_expr = IndexOptimizer().Mutate(com_info->index_);
    if (min_stride_eq1) {
      auto var = GetItem(com_info->var_, -1);
      index_expr = Simplify(EliminateVarInExpr(index_expr, {var}));
    }

    Expr offset_gcd = 1;
    Expr continuity_len = min_stride_eq1 ? GetItem(com_info->shape_, -1) : 1;
    index_expr = RegExprSub().run(index_expr);
    auto vars_tmp = GetVarsInExpr(index_expr);
    Array<Var> vars;
    for (const auto &var : vars_tmp) {
      if ((var->name_hint.find("cc") != std::string::npos) || (var->name_hint.find("fv") != std::string::npos) ||
          (var->name_hint.find("ee") != std::string::npos)) {
        vars.push_back(var);
      }
    }

    if (vars.empty()) {
      offset_gcd = index_expr;
    } else {
      auto strides_tmp = air::arith::DetectLinearEquation(index_expr, vars);
      Array<Expr> strides;
      for (auto stride : strides_tmp) {
        if (!Equal(stride, 1)) {
          strides.push_back(stride);
        }
      }

      if (strides.empty()) {
        offset_gcd = -2;
      } else {
        CHECK(!strides.empty());
        offset_gcd = 0;
        for (const auto &e : strides) {
          offset_gcd = ExprSimplifier().Gcd(offset_gcd, e);
        }
      }
    }

    AlignModifier func = std::bind(FixAlignBySize, _1, continuity_len);
    MergeAlignInfo(dict[com_info->data_], AlignInfo(com_info->dtype_, offset_gcd, func, is_spread));
  }
  return dict;
}

bool IsNonLinearScalar(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list) {
  if (std::any_of(dst_info_list.begin(), dst_info_list.end(),
                  [](const StmtStoreInfo &com_info) { return (!com_info->var_.empty()); })) {
    return true;
  }
  return std::any_of(src_info_list.begin(), src_info_list.end(),
                     [](const StmtStoreInfo &com_info) { return (!com_info->var_.empty()); });
}

inline bool IsTranspose(const StmtStoreInfo &dst, const StmtStoreInfo &src) {
  return dst->var_.size() > 1 && src->var_.size() > 1 && Equal(GetItem(dst->var_, -2), GetItem(src->var_, -1)) &&
         Equal(GetItem(dst->var_, -1), GetItem(src->var_, -2)) &&
         Equal(GetItem(dst->shape_, -1), GetItem(src->shape_, -2)) &&
         Equal(GetItem(dst->shape_, -2), GetItem(src->shape_, -1)) && Equal(GetItem(dst->strides_, -1), 1) &&
         Equal(GetItem(src->strides_, -1), 1) && Equal(GetItem(dst->strides_, -2), GetItem(src->shape_, -2)) &&
         Equal(GetItem(src->strides_, -2), GetItem(dst->shape_, -2));
}

void FixAlignByShape(Expr &align, const Expr &shape0, const Expr &shape1) {
  align = CastInt64ToInt32(align);
  CHECK(!Equal(shape0, 0));
  CHECK(!Equal(shape1, 0));
  if (Equal(ExprSimplifier().Simplify(FloorMod::make(align, shape0 * shape1)), 0) || Equal(align, shape0 * shape1)) {
    return;
  } else if (Equal(align, shape0) && Equal(ExprSimplifier().Simplify(FloorMod::make(align, shape0)), 0)) {
    auto times = div(align, shape0);
    align = ExprSimplifier().Gcd(times, shape1);
    return;
  }
  align = ExprSimplifier().Gcd(align, shape0);
}
AlignDict GenTransposeAlign(const StmtStoreInfo &ori_dst, const StmtStoreInfo &ori_src, StmtInfo &if_info,
                            StmtInfo &for_info) {
  auto dst = ori_dst.Copy();
  auto src = ori_src.Copy();

  auto var_old = GetItem(dst->var_, -1);
  auto var_new = GetItem(dst->var_, -2);
  dst.GetNode()->var_ = RemoveItemAtIndex(dst->var_, -1);
  src.GetNode()->var_ = RemoveItemAtIndex(src->var_, -2);

  Expr sh0 = GetItem(dst->shape_, -1);
  Expr sh1 = GetItem(dst->shape_, -2);
  auto shape = sh0 * sh1;

  dst.GetNode()->shape_ = RemoveItemAtIndex(dst->shape_, -1);
  src.GetNode()->shape_ = RemoveItemAtIndex(src->shape_, -1);
  SetItem(dst.GetNode()->shape_, -1, Expr(shape));
  SetItem(src.GetNode()->shape_, -1, Expr(shape));

  dst.GetNode()->strides_ = RemoveItemAtIndex(dst->strides_, -2);
  src.GetNode()->strides_ = RemoveItemAtIndex(src->strides_, -2);

  Map<Var, Expr> map({{var_old, Expr(0)}, {var_new, Expr(0)}});
  dst.GetNode()->index_ = Simplify(Substitute(dst->index_, map) + var_new);
  src.GetNode()->index_ = Simplify(Substitute(src->index_, map) + var_new);

  StmtInfoList dst_list({dst});
  StmtInfoList src_list({src});
  CompactComputationInfoList(dst_list, src_list, if_info, for_info);

  auto dict = GenNormalAlignDict(MergeTwo(dst_list, src_list), false);

  dict[dst->data_].modifiers_.clear();
  dict[dst->data_].modifiers_.emplace_back(std::bind(FixAlignByShape, _1, sh0, sh1));

  dict[src->data_].modifiers_.clear();
  dict[src->data_].modifiers_.emplace_back(std::bind(FixAlignByShape, _1, sh1, sh0));

  return dict;
}

bool IsScalarDMA(const Stmt &op) {
  StmtInfo f_info;
  StmtInfo i_info;
  std::string intrin;
  std::string dma;
  StmtInfoList src_info_list;
  StmtInfoList dst_info_list;
  GetDmaComputationInfo(op, dst_info_list, src_info_list, i_info, f_info, dma, intrin);

  CHECK_GT(dst_info_list.size(), 0);
  CHECK_GT(src_info_list.size(), 0);
  const auto &d_info = dst_info_list[0];
  const auto &s_info = src_info_list[0];

  bool last_dim_equal = !d_info->var_.empty() && !s_info->var_.empty() &&
                        (GetItem(d_info->var_, -1).get() == GetItem(s_info->var_, -1).get()) &&
                        !d_info->strides_.empty() && !s_info->strides_.empty() &&
                        !Equal(GetItem(d_info->strides_, -1), GetItem(s_info->strides_, -1));
  bool is_broadcast =
    ((!s_info->strides_.empty() && !Equal((GetItem(s_info->strides_, -1)), 1)) || s_info->var_.empty()) &&
    ((!d_info->strides_.empty() && !Equal((GetItem(d_info->strides_, -1)), 1)) || d_info->var_.empty());

  bool ubuf_scalar = (is_broadcast || last_dim_equal) && (intrin == INTRIN_NAME_COPY_UB_TO_UB);
  bool broadcast_scalar = is_broadcast && (intrin == "broadcast");

  if (broadcast_scalar || ubuf_scalar) {
    auto shape = GetItem(d_info->shape_, -1);
    auto stride = GetItem(d_info->strides_, -1);
    auto block_size = GetUbBlkSize(d_info->dtype_);
    if (!(ubuf_scalar && air::arith::Analyzer().CanProve(shape < block_size) && Equal(stride, block_size) &&
          IsTwoItemEqual(d_info->strides_, s_info->strides_, -1, true))) {
      return true;
    }
  }
  return false;
}

AlignDict GetDataAlign(const Stmt &op, const bool is_dma_copy, std::vector<StmtInfoList> &info_vec) {
  StmtInfo if_info;
  StmtInfo for_info;
  StmtInfoList dst_info_list;
  StmtInfoList src_info_list;
  GetCompactComputationInfo(op, dst_info_list, src_info_list, if_info, for_info, false, true);
  auto merged_com_list = MergeTwo(dst_info_list, src_info_list);

  info_vec.push_back(merged_com_list);

  Array<NodeRef> stores;
  Array<NodeRef> loads;
  GetStoreAndLoads(op, stores, loads);
  auto org_dst_info_list = GetComputationInfo(stores, for_info);
  auto org_src_info_list = GetComputationInfo(loads, for_info);

  StmtInfoList empty_com_list;

  // check load list
  if (src_info_list.empty()) {
    // broadcast/scalar mode, such as A[i, j] = 0.0 / A[1] = 2.0
    CHECK_GT(dst_info_list.size(), 0);
    if (dst_info_list[0]->var_.empty()) {
      return GenFreeAlignDict(dst_info_list);
    } else {
      return GenNormalAlignDict(merged_com_list, false);
    }
  } else if (src_info_list.size() == 1) {
    CHECK_GT(dst_info_list.size(), 0);
    auto dst_info = dst_info_list[0];
    auto src_info = src_info_list[0];

    if (dst_info->scope_ == SCOPE_UBUF && src_info->scope_ == SCOPE_UBUF) {
      if (dst_info->var_.empty() && src_info->var_.empty()) {
        if (is_dma_copy) {
          if (IsNonLinearScalar(org_dst_info_list, org_src_info_list)) {
            // check if it is non-linear index scalar mov, such as
            // for (cc2, 0, 4) {
            //   for (cc3, 0, 6) {
            //     T_tile_local_UB[((cc2*6) + cc3)] = data_local__ub[(((cc2 % 2)*2) + (cc3 % 2))]
            //   }
            // }
            CleanNonLinearVar(org_dst_info_list, empty_com_list, if_info);
            auto align_src = GenFreeAlignDict(src_info_list);
            auto align_dst = GenNormalAlignDict(org_dst_info_list, false);
            return MergeAlignDict(align_src, align_dst);
          }
          // intrin_name = 'copy_ubuf_to_ubuf'
          // scalar op, will not influence the align
          return GenFreeAlignDict(merged_com_list);
        }
        // intrin_name = vadds or vmuls
        return GenNormalAlignDict(merged_com_list, false, true);
      } else if (src_info->var_.empty()) {
        if (Equal(GetItem(dst_info->strides_, -1), 1)) {
          // scalar broadcast
          CleanNonLinearVar(org_dst_info_list, empty_com_list, if_info);
          auto align_src = GenFreeAlignDict(src_info_list);
          auto align_dst = GenNormalAlignDict(org_dst_info_list, false);
          return MergeAlignDict(align_src, align_dst);
        }
        // intrin_name = vector_dup
        return GenFreeAlignDict(merged_com_list);
      } else if (!(dst_info->var_.empty()) && Equal(GetItem(dst_info->var_, -1), GetItem(src_info->var_, -1))) {
        if (Equal(GetItem(dst_info->strides_, -1), GetItem(src_info->strides_, -1)) &&
            Equal(GetItem(org_dst_info_list[0]->var_, -1), GetItem(org_src_info_list[0]->var_, -1))) {
          // elemwise mode, intrin_name = copy_ubuf_to_ubuf
          return GenNormalAlignDict(merged_com_list, true);
        }
        // scalar dma mode
        return GenFreeAlignDict(merged_com_list);
      } else if (IsTranspose(dst_info, src_info)) {
        if (is_dma_copy) {
          // intrin_name = vtranspose
          int block_size = GetUbBlkSize(dst_info->dtype_);
          CHECK_NE(block_size, 0);
          auto dst_shape = GetItem(dst_info->shape_, -1);
          auto src_shape = GetItem(src_info->shape_, -1);
          if (!Equal(floormod(dst_shape, block_size), 0) ||
              (!Equal(floormod(src_shape, block_size), 0) &&
               (air::arith::Analyzer().CanProve(src_shape > block_size) ||
                air::arith::Analyzer().CanProve(dst_shape > block_size)))) {
            return GenTransposeAlign(dst_info, src_info, if_info, for_info);
          } else {
            // special case optimization
            return GenNormalAlignDict(merged_com_list, false);
          }
        }
        // align = 1
        return GenSpecAlignDict(merged_com_list, 1, true);
      } else if (dst_info->var_.size() > 1 && src_info->var_.size() > 1 &&
                 !Equal(GetItem(dst_info->var_, -1), GetItem(src_info->var_, -1)) &&
                 Equal(GetItem(dst_info->var_, -2), GetItem(src_info->var_, -2))) {
        // intrin_name = broadcast
        // special case of last dim axis broadcast issue #675
        CleanNonLinearVar(org_dst_info_list, empty_com_list, if_info);
        auto align_src = GenFreeAlignDict(src_info_list);
        auto align_dst = GenNormalAlignDict(org_dst_info_list, false);
        return MergeAlignDict(align_src, align_dst);
      } else if (IsScalarDMA(op)) {
        return GenFreeAlignDict(merged_com_list);
      }
      return GenNormalAlignDict(merged_com_list, false);
    } else if (dst_info->scope_ != DMA_COPY_GLOBAL && src_info->scope_ != DMA_COPY_GLOBAL &&
               dst_info->var_.size() > 1 && src_info->var_.size() > 1 &&
               Equal(GetItem(dst_info->var_, -1), GetItem(src_info->var_, -2)) &&
               Equal(GetItem(dst_info->var_, -2), GetItem(src_info->var_, -1))) {
      // check transopse cbuf, ca, cb, cc
      if (is_dma_copy) {
        // intrin_name = vtranspose
        Expr align = GetItem(dst_info->shape_, -1) * GetItem(src_info->shape_, -1);
        return GenSpecAlignDict(merged_com_list, align, true);
      }
      // discontinuoust dma mov
      return GenSpecAlignDict(merged_com_list, 1, true);
    } else if (dst_info->var_.empty() && src_info->var_.empty()) {
      // not ub to ub mode, discontinuous dma mov
      return GenNormalAlignDict(merged_com_list, true, true);
    } else if (dst_info->var_.empty()) {
      LOG(FATAL) << "Error: Copy Vector into a scalar.";
    } else if (src_info->var_.empty()) {
      // broadcast between ub and gm
      return GenNormalAlignDict(merged_com_list, true, true);
    } else if (!Equal(GetItem(dst_info->var_, -1), GetItem(src_info->var_, -1)) ||
               !Equal(GetItem(dst_info->strides_, -1), 1) || !Equal(GetItem(src_info->strides_, -1), 1)) {
      // discontinuoust dma mov
      return GenSpecAlignDict(merged_com_list, 1, true);
    }
    return GenNormalAlignDict(merged_com_list, true);
  } else if (src_info_list.size() < 5) {
    if (IsLastAxisReduction(dst_info_list, src_info_list)) {
      // reduction mode
      if (Equal(GetItem(dst_info_list[0]->shape_, -1), 1)) {
        // reduce to a scalar
        return GenFreeAlignDict(merged_com_list);
      }
      // last dim is compacted separately
      return GenNormalAlignDict(merged_com_list, false);
    } else if (IsElementwise(dst_info_list, src_info_list)) {
      // elementwise mode
      return GenNormalAlignDict(merged_com_list, true, true);
    } else if (IsBroadcast(dst_info_list, src_info_list)) {
      // broadcast mode
      bool need_spread_ = !IsLastAxisBroadcast(dst_info_list, src_info_list);
      return GenNormalAlignDict(merged_com_list, need_spread_);
    }
    return GenNormalAlignDict(merged_com_list, true);
  } else {
    LOG(FATAL) << "Error: Can not support more than 4 loads.";
  }
  // error, and return empty map
  return AlignDict();
}

class AlignVistor : public IRVisitor {
 public:
  explicit AlignVistor(const Var2Scope &table) : storage_scope_(table) {}
  ~AlignVistor() override = default;

  void Run(const Stmt &stmt) {
    this->Visit(stmt);
    UpdateAlign();
  }

  std::map<const Variable *, Expr> min_align_;
  std::set<const Variable *> gbl_storage_;

 private:
  void Visit_(const AttrStmt *op) final {
    // nested scop, just return
    if (op->attr_key == "isolate_range") return;

    if (auto str_ptr = op->node.as<StringImm>()) {
      if (str_ptr->value == "no_align") {
        return IRVisitor::Visit_(op);
      }
    }

    // only scan dma insns
    if (op->attr_key == "pragma_ub_gm" || (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
                                           op->value.as<StringImm>()->value != "vec_binary_dropout" &&
                                           exclude_list.count(op->value.as<StringImm>()->value) == 0)) {
      bool in_dma_copy = false;
      if (op->value.as<StringImm>() && op->value.as<StringImm>()->value == "dma_copy") {
        in_dma_copy = true;
      }

      auto dict = GetDataAlign(op->body, in_dma_copy, info_vec_);
      for (auto it = dict.begin(); it != dict.end();) {
        if (!IsInStorageScope(storage_scope_, it->first.get())) {
          gbl_storage_.insert(it->first.get());
          it = dict.erase(it);
        } else {
          ++it;
        }
      }

      std::vector<Var> spread_var;
      for (const auto &e : dict) {
        if (e.second.need_spread_) {
          spread_var.push_back(e.first);
        }
        MergeAlignInfo(all_aligns_[e.first], e.second);
      }
      if (spread_var.size() > 1) {
        spread_vec_.push_back(std::move(spread_var));
      }
    }
    return IRVisitor::Visit_(op);
  }

  void UpdateAlign() {
    for (auto e : gbl_storage_) {
      auto var_ptr = const_cast<Variable *>(e);
      all_aligns_.emplace(Var(GetObjectPtr<Object>(var_ptr)), AlignInfo(var_ptr->type));
    }
    do {
      for (auto &e : all_aligns_) {
        auto &info = e.second;
        auto block_size = info.block_size_;
        CHECK_NE(block_size, 0);
        if (!Equal(ExprSimplifier().Simplify(
                     FloorMod::make(CastInt64ToInt32(info.base_offset_), CastInt64ToInt32(block_size))),
                   0)) {
          while (!Equal(info.base_offset_, 1)) {
            bool done = true;
            for (const auto &func : info.modifiers_) {
              auto old = info.base_offset_;
              func(info.base_offset_);
              if (air::arith::Analyzer().CanProve(CastInt64ToInt32(info.base_offset_) < old)) {
                done = false;
              }
            }
            if (done && FixLoopAxis()) {
              break;
            }
          }
        }
      }
    } while (!DealWithSpread());
    for (const auto &e : all_aligns_) {
      if (IsInStorageScope(storage_scope_, e.first.get())) {
        min_align_.emplace(e.first.get(), e.second.base_offset_);
      }
    }
  }

  bool FixLoopAxis() {
    for (const auto &vec_ele : info_vec_) {
      // for_v -> times
      std::map<Var, std::vector<Expr>, VarComp> coef_table;
      // for_v -> [buffer -> times]
      std::map<Var, std::map<Var, Expr, VarComp>, VarComp> buf_table;

      for (const auto &info : vec_ele) {
        auto it = all_aligns_.find(info->data_);
        CHECK(it != all_aligns_.end());
        if (air::arith::Analyzer().CanProve(CastInt64ToInt32(it->second.base_offset_) <= 1)) {
          continue;
        }
        for (size_t i = 0; i != info->var_.size(); ++i) {
          auto stride = info->strides_[i];
          auto extent = info->shape_[i];
          auto align = it->second.base_offset_;
          align = CastInt64ToInt32(align);
          if (((!is_const(align) && is_const(stride)) ||
               (is_const(stride) && is_const(align) && air::arith::Analyzer().CanProve(stride < align))) &&
              Equal(ExprSimplifier().Simplify(FloorMod::make(stride * extent, align)), 0)) {
            if (!Equal(ExprSimplifier().Simplify(FloorMod::make(align, stride)), 0)) {
              it->second.base_offset_ = ExprSimplifier().Gcd(align, stride);
              return false;
            } else if (!Equal(ExprSimplifier().Simplify(FloorMod::make(extent, div(align, stride))), 0)) {
              auto times = div(align, stride);
              auto new_times = ExprSimplifier().Gcd(extent, Simplify(times));
              it->second.base_offset_ = div(it->second.base_offset_ * new_times, times);
              return false;
            }

            auto var = info->var_[i];
            auto times = div(align, stride);
            coef_table[var].push_back(times);

            auto &times_record = buf_table[var][it->first];
            times_record = times;
          }
        }
      }

      for (const auto &i : coef_table) {
        auto align = i.second.front();
        bool changed = false;
        for (const auto &ele : i.second) {
          changed = changed || (!Equal(ele, align));
          align = CastInt64ToInt32(align);
          align = ExprSimplifier().Gcd(align, ele);
        }
        if (changed) {
          for (const auto &v : buf_table[i.first]) {
            all_aligns_[v.first].base_offset_ *= align;
            all_aligns_[v.first].base_offset_ = div(all_aligns_[v.first].base_offset_, v.second);
          }
          return false;
        }
      }
    }
    return true;
  }

  bool DealWithSpread() {
    for (const auto &vec : spread_vec_) {
      auto it = all_aligns_.find(vec.front());
      CHECK(it != all_aligns_.end());

      auto align = it->second.base_offset_;
      bool changed = false;
      for (const auto &e : vec) {
        auto it_in = all_aligns_.find(e);
        CHECK(it_in != all_aligns_.end());
        align = CastInt64ToInt32(align);
        changed = changed || (!Equal(it_in->second.base_offset_, align));
        align = ExprSimplifier().Gcd(align, it_in->second.base_offset_);
      }
      if (changed) {
        for (const auto &e : vec) {
          auto it_in = all_aligns_.find(e);
          CHECK(it_in != all_aligns_.end());
          it_in->second.base_offset_ = align;
        }
        return false;
      }
    }
    return true;
  }

  // storage scope
  const Var2Scope &storage_scope_;
  // all align_ info
  AlignDict all_aligns_;
  std::vector<std::vector<Var>> spread_vec_;
  std::vector<StmtInfoList> info_vec_;
};

// predicate is for GPU, use it to hold min align
class AlignInsert : public IRMutator {
 public:
  AlignInsert() = default;
  ~AlignInsert() override = default;

  Stmt Run(const Stmt &stmt, const Var2Scope &storage_scope) {
    AlignVistor visitor(storage_scope);
    visitor.Run(stmt);
    min_align_ = std::move(visitor.min_align_);
    gbl_storage_ = std::move(visitor.gbl_storage_);

    return this->Mutate(stmt);
  }

 private:
  Stmt Mutate_(const Store *op, const Stmt &s) final {
    auto value = this->Mutate(op->value);
    auto index = this->Mutate(op->index);

    Expr val = gbl_storage_.find(op->buffer_var.get()) == gbl_storage_.end() ? -2 : 1;
    auto it = min_align_.find(op->buffer_var.get());
    if (it != min_align_.end()) {
      val = GetAlignValue(it->second, op->value.type());
    }
    if ((val.type() == Int(64)) && val.as<IntImm>()) {
      val = Expr(static_cast<int32_t>(val.as<IntImm>()->value));
    }
    return Store::make(op->buffer_var, value, index, val);
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    auto index = this->Mutate(op->index);

    Expr val = gbl_storage_.find(op->buffer_var.get()) == gbl_storage_.end() ? -2 : 1;
    auto it = min_align_.find(op->buffer_var.get());
    if (it != min_align_.end()) {
      val = GetAlignValue(it->second, op->type);
    }
    if ((val.type() == Int(64)) && val.as<IntImm>()) {
      val = Expr(static_cast<int32_t>(val.as<IntImm>()->value));
    }
    return Load::make(op->type, op->buffer_var, index, val);
  }

  static Expr GetAlignValue(const Expr &val, const air::DataType dtype) {
    return Equal(
             ExprSimplifier().Simplify(FloorMod::make(CastInt64ToInt32(val), CastInt64ToInt32(GetUbBlkSize(dtype)))), 0)
             ? Expr(FREE_ALIGN)
             : Expr(val);
  }

  std::map<const Variable *, Expr> min_align_;
  std::set<const Variable *> gbl_storage_;
};

class FindSameNameBuf : public IRVisitor {
 public:
  FindSameNameBuf() = default;
  ~FindSameNameBuf() override = default;
  Var2Scope storage_scope_;

 private:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == air::ir::attr::storage_scope) {
      const auto buf = op->node.as<Variable>();
      CHECK(buf);
      auto str = op->value.as<StringImm>();
      CHECK(str);
      storage_scope_[buf] = str->value;
    }
    IRVisitor::Visit(op->body);
  }
};

class InsertIsolate : public IRMutator {
 public:
  explicit InsertIsolate(const Var2Scope &table) : storage_scope_(table), first_with_bb_(0), insert_isolate_(false) {}
  ~InsertIsolate() override = default;

 private:
  Stmt Mutate_(const Block *op, const Stmt &s) final {
    Stmt stmt = op->first;
    bool has_block = HasBlock(stmt);
    if (has_block) {
      insert_isolate_ = false;
      stmt = this->Mutate(op->first);
      if (HasOutput(stmt)) {
        first_with_bb_ = 0;
      }
      if (!insert_isolate_) {
        ++first_with_bb_;
      }
    } else {
      ++first_with_bb_;
    }

    CHECK(op->rest.defined());
    bool single_bb = first_with_bb_ == 1;
    Stmt rest = this->Mutate(op->rest);
    bool rest_hasout = HasOutput(rest);
    stmt = Block::make(stmt, rest);
    if (!has_block && single_bb && rest_hasout) {
      stmt = AttrStmt::make(make_zero(Int(32)), "isolate_range", 2, stmt);
      insert_isolate_ = true;
    }

    if (!has_block && first_with_bb_ > 0) {
      --first_with_bb_;
    }
    return stmt;
  }

  bool HasOutput(const Stmt &s) const {
    bool found_out = false;
    auto CheckOutput = [&found_out, this](const NodeRef &op) {
      const auto st = op.as<Store>();
      // A = A_ub
      if (st && !IsInStorageScope(this->storage_scope_, st->buffer_var.get())) {
        found_out = true;
      }
    };
    PostOrderVisit(s, CheckOutput);
    return found_out;
  }

  static bool HasBlock(const Stmt &s) {
    bool found_block = false;
    auto CheckBlock = [&found_block](const NodeRef &op) {
      if (op.as<Block>() != nullptr) {
        found_block = true;
      }
    };
    PostOrderVisit(s, CheckBlock);
    return found_block;
  }

  const Var2Scope &storage_scope_;
  int first_with_bb_;
  bool insert_isolate_;
};

// process each isolate_range once a time
class ProcessParts : public IRMutator {
 public:
  explicit ProcessParts(const Var2Scope &table) : level_(0), storage_scope_(table) {}
  ~ProcessParts() override = default;

  Stmt Run(Stmt stmt) {
    stmt = this->Mutate(stmt);
    if (level_ == 0) {
      stmt = AlignInsert().Run(stmt, storage_scope_);
    }
    return stmt;
  }

 private:
  Stmt Mutate_(const Block *op, const Stmt &s) final {
    if (!HasIsolate(s)) {
      Stmt stmt = s;
      stmt = AlignInsert().Run(stmt, storage_scope_);
      level_++;
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "isolate_range") {
      level_++;
      int cur_level = level_;
      Stmt stmt = IRMutator::Mutate_(op, s);
      // no isolate_range in this attr
      if (cur_level == level_) {
        stmt = AlignInsert().Run(stmt, storage_scope_);
      }
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  static bool HasIsolate(const Stmt &s) {
    bool found_isolate = false;
    auto CheckIsolate = [&found_isolate](const NodeRef &op) {
      const auto attr = op.as<AttrStmt>();
      if (attr && attr->attr_key == "isolate_range") {
        found_isolate = true;
      }
    };
    PostOrderVisit(s, CheckIsolate);
    return found_isolate;
  }

  int level_;
  const Var2Scope &storage_scope_;
};

class AlignOne : public IRMutator {
  Stmt Mutate_(const Store *op, const Stmt &s) final {
    Expr value = this->Mutate(op->value);
    Expr index = this->Mutate(op->index);
    return Store::make(op->buffer_var, value, index, IntImm::make(Int(32), 1));
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    Expr index = this->Mutate(op->index);
    return Load::make(op->type, op->buffer_var, index, IntImm::make(Int(32), 1));
  }
};
}  // namespace

Stmt AnalyzeMinAlignDynamic(Stmt stmt, bool enable_conv_analyze_align, bool set_scalar_align) {
  stmt = air::ir::ConvertSSA(stmt);

  FindSameNameBuf find_visitor;
  find_visitor.Visit(stmt);

  stmt = MergeLoops(stmt, true);

  if (enable_conv_analyze_align) {
    if (!set_scalar_align) {
      stmt = InsertIsolate(find_visitor.storage_scope_).Mutate(stmt);
      stmt = ProcessParts(find_visitor.storage_scope_).Run(stmt);
    } else {
      stmt = AlignOne().Mutate(stmt);
    }
    stmt = RewriteByAlignDynamic(stmt);
  }
  return stmt;
}
}  // namespace ir
}  // namespace akg
