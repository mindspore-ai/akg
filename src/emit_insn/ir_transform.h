/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef IR_TRANSFORM_H_
#define IR_TRANSFORM_H_

#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/registry.h>

#include <unordered_set>
#include <map>
#include <numeric>
#include <set>
#include <algorithm>

#include "ir_pass.h"
#include "common/array_api.h"

#include "insn_with_variable.h"
#include "insn_builder.h"
#include "insn_info.h"
#include "insn_pattern.h"
#include "../pass/analyze_align.h"

const int TransTotalSize = 256;
const int TransAxisLen = 16;

namespace akg {
namespace ir {

Expr GetVarCoefExpr(const Expr &index, const Var &loop_var);

std::string GetBufferType(Expr address);

class TransposeTransform : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
        (op->value.as<StringImm>()->value == "dma_copy")) {
      pre_transpose_buffer_ = Var("srcTranspose_local_UB");
      post_transpose_buffer_ = Var("dstTranspose_local_UB");
      pre_trans_cast_ = Var("pre_trans_cast__local_UB");
      post_trans_cast_ = Var("post_trans_cast__local_UB");
      loop_vars_ = {};
      loop_extends_ = {};
      is_candidate_ = true;
      is_block_transpose_ = false;
      is_native_transpose_ = false;
      align_value = FREE_ALIGN;
      remain_fors_.clear();
      auto body = this->Mutate(op->body);
      is_candidate_ = false;
      if (is_block_transpose_) {
        is_block_transpose_ = false;
        if (t_type_ == Float(32)) {  // need cast
          body = Allocate::make(pre_trans_cast_, Float(16), {TransTotalSize}, const_true(1), body);
          body = AttrStmt::make(pre_trans_cast_, "storage_scope", Expr("local.UB"), body);
          body = Allocate::make(post_trans_cast_, Float(16), {TransTotalSize}, const_true(1), body);
          body = AttrStmt::make(post_trans_cast_, "storage_scope", Expr("local.UB"), body);
        }
        auto allocate_pre_buffer =
          Allocate::make(pre_transpose_buffer_, t_type_, {TransTotalSize}, const_true(1), body);
        auto attr_pre_buffer =
          AttrStmt::make(pre_transpose_buffer_, "storage_scope", Expr("local.UB"), allocate_pre_buffer);
        auto allocate_post_buffer =
          Allocate::make(post_transpose_buffer_, t_type_, {TransTotalSize}, const_true(1), attr_pre_buffer);
        auto attr_post_buffer =
          AttrStmt::make(post_transpose_buffer_, "storage_scope", Expr("local.UB"), allocate_post_buffer);
        Stmt ret = attr_post_buffer;
        if (align_value != FREE_ALIGN) {
          ret = AttrStmt::make(align_buffer_, "align_info", Expr(align_value), ret);
        }
        return ret;
      }
      if (is_native_transpose_) {
        Stmt ret = AttrStmt::make(op->node, op->attr_key, Expr("dma_copy_transpose"), body);
        for (int i = 0; i <= static_cast<int>(remain_fors_.size()) - 1; ++i) {
          ret = For::make(remain_fors_[i]->loop_var, remain_fors_[i]->min, remain_fors_[i]->extent, ForType::Serial,
                          DeviceAPI::None, ret);
        }
        return ret;
      }
      return AttrStmt::make(op->node, op->attr_key, op->value, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (is_candidate_) {
      loop_vars_.push_back(op->loop_var);
      loop_extends_.push_back(op->extent);
      Stmt body = this->Mutate(op->body);
      if (is_block_transpose_ && IsInArray(trans_vars_, op->loop_var)) {
        return body;
      }
      if (is_native_transpose_) {
        if (IsInArray(trans_vars_, op->loop_var)) {
          return For::make(op->loop_var, op->min, op->extent, ForType::Serial, DeviceAPI::None, body);
        }
        remain_fors_.push_back(op);
        return body;
      }
      return For::make(op->loop_var, op->min, op->extent, ForType::Serial, DeviceAPI::None, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (is_candidate_) {
      auto value = op->value;
      if (auto cast = op->value.as<Cast>()) {
        value = cast->value;
      }
      CHECK(value.as<Load>());
      auto src_ptr = value.as<Load>();
      if (GetBufferType(op->buffer_var) == SCOPE_UBUF && GetBufferType(src_ptr->buffer_var) == SCOPE_UBUF &&
          src_ptr->type == Float(16)) {
        int dst_pos = GetVectorizedVarPosition(op->index, loop_vars_);
        int src_pos = GetVectorizedVarPosition(src_ptr->index, loop_vars_);
        if (dst_pos != -1 && src_pos != -1 && dst_pos != src_pos && HasVars(src_ptr->index, loop_vars_[dst_pos]) &&
            HasVars(op->index, loop_vars_[src_pos]) && floormod(loop_extends_[dst_pos], TransAxisLen).as<IntImm>() &&
            floormod(loop_extends_[dst_pos], TransAxisLen).as<IntImm>()->value == 0 &&
            Equal(GetVarCoefExpr(op->index, loop_vars_[src_pos]), loop_extends_[dst_pos])) {
          if (loop_extends_[dst_pos].as<IntImm>() && loop_extends_[dst_pos].as<IntImm>()->value == TransAxisLen &&
              loop_extends_[src_pos].as<IntImm>() && loop_extends_[src_pos].as<IntImm>()->value == TransAxisLen) {
            trans_vars_ = {};
            trans_vars_.push_back(loop_vars_[src_pos]);
            trans_vars_.push_back(loop_vars_[dst_pos]);
            is_native_transpose_ = true;
            return s;
          }
          is_block_transpose_ = true;
          if (GetVarCoefExpr(src_ptr->index, loop_vars_[dst_pos]).as<IntImm>()) {
            int coef_t = GetVarCoefExpr(src_ptr->index, loop_vars_[dst_pos]).as<IntImm>()->value;
            if (coef_t % TransAxisLen != 0) {
              align_value = coef_t;
              align_buffer_ = src_ptr->buffer_var;
            }
          }
          t_type_ = src_ptr->type;
          trans_vars_ = {};
          trans_vars_.push_back(loop_vars_[src_pos]);
          trans_vars_.push_back(loop_vars_[dst_pos]);
          Expr ori_w = GetVarCoefExpr(src_ptr->index, loop_vars_[dst_pos]);
          Expr ori_h = loop_extends_[dst_pos];
          Expr ori_block_w = floordiv(ori_w, TransAxisLen);
          // padding the width
          Expr unit_width = TransAxisLen;
          if (!Equal(floormod(ori_w, TransAxisLen), 0)) {
            ori_block_w = ori_block_w + 1;
          }
          if (ori_w.as<IntImm>() && ori_w.as<IntImm>()->value < TransAxisLen) {
            unit_width = ori_w;
          }
          Expr ori_block_h = floordiv(ori_h, TransAxisLen);
          Var loop_w = Var("block_w");
          Var loop_h = Var("block_h");
          Expr src_base_index = EliminateVarInExpr(src_ptr->index, trans_vars_);
          Expr dst_base_index = EliminateVarInExpr(op->index, trans_vars_);
          Var tt0 = Var("tt0");
          Var tt1 = Var("tt1");
          auto pre_copy = Store::make(
            pre_transpose_buffer_,
            Load::make(t_type_, src_ptr->buffer_var,
                       src_base_index + loop_h * TransAxisLen * ori_w + loop_w * TransAxisLen + tt1 * ori_w + tt0, 1),
            tt1 * TransAxisLen + tt0, 1);
          auto pre_l0 = For::make(tt0, 0, unit_width, ForType::Serial, DeviceAPI::None, pre_copy);
          auto pre_l1 = For::make(tt1, 0, TransAxisLen, ForType::Serial, DeviceAPI::None, pre_l0);
          auto pre_attr = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("dma_copy"), pre_l1);
          Stmt trans_attr = Stmt();
          if (t_type_ == Float(16)) {
            auto transpose =
              Store::make(post_transpose_buffer_,
                          Load::make(t_type_, pre_transpose_buffer_, tt1 * TransAxisLen + tt0, 1), tt0 * 16 + tt1, 1);
            auto trans_l0 = For::make(tt0, 0, TransAxisLen, ForType::Serial, DeviceAPI::None, transpose);
            auto trans_l1 = For::make(tt1, 0, TransAxisLen, ForType::Serial, DeviceAPI::None, trans_l0);
            trans_attr = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("dma_copy_transpose"), trans_l1);
          } else {
            auto pre_cast_store = Store::make(
              pre_trans_cast_, Cast::make(Float(16), Load::make(t_type_, pre_transpose_buffer_, tt0, 1)), tt0, 1);
            auto pre_cast_for = For::make(tt0, 0, TransTotalSize, ForType::Serial, DeviceAPI::None, pre_cast_store);
            auto pre_cast_attr =
              AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("vec_single_cast"), pre_cast_for);

            auto transpose = Store::make(
              post_trans_cast_, Load::make(Float(16), pre_trans_cast_, tt1 * TransAxisLen + tt0, 1), tt0 * 16 + tt1, 1);
            auto trans_l0 = For::make(tt0, 0, TransAxisLen, ForType::Serial, DeviceAPI::None, transpose);
            auto trans_l1 = For::make(tt1, 0, TransAxisLen, ForType::Serial, DeviceAPI::None, trans_l0);
            auto trans_block =
              AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("dma_copy_transpose"), trans_l1);

            auto post_cast_store = Store::make(
              post_transpose_buffer_, Cast::make(t_type_, Load::make(Float(16), post_trans_cast_, tt0, 1)), tt0, 1);
            auto post_cast_for = For::make(tt0, 0, TransTotalSize, ForType::Serial, DeviceAPI::None, post_cast_store);
            auto post_cast_attr =
              AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("vec_single_cast"), post_cast_for);

            trans_attr = Block::make(Block::make(pre_cast_attr, trans_block), post_cast_attr);
          }
          auto post_copy =
            Store::make(op->buffer_var, Load::make(t_type_, post_transpose_buffer_, tt1 * TransAxisLen + tt0, 1),
                        dst_base_index + loop_w * TransAxisLen * ori_h + loop_h * TransAxisLen + tt1 * ori_h + tt0, 1);
          auto post_l0 = For::make(tt0, 0, TransAxisLen, ForType::Serial, DeviceAPI::None, post_copy);
          auto post_l1 = For::make(tt1, 0, unit_width, ForType::Serial, DeviceAPI::None, post_l0);
          auto post_attr = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("dma_copy"), post_l1);
          auto full_inner = Block::make(Block::make(pre_attr, trans_attr), post_attr);
          auto inner_w = For::make(loop_w, 0, ori_block_w, ForType::Serial, DeviceAPI::None, full_inner);
          if (ori_block_w.as<IntImm>() && ori_block_w.as<IntImm>()->value == 1) {
            std::unordered_map<const Variable *, Expr> init;
            init[loop_w.get()] = 0;
            inner_w = Simplify(Substitute(full_inner, init));
          }
          auto inner_h = For::make(loop_h, 0, ori_block_h, ForType::Serial, DeviceAPI::None, inner_w);
          if (ori_block_h.as<IntImm>() && ori_block_h.as<IntImm>()->value == 1) {
            std::unordered_map<const Variable *, Expr> init;
            init[loop_h.get()] = 0;
            inner_h = Simplify(Substitute(inner_w, init));
          }
          return inner_h;
        }
      }
    }
    return s;
  }

 private:
  bool is_candidate_{false};
  bool is_native_transpose_{false};
  bool is_block_transpose_{false};
  int align_value{FREE_ALIGN};
  Var align_buffer_;
  Array<Var> trans_vars_;
  Array<Var> loop_vars_;
  Array<Expr> loop_extends_;
  std::vector<const For *> remain_fors_;
  Type t_type_;
  Var pre_transpose_buffer_;
  Var pre_trans_cast_;
  Var post_trans_cast_;
  Var post_transpose_buffer_;
};

class ForVarUnique : public IRMutator {
 public:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    auto body = this->Mutate(op->body);
    if (var_maps_.count(op->loop_var.get())) {
      Var new_var = Var("ii" + std::to_string(++index_));
      std::unordered_map<const Variable *, Expr> value_map;
      value_map[op->loop_var.get()] = new_var;
      auto new_body = Substitute(body, value_map);
      var_maps_[new_var.get()] = 1;
      return For::make(new_var, op->min, op->extent, ForType::Serial, DeviceAPI::None, new_body);
    }
    var_maps_[op->loop_var.get()] = 1;
    return For::make(op->loop_var, op->min, op->extent, ForType::Serial, DeviceAPI::None, body);
  }

 private:
  std::unordered_map<const Variable *, int> var_maps_;
  int index_{0};
};

class LoopReorder : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>()) {
      in_insn_ = true;
      pragma_ = op->value.as<StringImm>()->value;
      for_map_.clear();
      ori_vars_ = {};
      var_order_.clear();
      auto ret = this->Mutate(op->body);
      in_insn_ = false;
      if (!has_changed_) {
        return s;
      }
      if (var_order_.empty()) {
        ret = AttrStmt::make(op->node, op->attr_key, op->value, ret);
        for (size_t i = 0; i < ori_vars_.size(); ++i) {
          CHECK_GT(for_map_.count(ori_vars_[i].get()), 0);
          auto ptr = for_map_[ori_vars_[i].get()];
          ret = For::make(ptr->loop_var, ptr->min, ptr->extent, ptr->for_type, ptr->device_api, ret);
        }
        return ret;
      }
      for (size_t i = 0; i < var_order_.size(); ++i) {
        CHECK_GT(for_map_.count(var_order_[i].get()), 0);
        auto ptr = for_map_[var_order_[i].get()];
        ret = For::make(ptr->loop_var, ptr->min, ptr->extent, ptr->for_type, ptr->device_api, ret);
      }
      ret = AttrStmt::make(op->node, op->attr_key, op->value, ret);
      return ret;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (in_insn_) {
      for_map_[(op->loop_var).get()] = op;
      ori_vars_.push_back(op->loop_var);
      auto body = this->Mutate(op->body);
      return body;
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    int dst_pos = GetVectorizedVarPosition(op->index, ori_vars_);
    int len = static_cast<int>(ori_vars_.size());

    std::vector<const Load *> srcs;
    auto get_loads = [&srcs](const NodeRef &node) {
      if (const auto v = node.as<Load>()) {
        srcs.push_back(v);
      }
    };
    PostOrderVisit(op->value, get_loads);

    bool same_pos = true;
    std::vector<int> srcs_pos;
    for (int i = 0; i < static_cast<int>(srcs.size()); ++i) {
      int temp_pos = GetVectorizedVarPosition(srcs[i]->index, ori_vars_);
      srcs_pos.push_back(temp_pos);
      if (temp_pos != dst_pos) {
        same_pos = false;
      }
    }

    has_changed_ = false;
    if (dst_pos >= 0 && len >= 2 && dst_pos != (len - 1) && (same_pos || pragma_ == "broadcast")) {
      // Src Load empty; all Load and Dst has the same key axis; broadcast
      has_changed_ = true;
      var_order_.push_back(ori_vars_[dst_pos]);
      for (int i = len - 1; i >= 0; i--) {
        if (i != dst_pos) {
          var_order_.push_back(ori_vars_[i]);
        }
      }
    } else if (pragma_.find("reduce") != pragma_.npos && len >= 2 && srcs_pos[0] != (len - 1)) {
      // based on dst key axis: reduce
      has_changed_ = true;
      var_order_.push_back(ori_vars_[srcs_pos[0]]);
      for (int i = len - 1; i >= 0; i--) {
        if (i != srcs_pos[0]) {
          var_order_.push_back(ori_vars_[i]);
        }
      }
    }
    return s;
  }

 private:
  std::unordered_map<const Variable *, const For *> for_map_;
  std::vector<Var> var_order_;
  Array<Var> ori_vars_;
  bool has_changed_{false};
  bool in_insn_{false};
  std::string pragma_;
};

class IfReorder : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
        !exclude_align_analyze_list.count(op->value.as<StringImm>()->value)) {
      in_insn_ = true;
      for_vars_.clear();
      if_vars_.clear();
      for_vec_.clear();
      if_vec_.clear();
      auto body = this->Mutate(op->body);
      in_insn_ = false;
      if (!if_vec_.empty()) {
        Stmt new_s = AttrStmt::make(op->node, op->attr_key, op->value, body);
        for (auto if_op : if_vec_) {
          new_s = IfThenElse::make(if_op->condition, new_s);
        }
        for (auto for_op = for_vec_.rbegin(); for_op != for_vec_.rend(); ++for_op) {
          bool find_flag = false;
          for (auto for_iter = for_vars_.begin(); for_iter != for_vars_.end(); ++for_iter) {
            if (Equal((*for_iter), (*for_op)->loop_var)) {
              find_flag = true;
              break;
            }
          }
          if (find_flag) {
            new_s = For::make((*for_op)->loop_var, (*for_op)->min, (*for_op)->extent, ForType::Serial, DeviceAPI::None,
                              new_s);
          }
        }
        return new_s;
      }
      return s;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (in_insn_) {
      for_vec_.push_back(op);
      for_vars_.push_back(op->loop_var);
      Stmt body = this->Mutate(op->body);
      std::vector<Var>::iterator for_iter;
      for (for_iter = for_vars_.begin(); for_iter != for_vars_.end(); ++for_iter) {
        if (Equal((*for_iter), op->loop_var)) {
          break;
        }
      }

      if (!if_vec_.empty()) {
        std::vector<Var>::iterator if_iter;
        bool find_flag = false;
        for (if_iter = if_vars_.begin(); if_iter != if_vars_.end(); ++if_iter) {
          if (Equal((*if_iter), op->loop_var)) {
            find_flag = true;
            break;
          }
        }
        if (find_flag) {
          return body;
        }
        for_vars_.erase(for_iter);
        return For::make(op->loop_var, op->min, op->extent, ForType::Serial, DeviceAPI::None, body);
      }
      for_vars_.erase(for_iter);
      return For::make(op->loop_var, op->min, op->extent, ForType::Serial, DeviceAPI::None, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (in_insn_) {
      if_vec_.push_back(op);
      for (auto loop_var : for_vars_) {
        if (HasVars(op->condition, loop_var)) {
          if_vars_.push_back(loop_var);
        }
      }
      Stmt body = this->Mutate(op->then_case);
      return body;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (in_insn_) {
      return s;
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool in_insn_{false};
  std::vector<const IfThenElse *> if_vec_;
  std::vector<Var> if_vars_;
  std::vector<Var> for_vars_;
  std::vector<const For *> for_vec_;
  std::vector<const For *> before_if_;
};

}  // namespace ir
}  // namespace akg

#endif  // IR_TRANSFORM_H_