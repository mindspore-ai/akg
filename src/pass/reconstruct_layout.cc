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

#include <dmlc/common.h>
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_pass.h>
#include <tvm/buffer.h>
#include <tvm/target_info.h>
#include <tvm/build_module.h>
#include <tvm/runtime/device_api.h>

#include <unordered_map>

#include "common/common_util.h"
#include "pass/utils.h"
#include "ir_pass.h"

namespace akg {
namespace ir {

class TensorCoreMatcher : public IRVisitor {
 public:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == air::ir::attr::pragma_tensor_core) {
      tensor_core_on_ = true;
    } else if (op->attr_key == air::ir::attr::realize_scope) {
      auto pos = op->value.as<StringImm>()->value.find("wmma.matrix_");
      if (pos != std::string::npos) {
        wmma_matrix_.insert(std::make_pair(
            akg::common::GetGlobalName(op->node.as<PlaceholderOpNode>()->name), op->value.as<StringImm>()->value));
      }
    } else if (op->attr_key == "batch_axis_num") {
      batch_axis_num_ = op->value.as<IntImm>()->value;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Realize *op) final {
    if (tensor_core_on_ && op->func->func_name().find("shared") != std::string::npos) {
      std::vector<Expr> tmp;
      tmp.reserve(op->bounds.size() - batch_axis_num_);
      for (size_t i = batch_axis_num_; i < op->bounds.size(); i++) {
        tmp.push_back(op->bounds[i]->extent);
      }
      shared_bound_[akg::common::GetGlobalName(op->func->func_name())] = tmp;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Evaluate *op) final {
    if (const auto call = op->value.as<Call>()) {
      if (tensor_core_on_ && call->is_intrinsic(air::ir::intrinsic::tvm_load_matrix_sync)) {
        Expr warp_tile_m = call->args[1];
        Expr warp_tile_n = call->args[2];
        Expr warp_tile_k = call->args[3];
        auto it_matrix = wmma_matrix_.find(akg::common::GetGlobalName(call->args[0].as<Variable>()->name_hint));
        if (it_matrix != wmma_matrix_.end()) {
            wmma_layout_.insert(std::make_pair(it_matrix->second, call->args[7].as<StringImm>()->value));
          if (warp_tile_m.as<IntImm>()->value == 16 && warp_tile_n.as<IntImm>()->value == 16 && 
              warp_tile_k.as<IntImm>()->value == 8) {
            auto pair_name = std::pair<std::string, std::string>(it_matrix->second, call->args[7].as<StringImm>()->value);
            std::vector<Expr> tmp;
            tmp.reserve(2);
            if (it_matrix->second == "wmma.matrix_a" && call->args[7].as<StringImm>()->value == "row_major") {
              tmp.emplace_back(warp_tile_m);
              tmp.emplace_back(warp_tile_k);
            } else if (it_matrix->second == "wmma.matrix_a" && call->args[7].as<StringImm>()->value == "col_major") {
              tmp.emplace_back(warp_tile_k);
              tmp.emplace_back(warp_tile_m);
            } else if (it_matrix->second == "wmma.matrix_b" && call->args[7].as<StringImm>()->value == "row_major") {
              tmp.emplace_back(warp_tile_k);
              tmp.emplace_back(warp_tile_n);
            } else if (it_matrix->second == "wmma.matrix_b" && call->args[7].as<StringImm>()->value == "col_major") {
              tmp.emplace_back(warp_tile_n);
              tmp.emplace_back(warp_tile_k);
            } else {
              LOG(FATAL) << "Not supported layout " << call->args[7].as<StringImm>()->value << " for " << it_matrix->second;
            }
            tile_size_[pair_name] = tmp;
          }
        }
      }
    }

    IRVisitor::Visit_(op);
  }
  
  inline bool Matched() { return tensor_core_on_;}

  friend class SharedReconstruction;

 private:
  bool tensor_core_on_{false};
  unsigned int batch_axis_num_{0};
  std::unordered_map<std::string, std::string> wmma_matrix_;
  std::unordered_map<std::string, std::string> wmma_layout_;
  std::unordered_map<std::string, std::vector<Expr>> shared_bound_;
  std::unordered_map<std::pair<std::string, std::string>, std::vector<Expr>, PairHash> tile_size_;
};

class SharedReconstruction : public IRMutator {
 public:
  explicit SharedReconstruction(const TensorCoreMatcher &tensorcore_matcher)
      : batch_axis_num_(tensorcore_matcher.batch_axis_num_),
        wmma_matrix_(tensorcore_matcher.wmma_matrix_),
        wmma_layout_(tensorcore_matcher.wmma_layout_),
        shared_bound_(tensorcore_matcher.shared_bound_),
        tile_size_(tensorcore_matcher.tile_size_) {}

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Provide>();
    auto it_matrix = wmma_matrix_.find(akg::common::GetGlobalName(op->func->func_name()));
    if (it_matrix != wmma_matrix_.end() && op->func->func_name().find("shared") != std::string::npos) {
      auto it_layout = wmma_layout_.find(it_matrix->second);
      auto pair_name = std::pair<std::string, std::string>(it_layout->first, it_layout->second);
      auto it_tile = tile_size_.find(pair_name);
      auto it_bound = shared_bound_.find(akg::common::GetGlobalName(op->func->func_name()));
      Array<Expr> fuse_args;
      if (it_tile != tile_size_.end() && it_bound != shared_bound_.end()) {
        Array<Expr> split_args;
        split_args.push_back(Div::make(op->args[batch_axis_num_], it_tile->second[0]));
        split_args.push_back(Div::make(op->args[op->args.size() - 1], it_tile->second[1]));
        split_args.push_back(Mod::make(op->args[batch_axis_num_], it_tile->second[0]));
        split_args.push_back(Mod::make(op->args[op->args.size() - 1], it_tile->second[1]));
        for (size_t i = 0; i < op->args.size(); i++) {
          Expr new_arg = op->args[i];
          if (i == batch_axis_num_) {
            new_arg = split_args[0];
          }
          if (i == op->args.size() - 2) {
            new_arg = Add::make(Mul::make(new_arg, 
              Div::make(it_bound->second.back(), it_tile->second[1])), split_args[1]);
          }
          if (i == op->args.size() - 1) {
            new_arg = Add::make(Mul::make(split_args[2], it_tile->second[1]), split_args[3]);
          }
          fuse_args.push_back(new_arg);
        }
      } else {
        for (size_t i = 0; i < op->args.size(); i++) {
          fuse_args.push_back(op->args[i]);
        }
      }
      return Provide::make(op->func, op->value_index, op->value, fuse_args);
    }
    return stmt;
  }

  Stmt Mutate_(const Evaluate *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Evaluate>();
    auto wmma_op = op->value.as<Call>();
    if (wmma_op->is_intrinsic(air::ir::intrinsic::tvm_load_matrix_sync)) {
      auto it_matrix = wmma_matrix_.find(akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint));
      if (it_matrix != wmma_matrix_.end()) {
        auto it_layout = wmma_layout_.find(it_matrix->second);
        auto it_bound = shared_bound_.find(akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint));
        if (it_bound == shared_bound_.end()) {
          LOG(FATAL) << "Insufficient arguments for shared memory tensor " << 
              akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint);
        }
        Expr inner_bound = it_bound->second.back();
        for (size_t i = 1; i < it_bound->second.size() - 1; i++) {
          inner_bound = inner_bound * it_bound->second[i];
        }
        if (wmma_op->args[1].as<IntImm>()->value == 16 && wmma_op->args[2].as<IntImm>()->value == 16 && 
            wmma_op->args[3].as<IntImm>()->value == 8) {
          if ((it_layout->first == "wmma.matrix_a" && it_layout->second == "row_major") ||
            (it_layout->first == "wmma.matrix_b" && it_layout->second == "col_major")) {
            if (inner_bound.as<IntImm>()->value <= 16) {
              shared_offset_[akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint)] = IntImm::make(Int(32), 32);
            } else if (inner_bound.as<IntImm>()->value <= 40) {
              shared_offset_[akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint)] = IntImm::make(Int(32), 16);
            } else {
              shared_offset_[akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint)] = IntImm::make(Int(32), 8);
            }
            offset_expr_ = IntImm::make(Int(32), 8);
          } else if ((it_layout->first == "wmma.matrix_a" && it_layout->second == "col_major") ||
              (it_layout->first == "wmma.matrix_b" && it_layout->second == "row_major")) {
            if (inner_bound.as<IntImm>()->value <= 32) {
              shared_offset_[akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint)] = IntImm::make(Int(32), 32);
            } else {
              shared_offset_[akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint)] = IntImm::make(Int(32), 16);
            }
            offset_expr_ = IntImm::make(Int(32), 16);                        
          } else {
            LOG(FATAL) << "Not supported layout " << it_layout->second << " for " << it_layout->first;
          }
        } else {
          shared_offset_[akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint)] = IntImm::make(Int(32), 16);
          offset_expr_ = IntImm::make(Int(32), wmma_op->args[6].as<IntImm>()->value + 16);
        }
        auto shared_op = wmma_op->args[5].as<Call>();
        auto call_op = shared_op->args[0].as<Call>();
        Array<Expr> fuse_args;
        auto pair_name = std::pair<std::string, std::string>(it_layout->first, it_layout->second);
        auto it_tile = tile_size_.find(pair_name);
        if (it_tile != tile_size_.end() && it_bound != shared_bound_.end()) {
          Array<Expr> split_args;
          split_args.push_back(Div::make(call_op->args[batch_axis_num_], it_tile->second[0]));
          split_args.push_back(Div::make(call_op->args[call_op->args.size() - 1], it_tile->second[1]));
          split_args.push_back(Mod::make(call_op->args[batch_axis_num_], it_tile->second[0]));
          split_args.push_back(Mod::make(call_op->args[call_op->args.size() - 1], it_tile->second[1]));
          for (size_t i = 0; i < call_op->args.size(); i++) {
            Expr new_arg = call_op->args[i];
            if (i == batch_axis_num_) {
              new_arg = split_args[0];
            }
            if (i == call_op->args.size() - 2) {
              new_arg = Add::make(Mul::make(new_arg, 
                Div::make(it_bound->second.back(), it_tile->second[1])), split_args[1]);
            }
            if (i == call_op->args.size() - 1) {
              new_arg = Add::make(Mul::make(split_args[2], it_tile->second[1]), split_args[3]);
            }
            fuse_args.push_back(new_arg);
          }
        } else {
          for (size_t i = 0; i < call_op->args.size(); i++) {
            fuse_args.push_back(call_op->args[i]);
          }
        }
        Array<Expr> split_args_in;
        split_args_in.push_back(
            Call::make(call_op->type, call_op->name, fuse_args, Call::CallType::Halide, call_op->func, call_op->value_index));
        auto new_shared_op = Call::make(
            shared_op->type, shared_op->name, split_args_in, shared_op->call_type, shared_op->func, shared_op->value_index);
        return Evaluate::make(
            Call::make(Handle(), air::ir::intrinsic::tvm_load_matrix_sync, {
                wmma_op->args[0], wmma_op->args[1], wmma_op->args[2], 
                wmma_op->args[3], wmma_op->args[4], new_shared_op,
                offset_expr_, wmma_op->args[7]}, Call::Intrinsic));
      }
    }
    return stmt;
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Realize>();
    if (!op) {
      return stmt;
    }
    auto it_matrix = wmma_matrix_.find(akg::common::GetGlobalName(op->func->func_name()));
    if (it_matrix != wmma_matrix_.end() && op->func->func_name().find("shared") != std::string::npos) {
      auto it_layout = wmma_layout_.find(it_matrix->second);
      auto pair_name = std::pair<std::string, std::string>(it_layout->first, it_layout->second);
      auto offset = shared_offset_.find(akg::common::GetGlobalName(op->func->func_name()));
      auto it_tile = tile_size_.find(pair_name);
      if (it_tile != tile_size_.end()) {
        Region new_bounds;
        for (size_t i = 0; i < op->bounds.size(); i++) {
          Expr new_extent = op->bounds[i]->extent;
          if (i == batch_axis_num_) {
            new_extent = new_extent / it_tile->second[0];
          }
          if (i == op->bounds.size() - 2) {
            new_extent = new_extent * (op->bounds[op->bounds.size() - 1]->extent / it_tile->second[1]);
          }
          if (i == op->bounds.size() - 1) {
            new_extent = offset->second + (it_tile->second[0] * it_tile->second[1]);
          }
          new_bounds.push_back(Range::make_by_min_extent(op->bounds[i]->min, new_extent));
        }
        return Realize::make(op->func, op->value_index, op->type, new_bounds, op->condition, op->body);
      } else {
        Region new_bounds;
        for (size_t i = 0; i < op->bounds.size() - 1; ++i) {
          new_bounds.push_back(Range::make_by_min_extent(op->bounds[i]->min, op->bounds[i]->extent));
        }
        new_bounds.push_back(
            Range::make_by_min_extent(op->bounds[op->bounds.size() - 1]->min, 
            op->bounds[op->bounds.size() - 1]->extent + offset->second));
        return Realize::make(op->func, op->value_index, op->type, new_bounds, op->condition, op->body);
      }
    }
    return stmt;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "batch_axis_num") {
      return this->Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  unsigned int batch_axis_num_{0};
  std::unordered_map<std::string, std::string> wmma_matrix_;
  std::unordered_map<std::string, std::string> wmma_layout_;
  std::unordered_map<std::string, std::vector<Expr>> shared_bound_;    
  std::unordered_map<std::pair<std::string, std::string>, std::vector<Expr>, PairHash> tile_size_;
  std::unordered_map<std::string, Expr> shared_offset_;
  Expr offset_expr_;
};

class PackedMatcher : public IRVisitor {
 public:
  explicit PackedMatcher() {}

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "pack_a" || op->attr_key == "pack_b") {
      is_matched_ = true;
    }
    IRVisitor::Visit_(op);
  }

  inline bool Matched() { return is_matched_; }
 private:
  bool is_matched_ = false;
};

class PackedReconstruction : public IRMutator {
 public:
  explicit PackedReconstruction() {}

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pack_a") {
      int value = op->value.as<IntImm>()->value;
      a_block_size_ = value;
    } else if (op->attr_key == "pack_b") {
      int value = op->value.as<IntImm>()->value;
      b_block_size_ = value;
    } else {
      auto value_ptr = op->value.as<StringImm>();
      if (!value_ptr) {
        return IRMutator::Mutate_(op, s);
      }
      auto value = value_ptr->value;
      if (value == "col_major_matrix_b") {
        b_col_major_ = true;
        b_n_len_ = b_extent0_;
        b_k_len_ = b_extent1_;
      } else if(value == "row_major_matrix_b") {
        b_col_major_ = false;
        b_k_len_ = b_extent0_;
        b_n_len_ = b_extent1_;
      } else if(value == "col_major_matrix_a") {
        a_col_major_ = true;
        a_k_len_ = a_extent0_;
        a_m_len_ = a_extent1_;
      } else if(value == "row_major_matrix_a") {
        a_col_major_ = false;
        a_m_len_ = a_extent0_;
        a_k_len_ = a_extent1_;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    auto block_ptr = op->body.as<Block>();
    if (!block_ptr) {
      return IRMutator::Mutate_(op, s);
    }
    auto attr_ptr = block_ptr->first.as<AttrStmt>();
    if (!attr_ptr) {
      return IRMutator::Mutate_(op, s);
    }
    auto attr_key = attr_ptr->attr_key;
    auto matrix_name = attr_key.size() >= 8 ? attr_key.substr(attr_key.size() - 8) : "";
    auto local_name = op->func->func_name();
    if (matrix_name == "matrix_a") {
      a_local_name_ = local_name;
      a_dtype_ = op->type;
      a_extent0_ = op->bounds[0]->extent;
      a_extent1_ = op->bounds[1]->extent;
    }
    if (matrix_name == "matrix_b") {
      b_local_name_ = local_name;
      b_dtype_ = op->type;
      b_extent0_ = op->bounds[0]->extent;
      b_extent1_ = op->bounds[1]->extent;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto name = op->func->func_name();
    if (name == a_local_name_) {
      auto r = op->args[0];
      auto c = op->args[1];
      int block_size = b_block_size_;
      Expr new_ld = a_col_major_ ? a_m_len_ : a_k_len_;
      Expr new_idx = get_new_idx(a_col_major_, block_size, r, c, a_k_len_, a_m_len_);
      return Provide::make(op->func, op->value_index, op->value,
        {floordiv(new_idx, new_ld), indexmod(new_idx, new_ld)});
    }
    if (name == b_local_name_) {
      auto r = op->args[0];
      auto c = op->args[1];
      int block_size = a_block_size_;
      Expr new_ld = b_col_major_ ? b_k_len_ : b_n_len_;
      Expr new_idx = get_new_idx(!b_col_major_, block_size, r, c, b_k_len_, b_n_len_);
      return Provide::make(op->func, op->value_index, op->value,
        {floordiv(new_idx, new_ld), indexmod(new_idx, new_ld)});
    }
    return s;
  }

 private:
  Expr get_new_idx(bool km, int block_size, Expr r, Expr c, Expr k_len, Expr m_len) {
    auto block_num = ceil(div(m_len, block_size * 1.0));
    auto m_rest = indexmod(m_len, block_size);
    Expr k = km ? r : c;
    Expr m = km ? c : r;
    auto block_idx = floordiv(m, block_size);
    auto mm = indexmod(m, block_size);
    Type int_type = Int(32);
    auto in_last_block = block_idx == block_num - 1;
    Expr a = 0;
    Expr b = cast(int_type, m_rest == 0 || in_last_block == 0) * block_size;
    int small_block = block_size / 2;
    while (small_block > 0) {
      auto has_small_block = in_last_block && (m_rest & small_block) > 0;
      b = b + cast(int_type, has_small_block && mm >= a && mm < a + small_block && b == 0) * small_block;
      a = a + cast(int_type, has_small_block && b == 0) * small_block;
      small_block /= 2;
    }
    auto new_idx = (block_idx * block_size + a) * k_len + mm - a + b * k;
    return new_idx;
  }

  std::string a_local_name_ = "";
  bool a_col_major_ = false;
  int a_block_size_ = 8;
  Type a_dtype_;
  Expr a_k_len_;
  Expr a_m_len_;
  Expr a_extent0_;
  Expr a_extent1_;

  std::string b_local_name_ = "";
  bool b_col_major_ = false;
  int b_block_size_ = 4;
  Type b_dtype_;
  Expr b_k_len_;
  Expr b_n_len_;
  Expr b_extent0_;
  Expr b_extent1_;
};

Stmt ReconstructLayout(const Stmt &stmt) {
  PackedMatcher packed_matcher;
  packed_matcher.Visit(stmt);
  Stmt stmt_out;
  if (packed_matcher.Matched()) {
    PackedReconstruction packed_reconstruction;
    stmt_out = packed_reconstruction.Mutate(stmt);
  } else {
    stmt_out = stmt;
  }
  
  TensorCoreMatcher tensorcore_matcher;
  tensorcore_matcher.Visit(stmt_out);
  if (!tensorcore_matcher.Matched()) {
    return stmt_out;
  }
  return SharedReconstruction(tensorcore_matcher).Mutate(stmt_out);
}

} // namespace ir
} // namespace akg
