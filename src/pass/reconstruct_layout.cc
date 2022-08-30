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

static constexpr auto MATRIX_A = "matrix_a";
static constexpr auto MATRIX_B = "matrix_b";
static constexpr auto MATRIX_C = "matrix_c";
static constexpr auto GEMM_PACK_A = "pack_a";
static constexpr auto GEMM_PACK_B = "pack_b";
static constexpr auto PROMOTE_TRANSPOSE = "promoted_transpose";
static constexpr auto MATRIX_TRANSPOSE = "MatrixTranspose";
static constexpr auto LOCAL = "local";
static constexpr auto REGISTER = "register";
static constexpr auto ROW_MAJOR = "row_major";
static constexpr auto TRANS_A = "row_major_matrix_a";
static constexpr auto TRANS_B = "col_major_matrix_b";
static constexpr auto PREPARE_PACK = "prepare_pack";
static constexpr auto PACK_A_SIZE = 4;
static constexpr auto PACK_B_SIZE = 24;
static constexpr size_t LOOP_NUM = 2;
static constexpr auto NUM_2 = 2;
static constexpr auto NUM_3 = 3;
static constexpr auto NUM_4 = 4;
static constexpr auto NUM_5 = 5;
static constexpr auto NUM_6 = 6;
static constexpr auto NUM_7 = 7;
static constexpr auto NUM_8 = 8;
static constexpr auto NUM_9 = 9;
static constexpr auto NUM_10 = 10;
static constexpr auto NUM_16 = 16;
static constexpr auto NUM_32 = 32;
static constexpr auto NUM_40 = 40;
static constexpr auto INT32 = 32;

class TensorCoreMatcher : public IRVisitor {
 public:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == air::ir::attr::pragma_tensor_core) {
      tensor_core_on_ = true;
    } else if (op->attr_key == air::ir::attr::realize_scope) {
      auto pos = op->value.as<StringImm>()->value.find("wmma.matrix_");
      if (pos != std::string::npos) {
        wmma_matrix_.insert(std::make_pair(akg::common::GetGlobalName(op->node.as<PlaceholderOpNode>()->name),
                                           op->value.as<StringImm>()->value));
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
        Expr warp_tile_n = call->args[NUM_2];
        Expr warp_tile_k = call->args[NUM_3];
        auto it_matrix = wmma_matrix_.find(akg::common::GetGlobalName(call->args[0].as<Variable>()->name_hint));
        if (it_matrix != wmma_matrix_.end()) {
          wmma_layout_.insert(std::make_pair(it_matrix->second, call->args[NUM_7].as<StringImm>()->value));
          if (warp_tile_m.as<IntImm>()->value == NUM_16 && warp_tile_n.as<IntImm>()->value == NUM_16 &&
              warp_tile_k.as<IntImm>()->value == NUM_8) {
            auto pair_name =
              std::pair<std::string, std::string>(it_matrix->second, call->args[NUM_7].as<StringImm>()->value);
            std::vector<Expr> tmp;
            tmp.reserve(NUM_2);
            if (it_matrix->second == "wmma.matrix_a" && call->args[NUM_7].as<StringImm>()->value == "row_major") {
              tmp.emplace_back(warp_tile_m);
              tmp.emplace_back(warp_tile_k);
            } else if (it_matrix->second == "wmma.matrix_a" &&
                       call->args[NUM_7].as<StringImm>()->value == "col_major") {
              tmp.emplace_back(warp_tile_k);
              tmp.emplace_back(warp_tile_m);
            } else if (it_matrix->second == "wmma.matrix_b" &&
                       call->args[NUM_7].as<StringImm>()->value == "row_major") {
              tmp.emplace_back(warp_tile_k);
              tmp.emplace_back(warp_tile_n);
            } else if (it_matrix->second == "wmma.matrix_b" &&
                       call->args[NUM_7].as<StringImm>()->value == "col_major") {
              tmp.emplace_back(warp_tile_n);
              tmp.emplace_back(warp_tile_k);
            } else {
              LOG(FATAL) << "Not supported layout " << call->args[NUM_7].as<StringImm>()->value << " for "
                         << it_matrix->second;
            }
            tile_size_[pair_name] = tmp;
          }
        }
      }
    }

    IRVisitor::Visit_(op);
  }

  inline bool Matched() { return tensor_core_on_; }

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
          if (i == op->args.size() - NUM_2) {
            new_arg =
              Add::make(Mul::make(new_arg, Div::make(it_bound->second.back(), it_tile->second[1])), split_args[1]);
          }
          if (i == op->args.size() - 1) {
            new_arg = Add::make(Mul::make(split_args[NUM_2], it_tile->second[1]), split_args[NUM_3]);
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
          LOG(FATAL) << "Insufficient arguments for shared memory tensor "
                     << akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint);
        }
        Expr inner_bound = it_bound->second.back();
        for (size_t i = 1; i < it_bound->second.size() - 1; i++) {
          inner_bound = inner_bound * it_bound->second[i];
        }
        if (wmma_op->args[1].as<IntImm>()->value == NUM_16 && wmma_op->args[NUM_2].as<IntImm>()->value == NUM_16 &&
            wmma_op->args[NUM_3].as<IntImm>()->value == NUM_8) {
          if ((it_layout->first == "wmma.matrix_a" && it_layout->second == "row_major") ||
              (it_layout->first == "wmma.matrix_b" && it_layout->second == "col_major")) {
            if (inner_bound.as<IntImm>()->value <= NUM_16) {
              shared_offset_[akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint)] =
                IntImm::make(Int(INT32), NUM_32);
            } else if (inner_bound.as<IntImm>()->value <= NUM_40) {
              shared_offset_[akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint)] =
                IntImm::make(Int(INT32), NUM_16);
            } else {
              shared_offset_[akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint)] =
                IntImm::make(Int(INT32), NUM_8);
            }
            offset_expr_ = IntImm::make(Int(INT32), NUM_8);
          } else if ((it_layout->first == "wmma.matrix_a" && it_layout->second == "col_major") ||
                     (it_layout->first == "wmma.matrix_b" && it_layout->second == "row_major")) {
            if (inner_bound.as<IntImm>()->value <= NUM_32) {
              shared_offset_[akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint)] =
                IntImm::make(Int(INT32), NUM_32);
            } else {
              shared_offset_[akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint)] =
                IntImm::make(Int(INT32), NUM_16);
            }
            offset_expr_ = IntImm::make(Int(INT32), NUM_16);
          } else {
            LOG(FATAL) << "Not supported layout " << it_layout->second << " for " << it_layout->first;
          }
        } else {
          shared_offset_[akg::common::GetGlobalName(wmma_op->args[0].as<Variable>()->name_hint)] =
            IntImm::make(Int(INT32), NUM_16);
          offset_expr_ = IntImm::make(Int(INT32), wmma_op->args[NUM_6].as<IntImm>()->value + NUM_16);
        }
        auto shared_op = wmma_op->args[NUM_5].as<Call>();
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
            if (i == call_op->args.size() - NUM_2) {
              new_arg =
                Add::make(Mul::make(new_arg, Div::make(it_bound->second.back(), it_tile->second[1])), split_args[1]);
            }
            if (i == call_op->args.size() - 1) {
              new_arg = Add::make(Mul::make(split_args[NUM_2], it_tile->second[1]), split_args[NUM_3]);
            }
            fuse_args.push_back(new_arg);
          }
        } else {
          for (size_t i = 0; i < call_op->args.size(); i++) {
            fuse_args.push_back(call_op->args[i]);
          }
        }
        Array<Expr> split_args_in;
        split_args_in.push_back(Call::make(call_op->type, call_op->name, fuse_args, Call::CallType::Halide,
                                           call_op->func, call_op->value_index));
        auto new_shared_op = Call::make(shared_op->type, shared_op->name, split_args_in, shared_op->call_type,
                                        shared_op->func, shared_op->value_index);
        return Evaluate::make(
          Call::make(Handle(), air::ir::intrinsic::tvm_load_matrix_sync,
                     {wmma_op->args[0], wmma_op->args[1], wmma_op->args[NUM_2], wmma_op->args[NUM_3],
                      wmma_op->args[NUM_4], new_shared_op, offset_expr_, wmma_op->args[NUM_7]},
                     Call::Intrinsic));
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
          if (i == op->bounds.size() - NUM_2) {
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
        new_bounds.push_back(Range::make_by_min_extent(op->bounds[op->bounds.size() - 1]->min,
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

class CPULocalMatcher : public IRVisitor {
 public:
  explicit CPULocalMatcher() {}

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == GEMM_PACK_A || op->attr_key == GEMM_PACK_B) {
      is_matched_ = true;
      return;
    }
    IRVisitor::Visit_(op);
  }

  inline bool Matched() { return is_matched_; }

 private:
  bool is_matched_ = false;
};

class TransReadMutator : public IRMutator {
 public:
  explicit TransReadMutator(Tensor tensor) : tensor_(tensor) {}

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    return Provide::make(tensor_->op, op->value_index, op->value, args_);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    args_.push_back(op->loop_var);
    return IRMutator::Mutate_(op, s);
  }

 private:
  Tensor tensor_;
  Array<Expr> args_;
};

class TransWriteMutator : public IRMutator {
 public:
  explicit TransWriteMutator(Tensor tensor) : tensor_(tensor) {}

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto index = args_[1] * tensor_->shape[0] + args_[0];
    auto value = tensor_(Array<Expr>{floordiv(index, tensor_->shape[1]), indexmod(index, tensor_->shape[1])});
    return Provide::make(op->func, op->value_index, value, op->args);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    args_.push_back(op->loop_var);
    auto body = IRMutator::Mutate(op->body);
    if (auto second = body.as<For>()) {
      auto stmt = For::make(op->loop_var, op->min, op->extent, second->for_type, second->device_api, second->body);
      return For::make(second->loop_var, second->min, second->extent, op->for_type, op->device_api, stmt);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  Tensor tensor_;
  Array<Expr> args_;
};

class CPULocalReconstruction : public IRMutator {
 public:
  explicit CPULocalReconstruction() {}

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == GEMM_PACK_A) {
      a_pack_size_ = op->value.as<IntImm>()->value;
      a_block_size_ = a_pack_size_;
      return IRMutator::Mutate(op->body);
    } else if (op->attr_key == GEMM_PACK_B) {
      b_pack_size_ = op->value.as<IntImm>()->value;
      b_block_size_ = b_pack_size_;
      return IRMutator::Mutate(op->body);
    } else if (op->attr_key == PROMOTE_TRANSPOSE) {
      return PromoteForTranspose(op);
    } else if (op->attr_key == PREPARE_PACK) {
      auto value = op->value.as<StringImm>()->value;
      CHECK(value.size() > NUM_10);
      matrix_b_ = value.substr(0, value.size() - NUM_10);
      b_trans_ = value.substr(value.size() - NUM_9) == ROW_MAJOR ? false : true;
      return IRMutator::Mutate(op->body);
    } else {
      auto value_ptr = op->value.as<StringImm>();
      if (!value_ptr) {
        return IRMutator::Mutate_(op, s);
      }
      auto value = value_ptr->value;
      if (value == TRANS_B) {
        b_trans_ = true;
        return IRMutator::Mutate(op->body);
      } else if (value == TRANS_A) {
        a_trans_ = true;
        return IRMutator::Mutate(op->body);
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
    auto matrix_name = attr_key.size() >= NUM_8 ? attr_key.substr(attr_key.size() - NUM_8) : "";
    auto local_name = op->func->func_name();
    if (matrix_name == MATRIX_A || matrix_name == MATRIX_B) {
      if (matrix_name == MATRIX_A) {
        a_func_ = op->func;
      } else {
        b_func_ = op->func;
      }
      auto body = IRMutator::Mutate(op->body);
      auto block_size = matrix_name == MATRIX_A ? a_block_size_ : b_block_size_;
      auto trans = matrix_name == MATRIX_A ? a_trans_ : b_trans_;
      int start_pos = std::max(static_cast<int>(op->bounds.size()) - NUM_2, 0);
      auto bound_n = trans ? op->bounds[start_pos] : op->bounds[start_pos + 1];
      auto bound_k = trans ? op->bounds[start_pos + 1] : op->bounds[start_pos];
      if (auto bound_int = bound_n->extent.as<IntImm>()) {
        if (static_cast<int>(bound_int->value) < block_size) {
          if (matrix_name == MATRIX_A) {
            a_block_size_ = bound_int->value;
          } else {
            b_block_size_ = bound_int->value;
          }
          block_size = bound_int->value;
        }
      }
      Region new_bounds;
      for (int i = 0; i < start_pos; ++i) {
        new_bounds.push_back(op->bounds[i]);
      }
      new_bounds.push_back(Range::make_by_min_extent(bound_n->min, floordiv(bound_n->extent, block_size)));
      new_bounds.push_back(bound_k);
      new_bounds.push_back(Range::make_by_min_extent(bound_n->min, block_size));
      return Realize::make(op->func, op->value_index, op->type, new_bounds, op->condition, body);
    } else if (!matrix_b_.empty() && matrix_name == MATRIX_C) {
      if (auto bound_int = op->bounds[op->bounds.size() - 1]->extent.as<IntImm>()) {
        if (static_cast<int>(bound_int->value) < b_pack_size_) {
          b_block_size_ = bound_int->value;
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (op->func == a_func_ || op->func == b_func_) {
      Array<Expr> new_args = GetNewArgs(op);
      auto provide = Provide::make(op->func, op->value_index, op->value, new_args);
      provide_ = provide;
      return provide;
    }
    auto provide = IRMutator::Mutate_(op, s);
    provide_ = provide;
    return provide;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->func == a_func_ || op->func == b_func_) {
      Array<Expr> new_args = GetNewArgs(op);
      return Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);
    } else if (op->func->func_name() == matrix_b_) {
      int block_size = b_block_size_;
      Array<Expr> new_args;
      int start_pos = std::max(static_cast<int>(op->args.size()) - NUM_2, 0);
      Expr n = b_trans_ ? op->args[start_pos] : op->args[start_pos + 1];
      Expr k = b_trans_ ? op->args[start_pos + 1] : op->args[start_pos];
      for (int i = 0; i < start_pos; ++i) {
        new_args.push_back(op->args[i]);
      }
      const Operation *operation = static_cast<const Operation *>(&op->func);
      auto tensor = operation->output(0);
      Expr K = b_trans_ ? tensor->shape[start_pos + 1] : tensor->shape[start_pos];
      Expr col = tensor->shape[tensor.ndim() - 1];
      Expr index = floordiv(n, b_pack_size_) * b_pack_size_ * K + k * block_size + indexmod(n, b_pack_size_);
      new_args.push_back(floordiv(index, col));
      new_args.push_back(indexmod(index, col));
      return Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);
    }
    return IRMutator::Mutate_(op, e);
  }

  template <class T>
  Array<Expr> GetNewArgs(const T *op) {
    Array<Expr> new_args;
    int block_size = op->func == a_func_ ? a_block_size_ : b_block_size_;
    bool trans = op->func == a_func_ ? a_trans_ : b_trans_;
    int start_pos = std::max(static_cast<int>(op->args.size()) - NUM_2, 0);
    Expr n = trans ? op->args[start_pos] : op->args[start_pos + 1];
    Expr k = trans ? op->args[start_pos + 1] : op->args[start_pos];

    for (int i = 0; i < start_pos; ++i) {
      new_args.push_back(op->args[i]);
    }
    new_args.push_back(floordiv(n, block_size));
    new_args.push_back(k);
    new_args.push_back(indexmod(n, block_size));
    return new_args;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    extents_.push_back(op->extent);
    fors_.push_back(op);
    return IRMutator::Mutate_(op, s);
  }

 private:
  Stmt PromoteForTranspose(const AttrStmt *op) {
    extents_.clear();
    auto stmt = IRMutator::Mutate(op->body);
    if (extents_.size() != LOOP_NUM) {
      return stmt;
    }

    Array<Expr> shapes;
    shapes.assign(extents_.begin(), extents_.end());
    extents_.clear();

    auto provide = provide_.as<Provide>();
    Tensor tensor = placeholder(shapes, provide->value.type(), provide->func->func_name() + "_" + REGISTER);
    Region bounds;

    auto read = TransReadMutator(tensor).Mutate(stmt);
    auto write = TransWriteMutator(tensor).Mutate(stmt);
    Array<Expr> indices;
    Array<Expr> args;
    args.push_back(make_zero(Int(INT32)));
    args.push_back(make_zero(Int(INT32)));
    for (size_t i = 0; i < tensor.ndim(); i++) {
      indices.push_back(make_zero(Int(INT32)));
      args.push_back(tensor->shape[i]);
    }
    Expr addr = Call::make(Handle(), air::ir::intrinsic::tvm_address_of, {tensor(indices)}, Call::PureIntrinsic);
    args.Set(0, addr);
    args.Set(1, addr);
    Expr matrix_trans = Call::make(Handle(), MATRIX_TRANSPOSE, args, Call::Intrinsic);
    auto block = Block::make({read, Evaluate::make(matrix_trans), write});
    for (auto j : tensor->shape) {
      bounds.push_back(Range::make_by_min_extent(Expr(0), j));
    }
    auto realize = Realize::make(tensor->op, tensor->value_index, tensor->dtype, bounds, const_true(1), block);
    return AttrStmt::make(tensor->op, air::ir::attr::realize_scope, Expr(LOCAL), realize);
  }

 private:
  FunctionRef a_func_;
  FunctionRef b_func_;
  int a_pack_size_{PACK_A_SIZE};
  int b_pack_size_{PACK_B_SIZE};
  int a_block_size_{PACK_A_SIZE};
  int b_block_size_{PACK_B_SIZE};
  bool a_trans_{false};
  bool b_trans_{false};
  std::vector<Expr> extents_;
  std::vector<const For *> fors_;
  Stmt provide_;
  std::string matrix_b_;
};

Stmt ReconstructLayout(const Stmt &stmt) {
  CPULocalMatcher cpu_local_matcher;
  cpu_local_matcher.Visit(stmt);
  if (cpu_local_matcher.Matched()) {
    return CPULocalReconstruction().Mutate(stmt);
  }

  TensorCoreMatcher tensorcore_matcher;
  tensorcore_matcher.Visit(stmt);
  if (tensorcore_matcher.Matched()) {
    return SharedReconstruction(tensorcore_matcher).Mutate(stmt);
  }
  return stmt;
}

}  // namespace ir
}  // namespace akg
