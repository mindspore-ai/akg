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
constexpr auto REDUCE_AREA_FLAG = "reduce_area";
constexpr auto VECTOR_LENGTH = "VECTOR_LENGTH";
constexpr auto REDUCE_PROVIDE = "REDUCE_PROVIDE";

struct ReductionData {
  Stmt body;
  Stmt isolate;
  const Provide *reduce_provide{nullptr};
  std::set<const Provide *> parallel_internal_reduce_provides{};
  const For *vector_parallel_for{nullptr};
  std::set<const For *> isolate_vector_for{};
  std::set<const Provide *> isolate_reduce_provide{};
  const For *unroll_for{nullptr};
  Type reduce_data_type_info;
  Tensor temp_tensor;
  Buffer temp_buffer;
  Expr reduce_dst;
  Expr reduce_input;
  Expr reduce_temp;
  Expr init_value;
  std::shared_ptr<ReductionData> outter_reduction_data;
  std::shared_ptr<ReductionData> next_change_area_data;
  std::shared_ptr<ReductionData> pre_reduction_data;
  int vectorize_size{4};
};

class IdentifyReduceChance : public IRVisitor {
 public:
  explicit IdentifyReduceChance(const Map<Tensor, Buffer> &extern_buffer) {
    for (const auto &kv : extern_buffer) {
      buffer_type_[kv.first->op->func_name()] = kv.second->dtype;
    }
  }
  using IRVisitor::Visit_;

  void Visit_(const AttrStmt *op) {
    if (op->attr_key == REDUCE_AREA_FLAG) {
      in_reduce_area_++;
      if (cur_reduce_data_.get() == nullptr) {
        reduce_datas_[reduce_index_] = std::make_shared<ReductionData>();
        cur_reduce_data_ = reduce_datas_[reduce_index_];
        reduce_index_++;
      } else {
        if (is_parallel_area_) {
          reduce_datas_[reduce_index_] = std::make_shared<ReductionData>();
          reduce_datas_[reduce_index_]->outter_reduction_data = cur_reduce_data_;
          cur_reduce_data_ = reduce_datas_[reduce_index_];
          reduce_index_++;
        } else {
          reduce_datas_[reduce_index_] = std::make_shared<ReductionData>();
          auto temp = cur_reduce_data_;
          cur_reduce_data_ = reduce_datas_[reduce_index_];
          cur_reduce_data_->pre_reduction_data = temp;
          reduce_index_++;
        }
      }
      IRVisitor::Visit_(op);
      if (cur_reduce_data_->outter_reduction_data) {
        cur_reduce_data_ = cur_reduce_data_->outter_reduction_data;
      } else if (cur_reduce_data_->pre_reduction_data) {
        cur_reduce_data_ = cur_reduce_data_->pre_reduction_data;
      }
      in_reduce_area_--;
    } else if (op->attr_key == VECTOR_LENGTH) {
      CHECK(op->value.as<IntImm>());
      vectorize_size_ = op->value.as<IntImm>()->value;
      IRVisitor::Visit_(op);
    } else if (op->attr_key == REDUCE_PROVIDE) {
      is_reduce_provide_ = true;
      IRVisitor::Visit_(op);
      is_reduce_provide_ = false;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const For *op) {
    if (in_reduce_area_ > 0) {
      switch (op->for_type) {
        case air::ir::ForType::Serial:
          break;
        case air::ir::ForType::Parallel:
          is_parallel_area_ = true;
          cur_reduce_data_->vector_parallel_for = op;
          IRVisitor::Visit_(op);
          is_parallel_area_ = false;
          return;
        case air::ir::ForType::Vectorized: {
          auto ex = op->extent;
          CHECK(ex.as<IntImm>());
          int value = ex.as<IntImm>()->value;
          if (cur_reduce_data_->vector_parallel_for == nullptr) {
            if (value < vectorize_size_) {
              cur_reduce_data_->isolate_vector_for.insert(op);
            } else {
              cur_reduce_data_->vector_parallel_for = op;
            }
          } else {
            if (value < vectorize_size_) {
              cur_reduce_data_->isolate_vector_for.insert(op);
            } else {
              cur_reduce_data_->next_change_area_data = std::make_shared<ReductionData>();
              auto temp = cur_reduce_data_;
              cur_reduce_data_ = cur_reduce_data_->next_change_area_data;
              cur_reduce_data_->outter_reduction_data = temp->outter_reduction_data;
              cur_reduce_data_->pre_reduction_data = temp->pre_reduction_data;
              cur_reduce_data_->vector_parallel_for = op;
            }
          }
          break;
        }
        case air::ir::ForType::Unrolled:

          if (cur_reduce_data_->unroll_for == nullptr) {
            cur_reduce_data_->unroll_for = op;
          } else {
            cur_reduce_data_->next_change_area_data = std::make_shared<ReductionData>();
            auto temp = cur_reduce_data_;
            cur_reduce_data_ = cur_reduce_data_->next_change_area_data;
            cur_reduce_data_->outter_reduction_data = temp->outter_reduction_data;
            cur_reduce_data_->pre_reduction_data = temp->pre_reduction_data;
            cur_reduce_data_->unroll_for = op;
          }
          break;
        case air::ir::ForType::Swizzled:
          LOG(WARNING) << "Swizzled type for is in reduce area";
          break;

        default:
          break;
      }
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Provide *op) {
    if (in_reduce_area_ > 0) {
      if (cur_reduce_data_->reduce_provide == nullptr) {
        if (is_reduce_provide_) {
          is_reduce_provide_ = false;
          if (buffer_type_.count(op->func->func_name())) {
            cur_reduce_data_->reduce_data_type_info = buffer_type_[op->func->func_name()];
          } else {
            LOG(FATAL) << "can not get the buffer type info";
          }

          cur_reduce_data_->reduce_provide = op;
          auto init_value = GetInitValue(op);
          CHECK(init_value.defined());
          cur_reduce_data_->init_value = init_value;
          cur_reduce_data_->vectorize_size = vectorize_size_;
        }
      } else {
        if (op->func == cur_reduce_data_->reduce_provide->func) {
          cur_reduce_data_->isolate_reduce_provide.insert(op);
        }
      }
    }

    if (op->value.as<FloatImm>() != nullptr || op->value.as<IntImm>() != nullptr ||
        op->value.as<UIntImm>() != nullptr) {
      init_values_[op] = op->value;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Realize *op) {
    if (buffer_type_.count(op->func->func_name())) {
      Visit(op->body);
    } else {
      buffer_type_[op->func->func_name()] = op->type;
      Visit(op->body);
    }
  }

  Expr GetInitValue(const Provide *op) {
    for (auto &i : init_values_) {
      if (op->func == i.first->func) {
        return i.second;
      }
    }
    return Expr();
  }

 public:
  std::map<int, std::shared_ptr<ReductionData>> reduce_datas_;

 private:
  int in_reduce_area_{0};
  int reduce_index_{1};
  std::shared_ptr<ReductionData> cur_reduce_data_;
  int vectorize_size_{4};
  bool is_parallel_area_{false};
  std::map<const Provide *, Expr> init_values_;
  bool is_reduce_provide_{false};
  std::map<std::string, Type> buffer_type_;
};

class DivideBlock {
 public:
  explicit DivideBlock(int vectorize_size) : vectorize_size_(vectorize_size) {}
  std::vector<Stmt> Run(Stmt stmt) {
    CHECK(stmt.as<Block>());
    auto blo = stmt.as<Block>();
    first_part = blo->first;
    auto temp = blo->rest;
    while (auto left_blo = temp.as<Block>()) {
      auto temp_first = left_blo->first;
      if (temp_first.as<Provide>()) {
        first_part = Block::make(first_part, temp_first);
      } else if (auto for_node = temp_first.as<For>()) {
        auto ex = for_node->extent;
        CHECK(ex.as<IntImm>());
        int value = ex.as<IntImm>()->value;
        if ((value < vectorize_size_) && (for_node->for_type == air::ir::ForType::Vectorized)) {
          first_part = Block::make(first_part, temp_first);
        } else {
          second_part = temp_first;
        }
      }
      temp = left_blo->rest;
    }

    if (second_part.defined()) {
      second_part = Block::make(second_part, temp);
    } else {
      second_part = temp;
    }
    return {first_part, second_part};
  }

 private:
  Stmt first_part;
  Stmt second_part;
  int vectorize_size_{4};
};

class VectorizedForAmend : public IRMutator {
 public:
  explicit VectorizedForAmend(std::shared_ptr<ReductionData> &cur_reduce_data) : cur_reduce_data_(cur_reduce_data) {}

  Stmt Mutate_(const For *op, const Stmt &s) {
    if (cur_reduce_data_->isolate_vector_for.count(op)) {
      auto body = IRMutator::Mutate(op->body);
      return For::make(op->loop_var, op->min, op->extent, air::ir::ForType::Serial, op->device_api, body);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::shared_ptr<ReductionData> cur_reduce_data_;
};

class MutateReduceBody : public IRMutator {
 public:
  explicit MutateReduceBody(std::shared_ptr<ReductionData> &cur_reduce_data) : cur_reduce_data_(cur_reduce_data) {}

  Stmt Mutate_(const For *op, const Stmt &s) {
    if (cur_reduce_data_->isolate_vector_for.count(op)) {
      auto old_op = op;
      Stmt stmt = IRMutator::Mutate_(op, s);
      auto new_op = stmt.as<For>();
      cur_reduce_data_->isolate_vector_for.erase(old_op);
      cur_reduce_data_->isolate_vector_for.insert(new_op);
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    if (op == cur_reduce_data_->reduce_provide || cur_reduce_data_->parallel_internal_reduce_provides.count(op)) {
      auto value = op->value;
      auto temp_call = cur_reduce_data_->reduce_temp;
      auto input_call = GetTheInputExpr(op);
      if (value.as<Min>()) {
        value = Min::make(temp_call, input_call);
      } else if (value.as<Max>()) {
        value = Max::make(temp_call, input_call);
      } else if (value.as<And>()) {
        value = And::make(temp_call, input_call);
      } else if (value.as<Or>()) {
        value = Or::make(temp_call, input_call);
      } else if (value.as<Add>()) {
        value = Add::make(temp_call, input_call);
      } else if (value.as<Mul>()) {
        value = Mul::make(temp_call, input_call);
      }

      CHECK(cur_reduce_data_->reduce_temp.as<Call>());
      return Provide::make(cur_reduce_data_->temp_tensor->op, cur_reduce_data_->temp_tensor->value_index, value,
                           cur_reduce_data_->reduce_temp.as<Call>()->args);
    } else if (cur_reduce_data_->isolate_reduce_provide.count(op) &&
               (cur_reduce_data_->outter_reduction_data != nullptr ||
                cur_reduce_data_->pre_reduction_data != nullptr)) {
      auto value = op->value;
      auto temp_call = MakeUniqueTempCallForIsolate(cur_reduce_data_->reduce_temp);
      auto input_call = GetTheInputExpr(op);
      if (value.as<Min>()) {
        value = Min::make(temp_call, input_call);
      } else if (value.as<Max>()) {
        value = Max::make(temp_call, input_call);
      } else if (value.as<And>()) {
        value = And::make(temp_call, input_call);
      } else if (value.as<Or>()) {
        value = Or::make(temp_call, input_call);
      } else if (value.as<Add>()) {
        value = Add::make(temp_call, input_call);
      } else if (value.as<Mul>()) {
        value = Mul::make(temp_call, input_call);
      }
      CHECK(cur_reduce_data_->reduce_temp.as<Call>());

      Array<Expr> args;
      args.push_back(Expr(0));
      return Provide::make(cur_reduce_data_->temp_tensor->op, cur_reduce_data_->temp_tensor->value_index, value, args);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr MakeUniqueTempCallForIsolate(const Expr &temp) {
    auto call = temp.as<Call>();
    Array<Expr> args;
    args.push_back(Expr(0));
    return Call::make(call->type, call->func->func_name(), args, call->call_type, call->func, call->value_index);
  }

  Expr GetTheInputExpr(const Provide *op) {
    auto value = op->value;
    auto dst_func = op->func;
    Expr a_val;
    Expr b_val;
    if (auto min = value.as<Min>()) {
      a_val = min->a;
      b_val = min->b;
    } else if (auto max = value.as<Max>()) {
      a_val = max->a;
      b_val = max->b;
    } else if (auto and_op = value.as<And>()) {
      a_val = and_op->a;
      b_val = and_op->b;
    } else if (auto or_op = value.as<Or>()) {
      a_val = or_op->a;
      b_val = or_op->b;
    } else if (auto add_op = value.as<Add>()) {
      a_val = add_op->a;
      b_val = add_op->b;
    } else if (auto mul_op = value.as<Mul>()) {
      a_val = mul_op->a;
      b_val = mul_op->b;
    } else {
      CHECK(false) << "reduce type is invalid";
    }

    if (a_val.as<Call>() && a_val.as<Call>()->func == dst_func) {
      return b_val;
    } else {
      return a_val;
    }
  }

 private:
  std::shared_ptr<ReductionData> cur_reduce_data_;
};

class ReduceVectorizeEnable : public IRMutator {
 public:
  explicit ReduceVectorizeEnable(std::map<int, std::shared_ptr<ReductionData>> &reduce_data)
      : reduce_datas_(reduce_data) {}

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    if (op->attr_key == REDUCE_AREA_FLAG) {
      cur_reduce_data_ = reduce_datas_[reduce_index_++];
      auto body = IRMutator::Mutate(op->body);
      CHECK(cur_reduce_data_) << "cur_reduce_data_ is nullptr";
      if (body.as<For>()) {
        cur_reduce_data_->body = body;
      } else if (auto blo = body.as<Block>()) {
        if (cur_reduce_data_->next_change_area_data != nullptr) {
          std::vector<Stmt> divide_block = DivideBlock(cur_reduce_data_->vectorize_size).Run(body);
          cur_reduce_data_->body = divide_block[0];
          cur_reduce_data_->next_change_area_data->body = divide_block[1];
        } else {
          auto first = blo->first;
          auto rest = blo->rest;
          if (first.as<For>() && first.as<For>()->for_type == air::ir::ForType::Parallel) {
            cur_reduce_data_->body = first;
            cur_reduce_data_->isolate = rest;
          } else {
            cur_reduce_data_->body = body;
          }
        }
      } else {
        return body;
      }

      CHECK(cur_reduce_data_->body.defined()) << "cur_reduce_data_ body is not defined";
      Stmt stmt = MakeReduceStmt();
      stmt = VectorizedForAmend(cur_reduce_data_).Mutate(stmt);
      auto current_level_entry = cur_reduce_data_;

      while (cur_reduce_data_->next_change_area_data != nullptr) {
        cur_reduce_data_ = cur_reduce_data_->next_change_area_data;
        Stmt temp = MakeReduceStmt();
        temp = VectorizedForAmend(cur_reduce_data_).Mutate(temp);
        stmt = Block::make(stmt, temp);
      }

      if (cur_reduce_data_->isolate.defined()) {
        stmt = Block::make(stmt, cur_reduce_data_->isolate);
      }

      if (current_level_entry->outter_reduction_data) {
        cur_reduce_data_ = current_level_entry->outter_reduction_data;
      } else if (current_level_entry->pre_reduction_data) {
        cur_reduce_data_ = current_level_entry->pre_reduction_data;
      }
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  void SetTensorBuffer(std::shared_ptr<ReductionData> &reduce_data, const std::string &tensor_name) {
    Array<Expr> shapes;
    shapes.push_back(reduce_data->vector_parallel_for->extent);
    Type type = reduce_data->reduce_data_type_info;

    Tensor tensor = placeholder(shapes, type, tensor_name);
    const Buffer buffer = decl_buffer(shapes, type, tensor_name);
    reduce_data->temp_tensor = tensor;
    reduce_data->temp_buffer = buffer;
  }

  void SetTheDstAndInputExpr(const Expr &a, const Expr &b, const FunctionRef &dst_func) {
    if (a.as<Call>() && a.as<Call>()->func == dst_func) {
      cur_reduce_data_->reduce_dst = a;
      cur_reduce_data_->reduce_input = b;
    } else {
      cur_reduce_data_->reduce_dst = b;
      cur_reduce_data_->reduce_input = a;
    }
  }

  Stmt MakeReduceStmt() {
    std::vector<Stmt> stmts;
    auto vector_parallel_for = cur_reduce_data_->vector_parallel_for;
    if (vector_parallel_for == nullptr) {
      return cur_reduce_data_->body;
    }
    auto reduce_provide = cur_reduce_data_->reduce_provide;
    CHECK(reduce_provide);

    // step 1: allocate the temp tensor
    auto name = reduce_provide->func->func_name();
    name += "_temp";
    name += std::to_string(tensor_name_count_++);
    SetTensorBuffer(cur_reduce_data_, name);
    Array<Expr> args;
    args.push_back(vector_parallel_for->loop_var);
    cur_reduce_data_->reduce_temp = MakeCallFromTempTensor(args);

    // step 2: final reduce area
    auto value = reduce_provide->value;
    auto dst_func = reduce_provide->func;
    if (auto min = value.as<Min>()) {
      SetTheDstAndInputExpr(min->a, min->b, dst_func);
      value = Min::make(cur_reduce_data_->reduce_dst, cur_reduce_data_->reduce_temp);
    } else if (auto max = value.as<Max>()) {
      SetTheDstAndInputExpr(max->a, max->b, dst_func);
      value = Max::make(cur_reduce_data_->reduce_dst, cur_reduce_data_->reduce_temp);
    } else if (auto and_op = value.as<And>()) {
      SetTheDstAndInputExpr(and_op->a, and_op->b, dst_func);
      value = And::make(cur_reduce_data_->reduce_dst, cur_reduce_data_->reduce_temp);
    } else if (auto or_op = value.as<Or>()) {
      SetTheDstAndInputExpr(or_op->a, or_op->b, dst_func);
      value = Or::make(cur_reduce_data_->reduce_dst, cur_reduce_data_->reduce_temp);
    } else if (auto add = value.as<Add>()) {
      SetTheDstAndInputExpr(add->a, add->b, dst_func);
      value = Add::make(cur_reduce_data_->reduce_dst, cur_reduce_data_->reduce_temp);
    } else if (auto mul = value.as<Mul>()) {
      SetTheDstAndInputExpr(mul->a, mul->b, dst_func);
      value = Mul::make(cur_reduce_data_->reduce_dst, cur_reduce_data_->reduce_temp);
    } else {
      CHECK(false) << "reduce type is invalid";
    }

    auto loop_var = Variable::make(vector_parallel_for->loop_var->type,
                                   vector_parallel_for->loop_var->name_hint + std::to_string(var_name_count_++));
    auto reduce_provide_new =
      Provide::make(reduce_provide->func, reduce_provide->value_index, value, reduce_provide->args);
    Map<Var, Expr> finial_replace_var;
    finial_replace_var.Set(vector_parallel_for->loop_var, loop_var);
    reduce_provide_new = air::ir::Substitute(reduce_provide_new, finial_replace_var);
    if (cur_reduce_data_->outter_reduction_data.get() != nullptr) {
      auto op = reduce_provide_new.as<Provide>();
      if (cur_reduce_data_->outter_reduction_data->reduce_provide == nullptr) {
        cur_reduce_data_->outter_reduction_data->reduce_provide = op;
        cur_reduce_data_->outter_reduction_data->reduce_data_type_info = cur_reduce_data_->reduce_data_type_info;
        cur_reduce_data_->outter_reduction_data->init_value = cur_reduce_data_->init_value;
        cur_reduce_data_->outter_reduction_data->vectorize_size = cur_reduce_data_->vectorize_size;
      }
      cur_reduce_data_->outter_reduction_data->parallel_internal_reduce_provides.insert(op);
    }
    Stmt final_reduce = For::make(loop_var, vector_parallel_for->min, vector_parallel_for->extent,
                                  air::ir::ForType::Serial, vector_parallel_for->device_api, reduce_provide_new);

    // step 3: vectorize or parallel area
    auto reduce_body = MutateReduceBody(cur_reduce_data_).Mutate(cur_reduce_data_->body);

    // step 4: init area
    loop_var = Variable::make(vector_parallel_for->loop_var->type,
                              vector_parallel_for->loop_var->name_hint + std::to_string(var_name_count_++));
    Map<Var, Expr> init_replace_var;
    init_replace_var.Set(vector_parallel_for->loop_var, loop_var);
    auto init_provide = Provide::make(cur_reduce_data_->temp_tensor->op, cur_reduce_data_->temp_tensor->value_index,
                                      cur_reduce_data_->init_value, cur_reduce_data_->reduce_temp.as<Call>()->args);
    init_provide = air::ir::Substitute(init_provide, init_replace_var);
    Stmt init_stmt = For::make(loop_var, vector_parallel_for->min, vector_parallel_for->extent,
                               air::ir::ForType::Vectorized, vector_parallel_for->device_api, init_provide);

    // 4 make block and insert realize
    Stmt stmt = Block::make({init_stmt, reduce_body, final_reduce});
    stmt = InsertRealizeForTempTensor(stmt);

    return stmt;
  }

  Expr MakeCallFromTempTensor(const Array<Expr> &args) {
    std::string name = cur_reduce_data_->temp_tensor->op->name;
    Type type = cur_reduce_data_->temp_buffer->dtype;
    Expr dst = Call::make(type, name, args, Call::Halide, cur_reduce_data_->temp_tensor->op, 0);
    return dst;
  }

  Stmt InsertRealizeForTempTensor(const Stmt &s) {
    Tensor t = cur_reduce_data_->temp_tensor;
    Region bounds;

    for (auto j : t->shape) {
      bounds.push_back(Range::make_by_min_extent(Expr(0), j));
    }
    auto buf = cur_reduce_data_->temp_buffer;
    Stmt stmt = Realize::make(t->op, t->value_index, t->dtype, bounds, const_true(1), s);
    stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, Expr("local"), stmt);

    return stmt;
  }

 public:
  int reduce_index_{1};
  std::map<int, std::shared_ptr<ReductionData>> reduce_datas_;
  std::shared_ptr<ReductionData> cur_reduce_data_;
  int var_name_count_{0};
  int tensor_name_count_{0};
};

Stmt ReductionFactor(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer) {
  IdentifyReduceChance identify_reduce(extern_buffer);
  identify_reduce.Visit(stmt);

  return ReduceVectorizeEnable(identify_reduce.reduce_datas_).Mutate(stmt);
}

}  // namespace ir
}  // namespace akg
