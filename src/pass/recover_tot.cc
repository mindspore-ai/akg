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
#include <tvm/buffer.h>
#include <tvm/target_info.h>
#include <tvm/build_module.h>
#include <tvm/runtime/device_api.h>

#include <unordered_map>

#include "common/common_util.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
namespace {
/*
fakeout = tot_op_id(par, update, index, index)
------------------->
par = tot_op_id(update, index, index)
*/
class RemoveFakeout : public IRMutator {
 public:
  explicit RemoveFakeout(Map<std::string, Map<std::string, NodeRef>> &op_attrs) : op_attrs_(op_attrs) {}
  ~RemoveFakeout() override = default;

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (op->value.as<Call>() && op->value.as<Call>()->name.rfind("tot_op_", 0) == 0) {
      auto tot_name = op->value.as<Call>()->name;
      auto attrs = op_attrs_[tot_name];
      if (attrs.count("is_fakeout") && 1 == ir::GetInt32Const(Downcast<Expr>(attrs["is_fakeout"]))) {
        auto realout_pos = ir::GetInt32Const(Downcast<Expr>(attrs["realout_pos"]));
        CHECK(0 <= realout_pos && (size_t)realout_pos < op->value.as<Call>()->args.size());
        // reset attrs
        auto tensor_of_tensor_pos = ir::GetInt32Const(Downcast<Expr>(attrs["tensor_of_tensor_pos"]));
        if (tensor_of_tensor_pos >= realout_pos) {
          attrs.Set("tensor_of_tensor_pos", Expr(tensor_of_tensor_pos - 1));
        }
        auto first_index_pos = ir::GetInt32Const(Downcast<Expr>(attrs["first_index_pos"]));
        attrs.Set("first_index_pos", Expr(first_index_pos - 1));
        attrs.Set("realout_pos", Expr(-1));
        op_attrs_.Set(tot_name, attrs);
        // remove fakeout
        auto old_args = op->value.as<Call>()->args;
        Array<Expr> new_args;
        for (size_t i = 0; i < old_args.size(); ++i) {
          if ((int)i != realout_pos) {
            new_args.push_back(old_args[i]);
          }
        }
        auto call = op->value.as<Call>();
        auto new_value = Call::make(call->type, call->name, new_args, call->call_type, call->func, call->value_index);
        auto provide =
          Provide::make(old_args[realout_pos].as<Call>()->func, 0, new_value, old_args[realout_pos].as<Call>()->args);
        return provide;
      }
    }
    return s;
  }

 private:
  Map<std::string, Map<std::string, NodeRef>> &op_attrs_;
};

/*
case 1, outbound_return_zero = 1:
  output(idx, idy) = tot_op_0(input_0(block_offset + idx, idy), input_1_shared(idx))
  ------------------>
  if 0 <= input_1_shared(idx) < input_0.shape[0] {
    output(idx, idy) = input_0(input_1_shared(idx), idy)
  } else {
    output(idx,idy) = 0
  }

case 2, is_atomic_add = 1:
  par(batch, blockoffset + idx.x, idx.y, idx.z) = tot_op_0(update(blockoffset + idx.x, idx.y, idx.z),
  index_shared(idx.x, 0), index_shared(idx, 1))
  ------------------>
  if (0 <= index_shared(idx.x, 0) && index_shared(idx.x, 0) < par.shape[batch.size + 0].size()
      && 0 <= index_shared(idx.x, 1) && index_shared(idx.x, 1) < par.shape[batch.size + 1].size())
  {
    // attr ["INFO"] tsa_atomic_add = 1
    par(batch, index_shared(idx.x, 0), index_shared(idx.x, 1), idx.z) = update(blockoffset + idx.x, idx.y, idx.z)
  }
*/
class RecoverMutator : public IRMutator {
 public:
  explicit RecoverMutator(const Map<std::string, Map<std::string, NodeRef>> &op_attrs) : op_attrs_(op_attrs) {}
  ~RecoverMutator() override = default;

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (op->value.as<Call>() && op->value.as<Call>()->name.rfind("tot_op_", 0) == 0) {
      auto tot_name = op->value.as<Call>()->name;
      CHECK(op_attrs_.count(tot_name));
      auto attrs = op_attrs_[tot_name];
      auto old_args = op->value.as<Call>()->args;
      auto first_index_pos = ir::GetInt32Const(Downcast<Expr>(attrs["first_index_pos"]));
      Array<Expr> indices;
      for (size_t i = (size_t)first_index_pos; i < old_args.size(); ++i) {
        indices.push_back(old_args[i]);
      }
      auto first_dst_pos = (size_t)ir::GetInt32Const(Downcast<Expr>(attrs["first_dst_pos"]));
      std::vector<size_t> dst_pos;
      for (size_t i = 0; i < indices.size(); ++i) {
        dst_pos.push_back(first_dst_pos + i);
      }
      auto tensor_of_tensor_pos = ir::GetInt32Const(Downcast<Expr>(attrs["tensor_of_tensor_pos"]));
      Tensor tot;
      if (-1 == tensor_of_tensor_pos) {
        tot = Downcast<Operation>(op->func).output(0);
      } else {
        tot = Downcast<Operation>(old_args[tensor_of_tensor_pos].as<Call>()->func).output(0);
      }
      // condition
      Expr condition = And::make(indices[0] >= 0, indices[0] < tot->shape[dst_pos[0]]);
      for (size_t i = 1; i < indices.size(); ++i) {
        condition = And::make(condition, And::make(indices[i] >= 0, indices[i] < tot->shape[dst_pos[i]]));
      }
      // if_provide else_provide
      Array<Expr> new_args;
      Stmt if_provide;
      Stmt else_provide;
      if (tensor_of_tensor_pos == -1) {
        new_args = op->args;
        for (size_t i = 0; i < dst_pos.size(); ++i) {
          new_args.Set(dst_pos[i], indices[i]);
        }
        Array<Expr> new_value_args;
        for (int i = 0; i < first_index_pos; ++i) {
          new_value_args.push_back(old_args[i]);
        }
        if_provide = Provide::make(op->func, 0, op->value.as<Call>()->args[0], new_args);
        else_provide = Provide::make(op->func, 0, make_const(tot->dtype, 0), new_args);
      } else {
        auto old_tot_call = old_args[tensor_of_tensor_pos].as<Call>();
        auto new_tot_args = old_tot_call->args;
        for (size_t i = 0; i < dst_pos.size(); ++i) {
          new_tot_args.Set(dst_pos[i], indices[i]);
        }
        auto new_tot_call = Call::make(old_tot_call->type, old_tot_call->name, new_tot_args, old_tot_call->call_type,
                                       old_tot_call->func, old_tot_call->value_index);
        CHECK(tensor_of_tensor_pos == 0 && first_index_pos == 1);
        if_provide = Provide::make(op->func, 0, new_tot_call, op->args);
        else_provide = Provide::make(op->func, 0, make_const(tot->dtype, 0), op->args);
      }
      // mark atomic_add
      auto is_atomic_add = ir::GetInt32Const(Downcast<Expr>(attrs["is_atomic_add"]));
      auto attr = if_provide;
      if (is_atomic_add == 1) {
        attr = AttrStmt::make(Expr("INFO"), "tsa_atomic_add", 1, if_provide);
      }
      // return ifthenelse
      auto outbound_return_zero = ir::GetInt32Const(Downcast<Expr>(attrs["outbound_return_zero"]));
      if (outbound_return_zero == 1) {
        return IfThenElse::make(condition, attr, else_provide);
      }
      return IfThenElse::make(condition, attr);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  const Map<std::string, Map<std::string, NodeRef>> &op_attrs_;
};

/*
  // attr ["INFO"] tsa_atomic_add = 1
  par(index(threadidx.x, 0), threadidx.y) = update(threadidx.x, threadidx.y)
  ---------------------------------->
  // attr ["INFO"] reduceLibType = "origin"
  akg_reduce::AkgAtomicReturn(dtype, "akg_reduce::SumOp", update(threadidx.x, threadidx.y), &par(index(threadidx.x, 0),
  threadidx.y))
*/
class AtomicReturnStmtEmit : public IRMutator {
 public:
  AtomicReturnStmtEmit(const std::string &lib, const Map<Tensor, Buffer> &binds) : reduce_lib_(lib), binds_(binds) {}
  explicit AtomicReturnStmtEmit(const Map<Tensor, Buffer> &binds)
      : reduce_lib_(REDUCE_LIB_TYPE_ORIGIN), binds_(binds) {}

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    if (op->attr_key == "tsa_atomic_add") {
      in_atomic_area_ = true;
      atomic_data_.reduce_op.clear();
      atomic_data_.reduce_op = AKG_REDUCE_LIB_SPACE;
      atomic_data_.reduce_op += "::SumOp";
      auto new_body = IRMutator::Mutate(op->body);
      return AttrStmt::make(Expr("INFO"), "reduceLibType", Expr(reduce_lib_), new_body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    if (in_atomic_area_) {
      in_atomic_area_ = false;
      Stmt stmt = IRMutator::Mutate_(op, s);
      atomic_data_.gm_write_stmt = stmt;
      auto op = stmt.as<Provide>();
      CHECK(op);
      atomic_data_.atomic_rhs = op->value;
      atomic_data_.output_tensor_data_type_info = GetDtypeOf(op->func->func_name());

      ConstructAtomicReturnFuncName(reduce_lib_, atomic_data_.reduce_op, atomic_data_.akg_atomic_api,
                                    atomic_data_.akg_atomic_template_arg);
      return MakeAtomicStmt(atomic_data_);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  Type GetDtypeOf(const std::string &tensor_name) const {
    for (auto i : binds_) {
      if (i.first->op->name == tensor_name) {
        return i.second->dtype;
      }
    }
    CHECK(false) << " no such tensor in binds: " << tensor_name;
    return Int(32);
  }

  AtomicReturnData atomic_data_;
  std::string reduce_lib_;
  Map<Tensor, Buffer> binds_;
  bool in_atomic_area_{false};
};
}  // namespace

Stmt RecoverTot(const Stmt &stmt, Map<std::string, Map<std::string, NodeRef>> &tot_attr,
                const Map<Tensor, Buffer> &binds) {
  Map<std::string, Map<std::string, NodeRef>> op_attrs(tot_attr);
  auto remove = RemoveFakeout(op_attrs);
  auto s = remove.Mutate(stmt);
  auto recover = RecoverMutator(op_attrs);
  auto s1 = recover.Mutate(s);
  auto emit_atomic_add = AtomicReturnStmtEmit(binds);
  auto s2 = emit_atomic_add.Mutate(s1);
  return s2;
}
}  // namespace ir
}  // namespace akg
