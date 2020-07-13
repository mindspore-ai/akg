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
#include "insn_info.h"

#include <tvm/runtime/packed_func.h>
#include <tvm/base.h>
#include <tvm/api_registry.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>

#include <map>
#include <algorithm>
#include <vector>
#include <numeric>
#include <unordered_set>

#include "common/array_api.h"
#include "common/util_cce.h"
#include "src/common/util.h"
#include "cce_params.h"
#include "pass/expr_alg_simplify.h"

namespace air {
/// Check if two StmtStoreInfo are equal
/// \param lhs    - Left arg
/// \param rhs    - Right arg
/// \return bool  - Whether they are equal or not
bool Equal(const akg::StmtStoreInfo &lhs, const akg::StmtStoreInfo &rhs) { return lhs == rhs; }
}  // namespace air

namespace akg {
TVM_REGISTER_NODE_TYPE(StmtStoreInfoNode);
TVM_REGISTER_NODE_TYPE(VectorArgInfoNode);
TVM_REGISTER_NODE_TYPE(ArgInfoNode);

using air::runtime::PackedFunc;
using air::runtime::TVMArgs;
using air::runtime::TVMRetValue;

IterVar CCE_AXIS_VAR = thread_axis(Range(), "cce");

/// GetCceAxis variable
/// \return CCE_AXIS_VAR
IterVar GetCceAxis() { return CCE_AXIS_VAR; }

/// Move flex_var back to elem_offset
void StmtStoreInfo::CleanFlexVar() {
  auto node = GetNode();
  CHECK(node != nullptr);
  for (auto v : node->flex_var_) {
    size_t loc = 0;
    if (GetIndexOfElement(node->var_, v, loc)) {
      auto stride = node->strides_[loc];
      node->var_ = RemoveItemAtIndex(node->var_, loc);
      node->strides_ = RemoveItemAtIndex(node->strides_, loc);
      node->shape_ = RemoveItemAtIndex(node->shape_, loc);
      node->elem_offset_ += v * stride;
    }
  }
  node->flex_var_ = Array<Var>();
}

/// Make a StmtStoreInfo copy
/// \return com_info - Copied StmtStoreInfo
StmtStoreInfo StmtStoreInfo::Copy() const {
  auto com_info = StmtStoreInfo(make_node<StmtStoreInfoNode>());
  auto node = com_info.GetNode();

  auto this_node = this->GetNode();
  CHECK(this_node != nullptr);

  node->strides_ = this_node->strides_;
  node->shape_ = this_node->shape_;
  node->var_ = this_node->var_;
  node->flex_var_ = this_node->flex_var_;
  node->scope_ = this_node->scope_;
  node->name_ = this_node->name_;
  node->index_ = this_node->index_;
  node->elem_offset_ = this_node->elem_offset_;
  node->insn_offset_ = this_node->insn_offset_;
  node->dtype_ = this_node->dtype_;
  node->data_alignment_ = this_node->data_alignment_;
  node->data_ = this_node->data_;
  node->buffer_ = this_node->buffer_;

  return com_info;
}

/// Make a StmtInfo copy
/// \return
StmtInfo StmtInfo::Copy() const {
  auto stmt_info = StmtInfo();
  stmt_info.ops_ = ops_;
  for (auto var : vars_) {
    auto new_var = Variable::make(var->type, var->name_hint);
    stmt_info.vars_.push_back(new_var);
  }

  for (size_t i = 0; i < vars_.size(); ++i) {
    for (size_t j = 0; j < stmt_info.ops_.size(); ++j) {
      auto new_op = substitute(vars_[i], stmt_info.vars_[i], stmt_info.ops_[j]);
      auto for_op = new_op.as<For>();
      CHECK(for_op != nullptr);
      new_op =
        For::make(stmt_info.vars_[j], for_op->min, for_op->extent, for_op->for_type, for_op->device_api, for_op->body);
      stmt_info.ops_.Set(j, new_op);
    }
  }

  return stmt_info;
}

/// Replace a / target * target
/// \param value
/// \param target
/// \return
int FloorTo(int value, int target) {
  CHECK_NE(target, 0);
  return value / target * target;
}

/// Replace (a + target - 1) / target * target
/// \param value
/// \param target
/// \return
int CeilTo(int value, int target) {
  CHECK_NE(target, 0);
  return (value + target - 1) / target * target;
}

/// Eliminate vars in expr
/// \param e    - Expr to be processed
/// \param vars - Var list to be eliminated
/// \return e   - Processed expr
Expr EliminateVarInExpr(Expr e, const Array<Var> &vars) {
  if (vars.empty()) {
    return e;
  }

  Map<Var, Expr> var_dict;
  for (auto var : vars) {
    var_dict.Set(var, Expr(0));
  }

  e = Substitute(e, var_dict);
  e = Simplify(e);

  return e;
}

/// Sort three arrays with one rule
/// \param vars    - Var list to be Sorted
/// \param shapes  - Shape list to be Sorted
/// \param strides - Stride list to be Sorted
/// \param reverse - Reverse order or not
void SortVarShapeAndStride(Array<Var> &vars, Array<Expr> &shapes, Array<Expr> &strides, bool reverse) {
  size_t size = std::min(vars.size(), std::min(shapes.size(), strides.size()));
  vars = GetRange(vars, 0, size);
  shapes = GetRange(shapes, 0, size);
  strides = GetRange(strides, 0, size);

  for (size_t i = 1; i < size; ++i) {
    for (size_t j = i; j > 0; --j) {
      CHECK(strides[j - 1].as<IntImm>());
      auto l_value = strides[j - 1].as<IntImm>()->value;
      CHECK(strides[j].as<IntImm>());
      auto r_value = strides[j].as<IntImm>()->value;
      bool cmp = reverse ? r_value > l_value : r_value < l_value;
      if (cmp) {
        auto stride = strides[j];
        strides.Set(j, strides[j - 1]);
        strides.Set(j - 1, stride);

        auto var = vars[j];
        vars.Set(j, vars[j - 1]);
        vars.Set(j - 1, var);

        auto shape = shapes[j];
        shapes.Set(j, shapes[j - 1]);
        shapes.Set(j - 1, shape);
      } else if (r_value == l_value) {
        auto l_var = vars[j - 1].get();
        auto r_var = vars[j].get();
        if (l_var < r_var) {
          auto stride = strides[j];
          strides.Set(j, strides[j - 1]);
          strides.Set(j - 1, stride);

          auto var = vars[j];
          vars.Set(j, vars[j - 1]);
          vars.Set(j - 1, var);

          auto shape = shapes[j];
          shapes.Set(j, shapes[j - 1]);
          shapes.Set(j - 1, shape);
        }
      }
    }
  }
}

/// Get buffer's scope from its name
/// \param name    - Name of buffer
/// \return string - Scope string
std::string GetBufScope(const std::string &name) {
  std::map<std::string, std::string> mem_dict = {{"UB", SCOPE_UBUF}, {"L1", SCOPE_CBUF}, {"L0A", SCOPE_CA},
                                                 {"L0B", SCOPE_CB},  {"L0C", SCOPE_CC},  {"REG", SCOPE_REG}};
  std::vector<std::string> split_list = air::common::Split(name, '.');
  if (split_list.size() == 1) {
    split_list = akg::common::Split(name, "_local_");
  }

  std::string key = split_list[split_list.size() - 1];
  for (auto &iter : mem_dict) {
    std::string::size_type pos = split_list[split_list.size() - 1].find(iter.first);
    if (pos != std::string::npos) {
      key = iter.first;
      break;
    }
  }
  if (split_list.size() == 1) {
    return DMA_COPY_GLOBAL;
  }

  return mem_dict[key];
}

/// Get insn offset by eliminate vars
/// \param com_info - Computation info
/// \param elim_var - Vars to be eliminated
/// \return expr   - Processed expr
Expr GetInsnOffset(const StmtStoreInfo &com_info, const Array<Var> &elim_var) {
  auto inner_index = Simplify(Sub::make(com_info->index_, com_info->elem_offset_));
  auto insn_offset = EliminateVarInExpr(inner_index, elim_var);
  return insn_offset;
}

/// Get for_info from stmt
/// \param s - Input stmt
/// \return for_info - For loop Information
StmtInfo GetForInfo(const Stmt &s) {
  StmtInfo if_info;
  StmtInfo for_info;
  GetIfForInfo(s, if_info, for_info);
  return for_info;
}

/// Get if stmt and for stmt from a stmt
/// \param s
/// \param if_info
/// \param for_info
void GetIfForInfo(const Stmt &s, StmtInfo &if_info, StmtInfo &for_info) {
  if (s->IsInstance<AttrStmt>()) {
    const auto attrstmt_ptr = s.as<AttrStmt>();
    GetIfForInfo(attrstmt_ptr->body, if_info, for_info);
  }
  if (s->IsInstance<For>()) {
    const auto for_ptr = s.as<For>();
    for_info.vars_.push_back(for_ptr->loop_var);
    for_info.ops_.push_back(s);
    GetIfForInfo(for_ptr->body, if_info, for_info);
  }
  if (s->IsInstance<IfThenElse>()) {
    if_info.ops_.push_back(s);
    const auto ifthenelse_ptr = s.as<IfThenElse>();
    const Expr &condition_ = ifthenelse_ptr->condition;

    auto temp_vars = GetVarsInExpr(condition_);
    for (auto iter : temp_vars) {
      if_info.vars_.push_back(iter);
    }

    if (ifthenelse_ptr->else_case.defined()) {
      LOG(FATAL) << "Unsupport \'else\' condition yet.";
    }

    GetIfForInfo(ifthenelse_ptr->then_case, if_info, for_info);
  }
}

/// Get the children of expr of binary operation
/// \param e - Expr to be processed
/// \return Array<Expr> e.a and e.b - If e is not binary op, then return empty Array.
Array<Expr> GetBinaryOpExprChildren(const Expr &e) {
  Array<Expr> children;
  if (auto add = e.as<Add>()) {
    children.push_back(add->a);
    children.push_back(add->b);
    return children;
  } else if (auto sub = e.as<Sub>()) {
    children.push_back(sub->a);
    children.push_back(sub->b);
    return children;
  } else if (auto mul = e.as<Mul>()) {
    children.push_back(mul->a);
    children.push_back(mul->b);
    return children;
  } else if (auto div = e.as<Div>()) {
    children.push_back(div->a);
    children.push_back(div->b);
    return children;
  } else if (auto f_div = e.as<FloorDiv>()) {
    children.push_back(f_div->a);
    children.push_back(f_div->b);
    return children;
  } else if (auto mod = e.as<Mod>()) {
    children.push_back(mod->a);
    children.push_back(mod->b);
    return children;
  } else if (auto f_mod = e.as<FloorMod>()) {
    children.push_back(f_mod->a);
    children.push_back(f_mod->b);
    return children;
  } else if (auto min = e.as<Min>()) {
    children.push_back(min->a);
    children.push_back(min->b);
    return children;
  } else if (auto max = e.as<Max>()) {
    children.push_back(max->a);
    children.push_back(max->b);
    return children;
  } else if (auto eq = e.as<EQ>()) {
    children.push_back(eq->a);
    children.push_back(eq->b);
    return children;
  } else if (auto ne = e.as<NE>()) {
    children.push_back(ne->a);
    children.push_back(ne->b);
    return children;
  } else if (auto lt = e.as<LT>()) {
    children.push_back(lt->a);
    children.push_back(lt->b);
    return children;
  } else if (auto le = e.as<LE>()) {
    children.push_back(le->a);
    children.push_back(le->b);
    return children;
  } else if (auto gt = e.as<GT>()) {
    children.push_back(gt->a);
    children.push_back(gt->b);
    return children;
  } else if (auto ge = e.as<GE>()) {
    children.push_back(ge->a);
    children.push_back(ge->b);
    return children;
  } else if (auto and_op = e.as<And>()) {
    children.push_back(and_op->a);
    children.push_back(and_op->b);
    return children;
  } else if (auto or_op = e.as<Or>()) {
    children.push_back(or_op->a);
    children.push_back(or_op->b);
    return children;
  } else {
    return children;
  }
}

/// Get all store nodes in stmt
/// \param s - Input stmt
/// \return Array<Stmt> - All stores in stmt
Array<Stmt> GetStores(const Stmt &s) {
  Array<NodeRef> stores;
  Array<NodeRef> loads;
  GetStoreAndLoads(s, stores, loads);

  Array<Stmt> result;
  std::transform(stores.begin(), stores.end(), std::back_inserter(result.CopyOnWrite()->data),
                 [](const NodeRef &v) { return (Downcast<Stmt>(v)); });

  return result;
}

/// Get stores and loads from a stmt
/// \param s      - Stmt to be processed
/// \param stores - Store list to be returned
/// \param loads  - Load list to be returned
void GetStoreAndLoads(const Stmt &s, Array<NodeRef> &stores, Array<NodeRef> &loads) {
  // Get stores
  Array<Expr> enable;
  enable.push_back(Expr("Store"));

  PackedFunc post_order = PackedFunc([&stores](const TVMArgs args, TVMRetValue *ret) {
    const auto &sptr = args[0].operator ObjectRef();
    if (sptr->IsInstance<Store>()) {
      stores.push_back(args[0]);
    }
    *ret = TVMRetValue();
  });

  static_cast<void>(air::ir::IRTransform(s, PackedFunc{nullptr}, post_order, enable));

  // Get loads in store
  PackedFunc pre_order;
  pre_order = PackedFunc([&loads, &pre_order](const TVMArgs args, TVMRetValue *ret) {
    Expr val = args[0];
    if (val.as<Load>()) {
      loads.push_back(val);
    } else if (val->IsInstance<Call>()) {
      auto tmp_args = val.as<Call>()->args;
      for (auto tmp_arg : tmp_args) {
        static_cast<void>(pre_order(tmp_arg));
      }
    } else if (val->IsInstance<Select>()) {
      auto tmp_args = val.as<Select>();
      static_cast<void>(pre_order(tmp_args->condition));
      static_cast<void>(pre_order(tmp_args->true_value));
      static_cast<void>(pre_order(tmp_args->false_value));
    } else if (val->IsInstance<Cast>()) {
      auto tmp_val = val.as<Cast>()->value;
      static_cast<void>(pre_order(tmp_val));
    } else {
      Array<Expr> tmp = GetBinaryOpExprChildren(val);
      if (!tmp.empty()) {
        static_cast<void>(pre_order(tmp[0]));
        static_cast<void>(pre_order(tmp[1]));
      }
    }
    *ret = TVMRetValue();
  });

  for (auto store : stores) {
    if (store->IsInstance<Store>()) {
      auto val = store.as<Store>()->value;
      static_cast<void>(pre_order(val));
    }
  }
}

/// Get buffer data alignment, depending on scope and data type also called 'block_size' in buffer
/// \param dst_info - Computation info of dst
/// \param src_info - Computation info of src
/// \return int    - Scope block size
int GetScopeBlockSize(const StmtStoreInfo &dst_info, const StmtStoreInfo &src_info) {
  auto dtype = dst_info->dtype_;
  auto data_width = dtype.bits();
  auto dst_scope = dst_info->scope_;
  auto src_scope = src_info->scope_;

  auto data_bytes = GLB_ELEM_BYTES;
  // Load 2D
  if (dst_scope == SCOPE_CB || dst_scope == SCOPE_CA || src_scope == SCOPE_CB || src_scope == SCOPE_CA) {
    data_bytes = BLOCK_IN * BLOCK_OUT * data_width / 8;
    if (data_width == 8) {
      data_bytes *= 2;
    }
  } else if (dst_scope == SCOPE_CC || src_scope == SCOPE_CC) {
    data_bytes = BLOCK_IN * BLOCK_OUT * data_width / 8;
  }

  CHECK_NE(data_width, 0);
  return data_bytes * 8 / data_width;
}

/// Get ub block size
/// \param type - Type of store
/// \return int - Block size
int GetUbBlkSize(const Type &type) {
  CHECK_NE(type.bits(), 0);
  int result = GLB_ELEM_BYTES * 8 / type.bits();
  CHECK_NE(result, 0) << "Get zero UB Block Size";
  return result;
}

/// Get all Var in expr
/// \param expr - Expr to be processed
/// \return Array<VarExpr> - List of var in expr
Array<VarExpr> GetVarsInExpr(const Expr &expr, bool exclude_upper_case_vars) {
  class VariableMutator : public IRMutator {
   public:
    explicit VariableMutator(Array<Var> &ivar_set, bool exclude_upper = false)
        : ivar_set_(ivar_set), exclude_upper_(exclude_upper) {}
    ~VariableMutator() override = default;

    Expr Mutate_(const Variable *op, const Expr &e) final {
      bool find_var = true;
      if (exclude_upper_) {
        for (auto c : op->name_hint) {
          if (c >= 'A' && c <= 'Z') {
            find_var = false;
            break;
          }
        }
      }
      if (find_var) {
        bool find = false;
        for (auto iter = ivar_set_.begin(); iter != ivar_set_.end(); ++iter) {
          if ((*iter).get() == op) {
            find = true;
            break;
          }
        }
        if (!find) {
          ivar_set_.push_back(Downcast<Var>(e));
        }
      }
      return e;
    }
    Array<Var> &ivar_set_;
    bool exclude_upper_{false};
  };

  Array<Var> ivar_set;
  VariableMutator(ivar_set, exclude_upper_case_vars).Mutate(expr);
  return ivar_set;
}

/// Get all variables in the given expr
/// \param expr - Input expr
/// \return unordered_set of variables
std::unordered_set<const Variable *> GetVariablesInExpr(const Expr &expr) {
  std::unordered_set<const Variable *> vars;
  PostOrderVisit(expr, [&vars](const NodeRef &node) {
    if (const auto v = node.as<Variable>()) {
      vars.insert(v);
    }
  });
  return vars;
}

/// Clean non-linear loop vars in index and offset
/// \param dst_info_list - Dst info list to be cleaned
/// \param src_info_list - Src info list to be cleaned
/// \param if_info      - If info of above infos
void CleanNonLinearVar(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list, const StmtInfo &if_info) {
  Map<Var, Expr> offset_vars;
  for (auto dst_info : dst_info_list) {
    auto vars = GetVarsInExpr(dst_info->elem_offset_);
    for (auto e : vars) {
      offset_vars.Set(e, Expr(0));
    }
  }
  for (auto src_info : src_info_list) {
    auto vars = GetVarsInExpr(src_info->elem_offset_);
    for (auto e : vars) {
      offset_vars.Set(e, Expr(0));
    }
  }

  for (auto dst_info : dst_info_list) {
    auto dst_var = dst_info->var_;
    auto dst_shape = dst_info->shape_;
    auto dst_strides = dst_info->strides_;
    auto dst_index = dst_info->index_;
    if (src_info_list.empty()) {
      // clean dst_vars
      for (uint64_t i = 0; i < dst_info->var_.size(); ++i) {
        auto d_var = dst_info->var_[i];
        auto is_flex_var = IsFlexVarInIf(d_var, if_info.ops_);
        if (IsInArray(if_info.vars_, VarExpr(d_var))) {
          if (i == dst_info->var_.size() - 1 && is_flex_var) {
            dst_info.GetNode()->flex_var_.push_back(d_var);
          } else {
            size_t index = 0;
            if (GetIndexOfElement(dst_var, d_var, index)) {
              dst_var = RemoveItemAtIndex(dst_var, index);
              dst_shape = RemoveItemAtIndex(dst_shape, index);
              dst_strides = RemoveItemAtIndex(dst_strides, index);
            }
          }
        }
      }
      dst_info.GetNode()->var_ = dst_var;
    } else {
      for (auto src_info : src_info_list) {
        auto src_var = src_info->var_;
        auto src_shape = src_info->shape_;
        auto src_strides = src_info->strides_;
        auto src_index = src_info->index_;

        // clean dst_vars
        for (uint64_t i = 0; i < dst_info->var_.size(); ++i) {
          auto d_var = dst_info->var_[i];
          auto is_flex_var = IsFlexVarInIf(d_var, if_info.ops_);
          if (i == dst_info->var_.size() - 1 && IsInArray(if_info.vars_, VarExpr(d_var)) &&
              offset_vars.count(d_var) == 0 && is_flex_var) {
            dst_info.GetNode()->flex_var_.push_back(d_var);
          } else if (offset_vars.count(d_var) != 0 || IsInArray(if_info.vars_, VarExpr(d_var))) {
            size_t index = 0;
            if (GetIndexOfElement(dst_var, d_var, index)) {
              dst_var = RemoveItemAtIndex(dst_var, index);
              dst_shape = RemoveItemAtIndex(dst_shape, index);
              dst_strides = RemoveItemAtIndex(dst_strides, index);
            }
          }
        }
        dst_info.GetNode()->var_ = dst_var;

        // clean src_vars
        for (uint64_t i = 0; i < src_info->var_.size(); ++i) {
          auto s_var = src_info->var_[i];
          auto is_flex_var = IsFlexVarInIf(s_var, if_info.ops_);
          if (i == src_info->var_.size() - 1 && IsInArray(if_info.vars_, VarExpr(s_var)) &&
              offset_vars.count(s_var) == 0 && is_flex_var) {
            src_info.GetNode()->flex_var_.push_back(s_var);
          } else if (offset_vars.count(s_var) != 0 || IsInArray(if_info.vars_, VarExpr(s_var))) {
            size_t index = 0;
            if (GetIndexOfElement(src_var, s_var, index)) {
              src_var = RemoveItemAtIndex(src_var, index);
              src_shape = RemoveItemAtIndex(src_shape, index);
              src_strides = RemoveItemAtIndex(src_strides, index);
            }
          }
        }
        src_info.GetNode()->var_ = src_var;

        if (src_var.empty()) {
          src_info.GetNode()->shape_ = {Expr(1)};
          src_info.GetNode()->strides_ = {Expr(1)};
        } else {
          src_info.GetNode()->shape_ = src_shape;
          src_info.GetNode()->strides_ = src_strides;
        }
        src_info.GetNode()->elem_offset_ = EliminateVarInExpr(src_index, src_var);
      }
    }

    if (dst_var.empty()) {
      dst_info.GetNode()->shape_ = {Expr(1)};
      dst_info.GetNode()->strides_ = {Expr(1)};
    } else {
      dst_info.GetNode()->shape_ = dst_shape;
      dst_info.GetNode()->strides_ = dst_strides;
    }
    dst_info.GetNode()->elem_offset_ = EliminateVarInExpr(dst_index, dst_var);
  }
}

/// Substitute the Div call of 'DivRoundToZero' to safe div method
/// The DivSubstitutor is a method of IRMutator
namespace {
class DivSubstitutor : public IRMutator {
 public:
  Expr Mutate_(const Call *op, const Expr &e) final {
    auto expr = IRMutator::Mutate_(op, e);
    const auto call = expr.as<Call>();
    CHECK(call != nullptr);
    if (call->name == "DivRoundToZero") {
      if (call->args.size() != 2) {
        LOG(FATAL) << "Error: <Call> must have exactly two parameters here!";
      }
      if (GetIntConst(call->args[1]) <= 0) {
        LOG(WARNING) << "Warning: the divisor is not constant or it is less than 0!";
      }
      auto cond = LT::make(call->args[0], IntImm::make(Int(32), 0));
      auto t_value =
        Div::make(Sub::make(Add::make(call->args[0], call->args[1]), IntImm::make(Int(32), 1)), call->args[1]);
      auto f_value = Div::make(call->args[0], call->args[1]);
      return Simplify(Select::make(cond, t_value, f_value));
    }
    return expr;
  }
};
}  // namespace

/// Get computation info from a stmt
/// \param stores  - Array of Store/Load
/// \param for_info
/// \return StmtInfoList - Array of computation
StmtInfoList GetComputationInfo(const Array<NodeRef> &stores, const StmtInfo &for_info) {
  Array<Expr> for_extents;
  for (auto op : for_info.ops_) {
    CHECK(op.as<For>());
    for_extents.push_back(op.as<For>()->extent);
  }

  StmtInfoList com_info_list;

  auto CleanNonLinearVarsInIndex = [](Array<Var> &var_list, Array<Expr> &shape_list, const Expr &index) {
    // to solve the DetectLinearEquation problem
    // for index = (((((cc0/2) + cc6)*16) + cc7) - ((((cc6*2) + cc0)/4)*32))
    // if cur_vars = [cc6, cc7], the strides will returned as []
    // expect is cleaned cur_vars = [cc7], and returned strides = [1]
    auto tmp_var_list = var_list;
    for (auto var : tmp_var_list) {
      auto tmp_strides = air::arith::DetectLinearEquation(index, {var});
      if (tmp_strides.empty() || air::arith::Analyzer().CanProve(tmp_strides[0] <= 0)) {
        size_t idx = 0;
        if (GetIndexOfElement(var_list, var, idx)) {
          var_list = RemoveItemAtIndex(var_list, idx);
          shape_list = RemoveItemAtIndex(shape_list, idx);
        }
      }
    }
  };

  for (auto s : stores) {
    auto info = StmtStoreInfo(make_node<StmtStoreInfoNode>());

    const Store *store = nullptr;
    const Load *load = nullptr;
    Expr index;
    Expr predicate;
    if (s->IsInstance<Store>()) {
      store = s.as<Store>();
      index = store->index;
      predicate = store->predicate;
    } else if (s->IsInstance<Load>()) {
      load = s.as<Load>();
      index = load->index;
      predicate = load->predicate;
    }

    // Substitute the <Call>'DivRoundToZero' in index expression to avoid
    // problems during CCE generation.
    index = DivSubstitutor().Mutate(index);

    // get vars in store
    auto index_vars = GetVarsInExpr(index);

    // get current vars of index_vars and for_vars
    Array<VarExpr> cur_vars = IntersectionArray(index_vars, for_info.vars_);
    Array<Expr> cur_shapes;
    for (size_t i = 0; i < cur_vars.size(); ++i) {
      size_t idx = 0;
      bool suc = GetIndexOfElement(for_info.vars_, cur_vars[i], idx);
      CHECK(suc);
      cur_shapes.push_back(for_extents[idx]);
    }

    if (cur_vars.empty() || for_info.vars_.empty()) {
      // store is a variable, not an array
      info.GetNode()->shape_.push_back(Expr(1));
      info.GetNode()->strides_.push_back(Expr(1));
      info.GetNode()->elem_offset_ = Expr(0);
    } else {
      // store is an array
      // order cur_vars
      // get stride of store index
      Array<Expr> strides = air::arith::DetectLinearEquation(index, cur_vars);
      // if cur_vars have non-linear vars, then clean the vars
      if (strides.empty()) {
        CleanNonLinearVarsInIndex(cur_vars, cur_shapes, index);
        strides = air::arith::DetectLinearEquation(index, cur_vars);
      }
      // strides with complicate expr, then strides will be [], so the vars and shapes should be [] too
      if (strides.empty()) {
        cur_vars = {};
        cur_shapes = {};
      }

      bool is_all_const = false;
      int const_cnt = 0;
      for (auto &e : strides) {
        if (IsConstExpr(e)) {
          const_cnt++;
        }
      }
      for (auto &e : cur_shapes) {
        if (IsConstExpr(e)) {
          const_cnt++;
        }
      }
      if (strides.size() > 0 && const_cnt == static_cast<int>(strides.size() + cur_shapes.size())) {
        is_all_const = true;
      }

      // only keep positive strides and relate var and shape
      size_t var_len = cur_vars.size();
      size_t var_idx = 0;
      while (var_idx < var_len) {
        bool temp_condition = false;
        if (is_all_const) {
          temp_condition =
            !IsConstExpr(cur_shapes[var_idx]) || !IsConstExpr(strides[var_idx]) || GetIntConst(strides[var_idx]) < 0;
        } else {
          temp_condition = IsConstExpr(strides[var_idx]) && GetIntConst(strides[var_idx]) < 0;
        }
        if (temp_condition) {
          cur_vars = RemoveItemAtIndex(cur_vars, var_idx);
          cur_shapes = RemoveItemAtIndex(cur_shapes, var_idx);
          strides = RemoveItemAtIndex(strides, var_idx);
          var_len -= 1;
        } else {
          var_idx += 1;
        }
      }

      // order vars with strides order
      if (!is_all_const) {
        strides = RemoveItemAtIndex(strides, -1);
      } else {
        SortVarShapeAndStride(cur_vars, cur_shapes, strides, true);
      }

      // get ordered current vars
      if (var_len == 0) {
        info.GetNode()->shape_.push_back(Expr(1));
        info.GetNode()->strides_.push_back(Expr(1));
      } else {
        info.GetNode()->var_ = cur_vars;
        info.GetNode()->shape_ = cur_shapes;
        info.GetNode()->strides_ = strides;
      }

      info.GetNode()->elem_offset_ = EliminateVarInExpr(index, cur_vars);
    }

    info.GetNode()->index_ = index;
    if (is_const(predicate)) {
      info.GetNode()->data_alignment_ = GetInt32Const(predicate) == FREE_ALIGN ? 0 : GetInt32Const(predicate);
    } else {
      info.GetNode()->data_alignment_ = FREE_ALIGN;
    }

    if (store != nullptr) {
      info.GetNode()->dtype_ = store->value.type();
      info.GetNode()->scope_ = GetBufScope(store->buffer_var->name_hint);
      info.GetNode()->name_ = store->buffer_var->name_hint;
      info.GetNode()->data_ = Var(store->buffer_var);
    } else if (load != nullptr) {
      info.GetNode()->dtype_ = load->type;
      info.GetNode()->scope_ = GetBufScope(load->buffer_var->name_hint);
      info.GetNode()->name_ = load->buffer_var->name_hint;
      info.GetNode()->data_ = Var(load->buffer_var);
    }

    com_info_list.push_back(info);
  }

  return com_info_list;
}

/// Determining whether the stmt contains a select
/// \param stmt       - Input stmt
/// \return has_select - True if the stmt contains a select, otherwise false
bool HasSelect(const Stmt &stmt) {
  bool has_select = false;
  PostOrderVisit(stmt, [&has_select](const NodeRef &node) {
    if (node.as<Select>()) {
      has_select = true;
      return;
    }
  });
  return has_select;
}

/// Get Deep Compact Computation Info.
/// Check last few axes are continuous and merge them
/// Besides, check other axes are continuous and merge them
/// Also should update for_info
/// \param stmt        - Input stmt
/// \param dst_info_list - Computed dst_info_list, which is Output
/// \param src_info_list - Computed src_info_list, which is Output
/// \param if_info      - Obtained 'if' condition info, which is Output
/// \param for_info     - Obtained 'for-loop' info, which is Output
/// \param same_dtype   - Set check same data type mode
/// \param clean_non_linear
void GetCompactComputationInfo(const Stmt &stmt, StmtInfoList &dst_info_list, StmtInfoList &src_info_list,
                               StmtInfo &if_info, StmtInfo &for_info, bool same_dtype, bool clean_non_linear) {
  if (!dst_info_list.empty() || !src_info_list.empty()) {
    LOG(FATAL) << "Error: dst_info_list and src_info_list must be empty!";
  }

  if (!if_info.vars_.empty() || !if_info.ops_.empty() || !for_info.vars_.empty() || !for_info.ops_.empty()) {
    LOG(FATAL) << "Error: if_info and for_info must be empty!";
  }

  Array<NodeRef> stores;
  Array<NodeRef> loads;

  GetStoreAndLoads(stmt, stores, loads);
  GetIfForInfo(stmt, if_info, for_info);
  CHECK(stores.size() == 1) << "Error: Can not support zero store and multiple stores.";

  dst_info_list = GetComputationInfo(stores, for_info);
  if (loads.empty()) {
    src_info_list = {};
  } else if (loads.size() <= 4) {
    src_info_list = GetComputationInfo(loads, for_info);
  }

  if (same_dtype) {
    Type dst_dtype = Float(32);
    if (!dst_info_list.empty()) {
      dst_dtype = dst_info_list[0]->dtype_;
    } else {
      LOG(FATAL) << "No dst Info, please check.";
    }

    // for vselect operators with 4 operands
    if (src_info_list.size() == 4) {
      // only check the comparison pair types are same, and the source buffers are same with destination
      // for example: the following is allowed: float = (int < int) ? float : float
      if (src_info_list[0]->dtype_ != src_info_list[1]->dtype_) {
        LOG(FATAL) << "comparison operands can not be different data type.";
      }
      if (src_info_list[2]->dtype_ != dst_dtype) {
        LOG(FATAL) << "source buffers and dst buffer can not be different data type.";
      }
      if (src_info_list[2]->dtype_ != src_info_list[3]->dtype_) {
        LOG(FATAL) << "source buffers can not be different data type.";
      }
    } else {  // other operators (normal case)
      bool can_diff_type = false;
      // UB to CC and CC to UB can be difference type
      if (src_info_list.size() == 1 && src_info_list.size() == 1) {
        std::string dst_scope = dst_info_list[0]->scope_;
        std::string src_scope = src_info_list[0]->scope_;
        if ((dst_scope == SCOPE_CC && src_scope == SCOPE_UBUF) || (dst_scope == SCOPE_UBUF && src_scope == SCOPE_CC)) {
          can_diff_type = true;
        }
      }
      bool is_sel = HasSelect(stmt);
      for (auto src_info : src_info_list) {
        if (dst_dtype != src_info->dtype_ && !is_sel && !can_diff_type) {
          LOG(FATAL) << "Unsupported dst and src with different data type yet.";
        }
      }
    }
  }
  if (clean_non_linear) {
    CleanNonLinearVar(dst_info_list, src_info_list, if_info);
  }
}

/// Compact Computation Info List.
/// Try to merge any two vars in for_info
/// Also should update for_info, dst_info_list, src_info_list
/// \param dst_info_list - The dst_info_list to be modified
/// \param src_info_list - The src_info_list to be modified
/// \param if_info      - The if-condition as input
/// \param for_info     - The for-loop info to be modified
void CompactComputationInfoList(StmtInfoList &dst_info_list, StmtInfoList &src_info_list, const StmtInfo &if_info,
                                StmtInfo &for_info) {
  auto MergeTwoVar = [](const Var &keep_var, const Var &delete_var, StmtInfoList &dst_info_list,
                        StmtInfoList &src_info_list, StmtInfo &for_info) {
    for (auto info : dst_info_list) {
      // find var
      size_t del_idx = 0;
      size_t keep_idx = 0;
      bool suc1 = GetIndexOfElement(info->var_, delete_var, del_idx);
      bool suc2 = GetIndexOfElement(info->var_, keep_var, keep_idx);
      if (suc1 && suc2) {
        Expr times = info->shape_[del_idx];
        // update
        info.GetNode()->shape_.Set(keep_idx, info->shape_[keep_idx] * times);
        // delete
        info.GetNode()->var_ = RemoveItemAtIndex(info->var_, del_idx);
        info.GetNode()->strides_ = RemoveItemAtIndex(info->strides_, del_idx);
        info.GetNode()->shape_ = RemoveItemAtIndex(info->shape_, del_idx);
        info.GetNode()->index_ = EliminateVarInExpr(info->index_, {delete_var});
      }
    }

    for (auto info : src_info_list) {
      // find var
      size_t del_idx = 0;
      size_t keep_idx = 0;
      bool suc1 = GetIndexOfElement(info->var_, delete_var, del_idx);
      bool suc2 = GetIndexOfElement(info->var_, keep_var, keep_idx);
      if (suc1 && suc2) {
        Expr times = info->shape_[del_idx];
        // update
        info.GetNode()->shape_.Set(keep_idx, info->shape_[keep_idx] * times);
        // delete
        info.GetNode()->var_ = RemoveItemAtIndex(info->var_, del_idx);
        info.GetNode()->strides_ = RemoveItemAtIndex(info->strides_, del_idx);
        info.GetNode()->shape_ = RemoveItemAtIndex(info->shape_, del_idx);
        info.GetNode()->index_ = EliminateVarInExpr(info->index_, {delete_var});
      }
    }

    // update for-loop
    size_t keep_var_for_idx = 0;
    size_t del_var_for_idx = 0;
    bool suc3 = GetIndexOfElement(for_info.vars_, VarExpr(keep_var), keep_var_for_idx);
    bool suc4 = GetIndexOfElement(for_info.vars_, VarExpr(delete_var), del_var_for_idx);
    CHECK(suc3);
    CHECK(suc4);
    auto op1 = for_info.ops_[keep_var_for_idx].as<For>();
    auto op2 = for_info.ops_[del_var_for_idx].as<For>();
    CHECK(op1 != nullptr);
    CHECK(op2 != nullptr);
    for_info.ops_.Set(keep_var_for_idx, For::make(op1->loop_var, op1->min, op1->extent * op2->extent, op1->for_type,
                                                  op1->device_api, op1->body));
    for_info.RemoveItem(del_var_for_idx);
  };

  auto CanMergeTwoVar = [&MergeTwoVar](const Var &l_var, const Var &r_var, StmtInfoList &dst_info_list,
                                       StmtInfoList &src_info_list, StmtInfo &for_info) {
    auto EqualExpr = [](Expr l_value, Expr r_value) {
      if (l_value.as<IntImm>() && r_value.as<IntImm>()) {
        return Equal(l_value, r_value);
      }
      Expr res = div(l_value, r_value);
      return Equal(ir::ExprSimplifier().Simplify(res), 1);
    };

    int merge_type = 0;
    bool can_merge = true;
    auto all_info_list = MergeTwo(dst_info_list, src_info_list);

    for (auto &info : all_info_list) {
      size_t l_idx = 0;
      size_t r_idx = 0;
      bool suc_l = GetIndexOfElement(info->var_, l_var, l_idx);
      bool suc_r = GetIndexOfElement(info->var_, r_var, r_idx);
      if (suc_l && suc_r) {
        if (EqualExpr(info->strides_[r_idx] * info->shape_[r_idx], info->strides_[l_idx])) {
          if (merge_type == 0) {
            merge_type = 1;
          } else if (merge_type != 1) {
            can_merge = false;
          }
        } else if (EqualExpr(info->strides_[l_idx] * info->shape_[l_idx], info->strides_[r_idx])) {
          if (merge_type == 0) {
            merge_type = 2;
          } else if (merge_type != 2) {
            can_merge = false;
          }
        } else {
          can_merge = false;
        }
      } else if (suc_l || suc_r || HasVars(info->index_, l_var) || HasVars(info->index_, r_var)) {
        can_merge = false;
      }

      if (!can_merge) {
        break;
      }
    }

    if (can_merge) {
      if (merge_type == 1) {
        MergeTwoVar(r_var, l_var, dst_info_list, src_info_list, for_info);
      } else if (merge_type == 2) {
        MergeTwoVar(l_var, r_var, dst_info_list, src_info_list, for_info);
      } else {
        CHECK(0) << "\nTwo vars are not in any dst or src\n" << l_var << " " << r_var;
      }
    }

    return can_merge;
  };

  CleanNonLinearVar(dst_info_list, src_info_list, if_info);
  CHECK_EQ(dst_info_list.size(), 1);
  bool try_merge = true;
  while (try_merge) {
    size_t var_cnt = for_info.vars_.size();
    if (var_cnt <= 1) {
      try_merge = false;
      break;
    }
    bool find_merge = false;
    for (size_t i = 0; (i < var_cnt - 1) && (!find_merge); i++) {
      for (size_t j = i + 1; j < var_cnt; j++) {
        if (CanMergeTwoVar(for_info.vars_[i], for_info.vars_[j], dst_info_list, src_info_list, for_info)) {
          find_merge = true;
          break;
        }
      }
    }
    if (find_merge) {
      try_merge = true;
    } else {
      try_merge = false;
      break;
    }
  }
}

/// A helper function for single dst_info's compact
/// \param dst_info
/// \param src_info_list
/// \param if_info
/// \param for_info
void CompactComputationInfoList(StmtStoreInfo &dst_info, StmtInfoList &src_info_list, const StmtInfo &if_info,
                                StmtInfo &for_info) {
  StmtInfoList dst_info_list = {dst_info};
  CompactComputationInfoList(dst_info_list, src_info_list, if_info, for_info);
  CHECK(!dst_info_list.empty());
  dst_info = dst_info_list[0];
}

/// Remove var from for var list
/// \param forop   - For stmt list
/// \param elim_var - Var list to be eliminated
/// Remove var from for var list, for reloading
void CleanForInfoVars(StmtInfo &for_info, const Array<Var> &elim_var) {
  for (auto item : elim_var) {
    size_t idx = 0;
    if (GetIndexOfElement(for_info.vars_, item, idx)) {
      for_info.RemoveItem(idx);
    }
  }
}

/// Get max length of vector cmd, for fp16 is 128, fp32 is 64
/// \param dtype - Data type of data
/// \return int  - Max length of vector cmd
int GetVecMaxLen(const Type &dtype) {
  CHECK_NE(dtype.bits(), 0);
  int result = 8 * 32 / (dtype.bits() / 8);
  CHECK_NE(result, 0) << "Get zero Vector Max Length";
  return result;
}

/// Generate vector mask by start and end
/// \param d_type  - Data type of data
/// \param start  - Start number
/// \param end    - End number
/// \param stride - Stride of mask
/// \return Array<Expr> - Mask list
Array<Expr> GenMaskVec(const Type &d_type, unsigned int start, unsigned int end, unsigned int stride) {
  Array<Expr> mask_list;
  int vec_max_len = GetVecMaxLen(d_type);
  if (d_type.bits() == 8) {
    vec_max_len = vec_max_len / 2;
  }
  if (end == UINT_MAX) {
    end = static_cast<unsigned int>(vec_max_len);
  }
  std::vector<std::string> mask_str_list(vec_max_len, "0");
  std::string mask_str;
  CHECK_NE(stride, 0);
  for (size_t i = start; i < end; i += stride) {
    mask_str_list[vec_max_len - 1 - i] = "1";
  }
  mask_str = std::accumulate(mask_str_list.begin(), mask_str_list.end(), mask_str);

  std::string mask_high;
  std::string mask_low;
  if (vec_max_len == 128) {
    mask_high = mask_str.substr(0, 64);
    mask_low = mask_str.substr(64, 64);
  } else if (vec_max_len == 64) {
    mask_high = "0";
    mask_low = mask_str;
  } else {
    LOG(FATAL) << "Error: mask length is error.";
  }

  mask_list.push_back(make_const(UInt(64), std::stoull(mask_high, nullptr, 2)));
  mask_list.push_back(make_const(UInt(64), std::stoull(mask_low, nullptr, 2)));

  return mask_list;
}

/// Check the dst_info_list and src_info_list is the elementwise mode IR
/// \param dst_info_list - Input dst_info_list
/// \param src_info_list - Input src_info_list
/// \return bool - Whether they are of elementwise mode or not
bool IsElementwise(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list) {
  if (dst_info_list.size() != 1 || src_info_list.empty()) {
    return false;
  }
  auto &dst_var_list = dst_info_list[0]->var_;
  for (auto &src_info : src_info_list) {
    auto &src_var_list = src_info->var_;
    if (!IsSame(dst_var_list, src_var_list)) {
      return false;
    }
  }
  return true;
}

/// Check the dst_info_list and src_info_list is the broadcast mode IR
/// \param dst_info_list - Input dst_info_list
/// \param src_info_list - Input src_info_list
/// \return bool - Whether they are of broadcast mode or not
bool IsBroadcast(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list) {
  if (IsElementwise(dst_info_list, src_info_list)) {
    return false;
  }
  if (dst_info_list.size() != 1) {
    return false;
  }
  if (src_info_list.empty()) {
    return true;
  }
  auto &dst_var_list = dst_info_list[0]->var_;
  for (auto &src_info : src_info_list) {
    auto &src_var_list = src_info->var_;
    if (dst_var_list.size() < src_var_list.size()) {
      return false;
    }
    if (!dst_var_list.empty() && !src_var_list.empty()) {
      if (!Equal(dst_var_list[dst_var_list.size() - 1], src_var_list[src_var_list.size() - 1])) {
        return true;
      }
    }
    for (auto &var : src_var_list) {
      if (!IsInArray(dst_var_list, var)) {
        return false;
      }
    }
  }
  return true;
}

/// Check the dst_info_list and src_info_list is the last-axis broadcast mode IR
/// \param dst_info_list - Input dst_info_list
/// \param src_info_list - Input src_info_list
/// \return bool - Whether they are of last-axis broadcast mode or not
bool IsLastAxisBroadcast(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list) {
  if (dst_info_list.size() != 1 || src_info_list.empty()) {
    return false;
  }
  auto &dst_var_list = dst_info_list[0]->var_;
  for (auto src_info : src_info_list) {
    auto src_var_list = src_info->var_;
    if (dst_var_list.size() < src_var_list.size()) {
      return false;
    }
    if (!dst_var_list.empty() && !src_var_list.empty()) {
      // check last axis broadcast
      if (!Equal(GetItem(dst_var_list, -1), GetItem(src_var_list, -1))) {
        return true;
      }
    }
    for (auto var : src_var_list) {
      if (!IsInArray(dst_var_list, var)) {
        return false;
      }
    }
  }
  return false;
}

/// Get last-axis reduction src's index
/// \param dst_info_list
/// \param src_info_list
/// \return
int GetLastAxisReductionIdx(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list) {
  if (dst_info_list.size() != 1 || src_info_list.size() != 2) {
    return -1;
  }
  auto &dst_var_list = dst_info_list[0]->var_;
  auto &src0_var_list = src_info_list[0]->var_;
  auto &src1_var_list = src_info_list[1]->var_;
  if (IsSame(dst_var_list, src0_var_list)) {
    if (dst_var_list.size() < src1_var_list.size()) {
      if (GetInt32Const(GetItem(src_info_list[1]->strides_, -1)) > 1) {
        return -1;
      }
      if (!IsInArray(dst_var_list, src1_var_list[src1_var_list.size() - 1])) {
        return 1;
      }
    }
  } else if (IsSame(dst_var_list, src1_var_list)) {
    if (dst_var_list.size() < src0_var_list.size()) {
      if (GetInt32Const(GetItem(src_info_list[0]->strides_, -1)) > 1) {
        return -1;
      }
      if (!IsInArray(dst_var_list, src0_var_list[src0_var_list.size() - 1])) {
        return 0;
      }
    }
  }
  return -1;
}

/// Check the dst_info_list and src_info_list is the last-axis reduction mode IR
/// \param dst_info_list - Input dst_info_list
/// \param src_info_list - Input src_info_list
/// \return bool - Whether they are of last-axis reduction mode or not
bool IsLastAxisReduction(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list) {
  return GetLastAxisReductionIdx(dst_info_list, src_info_list) != -1;
}

/// Get bisection reduction's variable index
/// \param dst_info_list
/// \param src_info_list
/// \param compare_idx
/// \return
int GetBisectionReductionIdx(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list, int &compare_idx) {
  int block_size = GetScopeBlockSize(dst_info_list[0], src_info_list[0]);
  CHECK_NE(block_size, 0);
  CHECK_NE(dst_info_list[0]->dtype_.bits(), 0);
  int vec_max_len = 256 / (dst_info_list[0]->dtype_.bits() / 8);
  if (dst_info_list.size() != 1 || src_info_list.size() != 2) {
    return 0;
  }
  auto &dst_var_list = dst_info_list[0]->var_;
  auto &src0_var_list = src_info_list[0]->var_;
  auto &src1_var_list = src_info_list[1]->var_;
  Array<Var> compare_var_list;
  compare_idx = 1;
  if (IsSame(dst_var_list, src0_var_list)) {
    compare_var_list = src1_var_list;
  } else if (IsSame(dst_var_list, src1_var_list)) {
    compare_idx = 0;
    compare_var_list = src0_var_list;
  }
  if (dst_var_list.size() < compare_var_list.size()) {
    // check Bisection mode available.
    // find the last different axis
    bool get_reduce_axis = false;
    int i;
    for (i = -1; i >= -static_cast<int>(compare_var_list.size()); --i) {
      if (i < -static_cast<int>(dst_var_list.size()) || !IsTwoItemEqual(dst_var_list, compare_var_list, i)) {
        get_reduce_axis = true;
        break;
      }
    }

    int last_dim_shape = GetInt32Const(GetItem(src_info_list[compare_idx]->shape_, -1));
    if (get_reduce_axis && (last_dim_shape + block_size - 1) / block_size * block_size *
                               GetIntConst(GetItem(src_info_list[compare_idx]->shape_, i)) >=
                             2 * vec_max_len) {
      return i;
    }
  }

  return 0;
}

/// Check the dst_info_list and src_info_list is the bisection reduction mode IR
/// \param dst_info_list - Input dst_info_list
/// \param src_info_list - Input src_info_list
/// \return bool - Whether they are of bisection reduction mode or not
bool IsBisectionReduction(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list) {
  int compare_idx = 0;
  int idx = GetBisectionReductionIdx(dst_info_list, src_info_list, compare_idx);
  return idx < 0;
}

bool HasVars(const Expr &index, const Var &vec_var) {
  Array<Var> vars = GetVarsInExpr(index);
  bool find_flag = false;
  for (size_t i = 0; i < vars.size(); i++) {
    if (vars[i]->name_hint == vec_var->name_hint) {
      find_flag = true;
      break;
    }
  }
  return find_flag;
}

int GetVectorizedVarPosition(const Expr &index, Array<Var> &loop_vars) {
  int pos = -1;
  for (size_t i = 0; i < loop_vars.size(); i++) {
    Array<Expr> coefs = air::arith::DetectLinearEquation(index, {loop_vars[i]});
    if (coefs.size() == 2) {
      if (coefs[0].as<IntImm>() && coefs[0].as<IntImm>()->value == 1) {
        pos = static_cast<int>(i);
      }
    }
  }
  return pos;
}

std::string GetOpType(const Expr &value) {
  if (value.as<Add>()) {
    return value.as<Add>()->_type_key;
  }
  if (value.as<Sub>()) {
    return value.as<Sub>()->_type_key;
  }
  if (value.as<Mul>()) {
    return value.as<Mul>()->_type_key;
  }
  if (value.as<Div>()) {
    return value.as<Div>()->_type_key;
  }
  if (value.as<Mod>()) {
    return value.as<Mod>()->_type_key;
  }
  if (value.as<FloorDiv>()) {
    return value.as<FloorDiv>()->_type_key;
  }
  if (value.as<FloorMod>()) {
    return value.as<FloorMod>()->_type_key;
  }
  if (value.as<Min>()) {
    return value.as<Min>()->_type_key;
  }
  if (value.as<Max>()) {
    return value.as<Max>()->_type_key;
  }
  if (value.as<Call>()) {
    return value.as<Call>()->name;
  }
  if (value.as<Load>() || value.as<IntImm>() || value.as<FloatImm>()) {
    return "DMACopy";
  }
  return "undefined";
}

/// TVM Function Register, enable python code to call these cpp function.
TVM_REGISTER_API("cce_util.GetCceAxis").set_body([](TVMArgs args, TVMRetValue *ret) { *ret = GetCceAxis(); });

TVM_REGISTER_API("cce_util.EliminateVarInExpr").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = EliminateVarInExpr(args[0], args[1]);
});

TVM_REGISTER_API("cce_util.GetBufScope").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = GetBufScope(args[0]);
});

TVM_REGISTER_API("cce_util.GetVarsInExpr").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = GetVarsInExpr(args[0]);
});

TVM_REGISTER_API("cce_util.IsElementwise").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = IsElementwise(args[0], args[1]);
});

TVM_REGISTER_API("cce_util.IsBroadcast").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = IsBroadcast(args[0], args[1]);
});

TVM_REGISTER_API("cce_util.IsLastAxisReduction").set_body([](const TVMArgs args, TVMRetValue *const ret) {
  *ret = IsLastAxisReduction(args[0], args[1]);
});
}  // namespace akg
