/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "tvm.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
/*
Example 1, kImmOnly:
Remap constant index to 0,1,2... and set the shape to max_index+1.

// attr [placeholder(A, 0x1815e10)] realize_scope = "local.UB"
realize A([0, 10]) {
  for (cc0, 0, 19) {
    if ((9 <= cc0)) {
      produce A {
        A(0) =input_1((cc0 - 9))
      }
    }
    if ((cc0 <= 9)) {
      produce A {
        A(9) =input_1(cc0)
      }
    }
    if ((cc0 <= 9)) {
      reduce_1_local_UB(0) =max(reduce_1_local_UB(0), A(9))
    }
-->
// attr [placeholder(A, 0x1815e10)] realize_scope = "local.UB"
realize A([0, 2]) {
  for (cc0, 0, 19) {
    if ((9 <= cc0)) {
      produce A {
        A(0) =input_2((cc0 - 9))
      }
    }
    if ((cc0 <= 9)) {
      produce A {
        A(1) =input_2(cc0)
      }
    }
    if ((cc0 <= 9)) {
      reduce_1_local_UB(0) =max(reduce_1_local_UB(0), A(1))
    }
===============================================================================
Example 2, kVarWithOneProvide:
// attr [placeholder(A, 0x4323630)] realize_scope = "local.UB"
realize A<float32>([0, 112]) {                 realize A<float32>([0, 2]) {
  produce A {                                    produce A {
    for (cc1, (cc0*2), 2) {                        for (cc1, (cc0*2), 2) {
      A(cc1) = input_1(cc1)                          A(cc1 - (cc0*2)) = input_1(cc1)
    }                                              }
  }                                      --->    }
  ...                                            ...
  for (cc2, 0, 2) {                              for (cc2, 0, 2) {
    C(cc2) = (A((cc0*2) + cc2) * B(cc2))           C(cc2) = (A(cc2) * B(cc2))
  }                                              }
}                                              }
===============================================================================
Example 3, kVarWithMultiProvide: (not support now)
realize A(20) {                      realize A(10) {
  for (cc0, 0, 5) {                    for (cc0, 0, 5) {
    A(cc0) = 0                           A(cc0) = 0
  }                       --->         }
  for (cc0, 10, 5) {                   for (cc0, 10, 5) {
    A(cc0) = 1                           A(cc0 - 5) = 1
  }                                    }
}                                    }
*/

enum IndexType {
  kDefault,
  kImmOnly,
  kImmInCall,
  kVarInCall,
  kVarWithOneProvide,
  kVarWithMultiProvide,
  kNoSupport,  // we don't handle it in this pass.
};

class CheckIndex : public IRVisitor {
 public:
  CheckIndex(const FunctionRef &func, size_t args_num, Array<Expr> &max_index, std::vector<IndexType> &index_type,
             std::vector<std::map<int, int>> &index_remap, std::vector<std::vector<Range>> &index_ranges,
             const std::set<const Variable *> &loop_vars)
      : func_(func),
        max_index_(max_index),
        index_type_(index_type),
        index_remap_(index_remap),
        index_ranges_(index_ranges),
        outer_loop_vars_(loop_vars) {
    for (size_t i = 0; i < args_num; i++) {
      max_index_.push_back(make_zero(Int(32)));
    }
    index_type_.resize(args_num, IndexType::kDefault);
    index_remap_.resize(args_num);
    index_ranges_.resize(args_num);
  }
  ~CheckIndex() override = default;

  bool Run(const Stmt &s) {
    bool visited = true;
    this->Visit(s);
    for (size_t i = 0; i < index_type_.size(); i++) {
      CHECK_NE(index_type_[i], IndexType::kImmInCall) << func_ << " Call without Provide";
      CHECK_NE(index_type_[i], IndexType::kVarInCall) << func_ << " Call without Provide";
      if (index_type_[i] == IndexType::kVarWithMultiProvide) {
        if (index_ranges_[i].size() == 1) {
          index_type_[i] = IndexType::kVarWithOneProvide;
        } else {
          // not support kVarWithMultiProvide now.
          index_type_[i] = IndexType::kNoSupport;
        }
      } else if (index_type_[i] == IndexType::kDefault) {
        visited = false;
      }
    }
    return visited;
  }

 private:
  void Visit_(const For *op) final {
    var_range_[op->loop_var.get()] = Range::make_by_min_extent(op->min, op->extent);
    IRVisitor::Visit_(op);
    var_range_.erase(op->loop_var.get());
  }
  void Visit_(const Provide *op) final {
    if (op->func == func_) {
      UpdateIndexType(op->args, true);
      for (size_t i = 0; i < op->args.size(); i++) {
        Expr var = op->args[i];
        Expr max_var = max_index_[i];
        if (index_type_[i] == IndexType::kImmOnly) {
          // remap all Imm index to 0,1,2...
          // and record the max index of each axis.
          int v = static_cast<int>(var.as<IntImm>()->value);
          if (index_remap_[i].count(v) == 0) {
            int maxv = static_cast<int>(max_var.as<IntImm>()->value);
            index_remap_[i][v] = maxv++;
            max_index_.Set(i, make_const(Int(32), maxv));
          }
        } else if (index_type_[i] == IndexType::kVarWithOneProvide ||
                   index_type_[i] == IndexType::kVarWithMultiProvide) {
          auto it = var_range_.find(Downcast<Var>(var).get());
          CHECK(it != var_range_.end()) << " " << func_ << ": " << var;
          // only support const extent loop.
          if (!it->second->extent.as<IntImm>()) {
            index_type_[i] = IndexType::kNoSupport;
            index_ranges_[i].clear();
            continue;
          }
          InsertRangeWithoutDuplicate(index_ranges_[i], it->second);
        }
      }
    }
    IRVisitor::Visit_(op);
  }
  void Visit_(const Call *op) final {
    if (op->func == func_) {
      UpdateIndexType(op->args, false);
      for (size_t i = 0; i < index_ranges_.size(); ++i) {
        // If the min of loop_var is a variable or expression,
        // then it needs to check that all Calls of the UB to be processed are in the same loop scope.
        auto CheckVarOutOfScope = [this](const Range &r) {
          bool result = false;
          PostOrderVisit(r->min, [this, &result](const NodeRef &node) {
            if (auto var = node.as<Variable>()) {
              if (var_range_.count(var) == 0 && outer_loop_vars_.count(var) == 0) {
                result = true;
              }
            }
          });
          return result;
        };
        if (std::any_of(index_ranges_[i].begin(), index_ranges_[i].end(), CheckVarOutOfScope)) {
          index_type_[i] = IndexType::kNoSupport;
          index_ranges_[i].clear();
        }
      }
    }
    IRVisitor::Visit_(op);
  }

 private:
  IndexType CheckIndexType(const Expr &var, bool in_provide) {
    IndexType t;
    if (var.as<IntImm>()) {
      if (in_provide) {
        t = IndexType::kImmOnly;
      } else {
        t = IndexType::kImmInCall;
      }
    } else {  // Var or complex expression.
      if (in_provide) {
        if (var.as<Variable>()) {
          t = IndexType::kVarWithOneProvide;
        } else {
          t = IndexType::kNoSupport;
        }
      } else {
        t = IndexType::kVarInCall;
      }

      /* no support local.L1 as loop_normlize doesn't fix its Var in Provide*/
      if (func_->func_name().find(LOCAL_C1) != std::string::npos) {
        t = IndexType::kNoSupport;
      }
    }
    return t;
  }
  void UpdateIndexType(const Array<Expr> &op_args, bool in_provide) {
    for (size_t i = 0; i < op_args.size(); i++) {
      Expr var = op_args[i];
      IndexType t = CheckIndexType(var, in_provide);
      // use state trans table to indicate the type change rules.
      // relation no in this table means kNoSupport.
      // Same type transfer is omitted in this table.  e.g. kImmOnly + kImmOnly -> kImmOnly.
      IndexType state_table[][3] = {
        {kImmOnly, kImmInCall, kImmOnly},
        {kImmInCall, kVarInCall, kVarInCall},
        {kImmInCall, kVarWithOneProvide, kVarWithOneProvide},
        {kImmInCall, kVarWithMultiProvide, kVarWithMultiProvide},
        {kVarInCall, kVarWithOneProvide, kVarWithOneProvide},
        {kVarInCall, kVarWithMultiProvide, kVarWithMultiProvide},
        {kVarWithOneProvide, kVarWithOneProvide, kVarWithMultiProvide},
      };
      if (index_type_[i] == IndexType::kDefault) {
        index_type_[i] = t;
      } else {
        bool found = false;
        for (const auto &state : state_table) {
          if (((t == state[0]) && (index_type_[i] == state[1])) || ((t == state[1]) && (index_type_[i] == state[0]))) {
            index_type_[i] = state[2];
            found = true;
            break;
          }
        }
        if (!found) {
          if (t != index_type_[i]) {
            index_type_[i] = IndexType::kNoSupport;
          }
        }
      }
    }
  }
  void InsertRangeWithoutDuplicate(std::vector<Range> &ranges, const Range &new_range) {
    bool found = false;
    for (const auto &r : ranges) {
      auto min = Simplify(r->min - new_range->min);
      auto extent = Simplify(r->extent - new_range->extent);
      if (min.as<IntImm>() && min.as<IntImm>()->value == 0 && extent.as<IntImm>() && extent.as<IntImm>()->value == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      ranges.push_back(new_range);
    }
  }

  FunctionRef func_;
  Array<Expr> &max_index_;
  std::vector<IndexType> &index_type_;
  std::vector<std::map<int, int>> &index_remap_;
  std::vector<std::vector<Range>> &index_ranges_;
  std::unordered_map<const Variable *, Range> var_range_;
  const std::set<const Variable *> &outer_loop_vars_;
};

class RealizeCompressor : public IRMutator {
 public:
  RealizeCompressor() = default;
  ~RealizeCompressor() override = default;

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "realize_scope") {
      if (!op->value.as<StringImm>() || op->value.as<StringImm>()->value.empty()) {
        LOG(FATAL) << "realize scope is undefined for " << op->node;
      }
      /* local.L1_tmp scope has not Provide, but has Call. we don't handle it */
      if (op->value.as<StringImm>()->value != "local.L1_tmp") {
        realize_.insert(op->node.get());
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    loop_vars_.insert(op->loop_var.get());
    auto s2 = IRMutator::Mutate_(op, s);
    loop_vars_.erase(op->loop_var.get());
    return s2;
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    if (realize_.count(op->func.get()) == 0) {
      return IRMutator::Mutate_(op, s);
    }
    CheckIndex checker(op->func, op->bounds.size(), realize_max_index_[op->func], index_type_[op->func],
                       index_remap_[op->func], index_ranges_[op->func], loop_vars_);
    if (!checker.Run(s)) {
      ClearRealize(op);
      return IRMutator::Mutate_(op, s);
    }
    CalVarMaxIndex(op->func);
    Stmt stmt = this->Mutate(op->body);
    Array<Range> newr;
    const auto &max_index = realize_max_index_[op->func];
    const auto &index_type = index_type_[op->func];

    CHECK_LE(max_index.size(), index_type.size());
    CHECK_LE(max_index.size(), op->bounds.size());
    for (size_t i = 0; i < max_index.size(); i++) {
      if (index_type[i] == IndexType::kImmOnly || index_type[i] == IndexType::kVarWithOneProvide) {
        Expr var = max_index[i];
        CHECK(var.as<IntImm>() && !is_zero(var))
          << "realize extent [" << var << "] cannot be var or zero. name=" << op->func;
        newr.push_back(Range::make_by_min_extent(op->bounds[i]->min, var));
      } else {
        newr.push_back(op->bounds[i]);
      }
    }
    ClearRealize(op);
    return Realize::make(op->func, op->value_index, op->type, newr, op->condition, stmt);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (realize_.count(op->func.get()) == 0) {
      return IRMutator::Mutate_(op, s);
    }
    Array<Expr> new_args;
    bool changed = ProcIndexRemap(op->func, op->args, new_args);
    if (changed) {
      Expr e = this->Mutate(op->value);
      return Provide::make(op->func, op->value_index, e, new_args);
    }
    return IRMutator::Mutate_(op, s);
  }
  Expr Mutate_(const Call *op, const Expr &e) final {
    if (realize_.count(op->func.get()) == 0) {
      return IRMutator::Mutate_(op, e);
    }
    Array<Expr> new_args;
    bool changed = ProcIndexRemap(op->func, op->args, new_args);
    if (changed) {
      return Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);
    }
    return IRMutator::Mutate_(op, e);
  }

  bool ProcIndexRemap(const FunctionRef &func, const Array<Expr> &op_args, Array<Expr> &new_args) {
    const auto &index_type = index_type_[func];
    bool changed = false;
    CHECK_GE(index_type.size(), op_args.size());
    for (size_t i = 0; i < op_args.size(); i++) {
      Expr var = op_args[i];
      if (index_type[i] == IndexType::kImmOnly) {
        changed = ProcImmIndexRemap(func, i, var, new_args) || changed;
      } else if (index_type[i] == IndexType::kVarWithOneProvide) {
        changed = ProcVarWithOneProvideRemap(func, i, var, new_args) || changed;
      } else {
        new_args.push_back(var);
      }
    }
    return changed;
  }
  bool ProcImmIndexRemap(const FunctionRef &func, size_t i, const Expr &var, Array<Expr> &new_args) {
    auto &index_remap = index_remap_[func];
    CHECK(var.as<IntImm>());
    int v = static_cast<int>(var.as<IntImm>()->value);
    CHECK_GT(index_remap.size(), i);
    CHECK(index_remap[i].count(v)) << "unmapped index " << v
                                   << ". "
                                      "Access before initialization. name="
                                   << func;
    int newv = index_remap[i][v];
    new_args.push_back(make_const(Int(32), newv));
    return newv != v;
  }
  bool ProcVarWithOneProvideRemap(const FunctionRef &func, size_t i, const Expr &var, Array<Expr> &new_args) {
    CHECK_GT(index_ranges_[func].size(), i);
    auto min = index_ranges_[func][i][0]->min;
    auto newv = Simplify(var - min);
    new_args.push_back(newv);
    return !newv.same_as(var);
  }

  void CalVarMaxIndex(const FunctionRef &func) {
    auto &index_type = index_type_[func];
    auto &max_index = realize_max_index_[func];
    auto &index_ranges = index_ranges_[func];
    CHECK_LE(max_index.size(), index_type.size());
    CHECK_LE(max_index.size(), index_ranges.size());
    for (size_t i = 0; i < max_index.size(); i++) {
      if (index_type[i] == IndexType::kVarWithOneProvide) {
        CHECK_EQ(index_ranges[i].size(), 1);
        max_index.Set(i, index_ranges[i][0]->extent);
      }
    }
  }

  void ClearRealize(const Realize *op) {
    realize_.erase(op->func.get());
    realize_max_index_.erase(op->func);
    index_type_.erase(op->func);
    index_remap_.erase(op->func);
    index_ranges_.erase(op->func);
  }

  std::unordered_set<const Node *> realize_;
  std::unordered_map<FunctionRef, Array<Expr>, air::NodeHash, air::NodeEqual> realize_max_index_;
  std::unordered_map<FunctionRef, std::vector<IndexType>, air::NodeHash, air::NodeEqual> index_type_;
  std::unordered_map<FunctionRef, std::vector<std::map<int, int>>, air::NodeHash, air::NodeEqual> index_remap_;
  std::unordered_map<FunctionRef, std::vector<std::vector<Range>>, air::NodeHash, air::NodeEqual> index_ranges_;
  std::set<const Variable *> loop_vars_;
};

class LoadIm2colCheck : public IRVisitor {
 public:
  bool is_load_im2col_{false};

 private:
  void Visit_(const Evaluate *op) override {
    auto call = op->value.as<Call>();
    if (call) {
      constexpr int im2col_argnum = 23;
      const std::string im2col_callname = "cce_img2col_ub";
      if (call->name == im2col_callname && call->args.size() == im2col_argnum) {
        is_load_im2col_ = true;
      }
    }
    IRVisitor::Visit_(op);
  }
};

Stmt RealizeCompress(Stmt stmt) {
  LoadIm2colCheck checker;
  checker.Visit(stmt);
  if (checker.is_load_im2col_) {
    return stmt;
  }
  stmt = RealizeCompressor().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
