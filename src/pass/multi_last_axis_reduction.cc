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

/**
 * ir before:
 * // attr [0] pragma_emit_insn = "vec_binary_add"
 * for (cc3, 0, 2) {
 *    for (cc4, 0, 5) {
 *      input1_red_local_UB[0] = (input1_local_UB[((cc3*5) + cc4)] + input1_red_local_UB[0])
 *    }
 * }
 *
 * ir after:
 * // attr [tmp_0_local_UB] storage_scope = "local.UB"
 * allocate tmp_0_local_UB[float16 * 5]
 * // attr [0] pragma_emit_insn = "vec_binary_add"
 * for (cc3, 0, 2) {
 *  for (cc4, 0, 5) {
 *     tmp_0_local_UB[cc4] = (input1_local_UB[((cc3*5) + cc4)] + tmp_0_local_UB[cc4])
 *  }
 * }
 * // attr [0] pragma_emit_insn = "vec_binary_add"
 * for (cc4, 0, 5) {
 *   input1_red_local_UB[0] = (input1_red_local_UB[0] + tmp_0_local_UB[cc4])
 * }
 */

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <numeric>

#include "emit_insn/insn_info.h"
#include "emit_insn/insn_builder.h"

namespace akg {
namespace ir {
class MultiLastAxisReduction : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn") {
      std::vector<Stmt> loopArray;
      std::vector<Expr> extentsArray;
      Array<Stmt> loopTmp;
      Array<Expr> strideArray;
      Array<Var> varArray;

      Stmt body = op->body;
      CHECK(op->value.as<StringImm>());
      std::string str = op->value.as<StringImm>()->value;
      const std::set<std::string> supportList = {"vec_binary_add", "vec_binary_max", "vec_binary_min"};
      if (supportList.count(str) != 0) {
        Array<NodeRef> stores, loads;
        StmtInfo ifInfo, forInfo;
        StmtInfoList srcInfoList, dstInfoList;

        GetStoreAndLoads(body, stores, loads);
        GetIfForInfo(body, ifInfo, forInfo);
        srcInfoList = GetComputationInfo(loads, forInfo);
        dstInfoList = GetComputationInfo(stores, forInfo);

        CHECK(srcInfoList.size() == 2) << "Error: can only handle two operators of reduction";

        // check body has For or IfThenElse struct
        while (body->IsInstance<For>() || body->IsInstance<IfThenElse>() || body->IsInstance<AttrStmt>()) {
          if (body->IsInstance<For>()) {
            loopArray.push_back(body);
            const auto op_for = body.as<For>();
            CHECK(op_for);
            body = op_for->body;
          } else if (body->IsInstance<IfThenElse>()) {
            loopArray.push_back(body);
            const auto op_for = body.as<IfThenElse>();
            CHECK(op_for);
            body = op_for->then_case;
          } else {
            body = body.as<AttrStmt>()->body;
          }
        }

        const auto storeTmp = body.as<Store>();
        CHECK(storeTmp);

        // check the condition for multi-last-axes : index/loop_size/value
        auto operators = GetBinaryOpExprChildren(storeTmp->value);
        CHECK_EQ(operators.size(), 2);
        Expr a = operators[0];
        Expr b = operators[1];
        Expr other;

        CHECK(a.as<Load>());
        bool sameAsA =
          Equal(storeTmp->buffer_var, a.as<Load>()->buffer_var) && Equal(storeTmp->index, a.as<Load>()->index);
        CHECK(b.as<Load>());
        bool sameAsB =
          Equal(storeTmp->buffer_var, b.as<Load>()->buffer_var) && Equal(storeTmp->index, b.as<Load>()->index);
        if (sameAsA || sameAsB) {
          Type type_tmp;
          if (sameAsA) {
            type_tmp = b.type();
            other = b;
          } else {
            type_tmp = a.type();
            other = a;
          }

          if (Equal(srcInfoList[0]->index_, storeTmp->index)) {
            varArray = srcInfoList[1]->var_;
            strideArray = srcInfoList[1]->strides_;
          } else if (Equal(srcInfoList[1]->index_, storeTmp->index)) {
            varArray = srcInfoList[0]->var_;
            strideArray = srcInfoList[0]->strides_;
          }

          varArray = Reverse(varArray);
          strideArray = Reverse(strideArray);

          bool hasVar = false;
          for (auto var : dstInfoList[0]->var_) {
            if (!varArray.empty() && Equal(var, varArray[0])) {
              hasVar = true;
            }
          }

          if ((srcInfoList[0]->var_.size() - dstInfoList[0]->var_.size() > 1 ||
               srcInfoList[1]->var_.size() - dstInfoList[0]->var_.size() > 1) &&
              !hasVar) {
            Expr loadTmp;
            Expr storeResult;
            Expr newIdx = Expr(0);
            Stmt stmt;

            VarExpr tmpBuffer = VarExpr("tmp_0_local_UB", type_tmp);
            Expr storeLeft = Load::make(type_tmp, storeTmp->buffer_var, storeTmp->index, storeTmp->predicate);

            std::vector<Stmt> loopTmpArray = loopArray;

            std::reverse(loopArray.begin(), loopArray.end());

            for (auto st : loopTmpArray) {
              const For *opFor = st.as<For>();
              CHECK(opFor);
              for (auto dstVar : dstInfoList[0]->var_) {
                if (Equal(dstVar, opFor->loop_var) && !Equal(dstVar, varArray[0])) {
                  Insert(loopTmp, 0, st);
                }
              }
              if (Equal(opFor->loop_var, varArray[0])) {
                Insert(loopTmp, 0, st);
              }
            }

            for (size_t i = 0; i < loopTmp.size(); ++i) {
              newIdx += loopTmp[i].as<For>()->loop_var * strideArray[i];
            }

            extentsArray.push_back(loopTmp[loopTmp.size() - 1].as<For>()->extent);
            Var tmpvar = loopTmp[loopTmp.size() - 1].as<For>()->loop_var;
            size_t idx = 0;
            if (GetIndexOfElement(varArray, tmpvar, idx)) {
              extentsArray.push_back(strideArray[idx]);
            }

            loadTmp = Load::make(type_tmp, tmpBuffer, newIdx, storeTmp->predicate);

            Expr lastResult;
            CHECK(other.defined());
            Expr broadcastNum = MakeConstScalar(type_tmp, 0);
            if (str == "vec_binary_max") {
              storeResult = Max::make(other, loadTmp);
              lastResult = Max::make(loadTmp, storeLeft);
              broadcastNum = type_tmp.min();
            } else if (str == "vec_binary_min") {
              storeResult = Min::make(other, loadTmp);
              lastResult = Min::make(loadTmp, storeLeft);
              broadcastNum = type_tmp.max();
            } else {
              storeResult = other + loadTmp;
              lastResult = loadTmp + storeLeft;
            }

            Stmt stForOnce = Store::make(tmpBuffer, storeResult, newIdx, storeTmp->predicate);
            Stmt stForTwice = Store::make(storeTmp->buffer_var, lastResult, storeTmp->index, storeTmp->predicate);
            Stmt stBroadcast = Store::make(tmpBuffer, broadcastNum, newIdx, storeTmp->predicate);

            for (auto loop : loopArray) {
              const For *tmp = loop.as<For>();
              CHECK(tmp);
              stForOnce = For::make(tmp->loop_var, tmp->min, tmp->extent, tmp->for_type, tmp->device_api, stForOnce);
            }

            for (auto loop : loopTmp) {
              const For *tmp = loop.as<For>();
              CHECK(tmp);
              stForTwice = For::make(tmp->loop_var, tmp->min, tmp->extent, tmp->for_type, tmp->device_api, stForTwice);
              stBroadcast =
                For::make(tmp->loop_var, tmp->min, tmp->extent, tmp->for_type, tmp->device_api, stBroadcast);
            }

            stForOnce = AttrStmt::make(VarExpr("0", Int(32)), "pragma_emit_insn", Expr(str), stForOnce);
            stForTwice = AttrStmt::make(VarExpr("0", Int(32)), "pragma_emit_insn", Expr(str), stForTwice);
            stBroadcast = AttrStmt::make(VarExpr("0", Int(32)), "pragma_emit_insn", Expr("broadcast"), stBroadcast);

            stmt = Block::make({stBroadcast, stForOnce, stForTwice});
            stmt = Allocate::make(tmpBuffer, type_tmp, extentsArray, const_true(), stmt);
            stmt = AttrStmt::make(tmpBuffer, "storage_scope", Expr("local.UB"), stmt);

            loopArray.clear();
            extentsArray.clear();
            return stmt;
          }
          loopArray.clear();
          extentsArray.clear();
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }
};

class BroadcastCalculate : public IRMutator {
 public:
  explicit BroadcastCalculate(bool is_dynamic) : dynamic(is_dynamic), buffer2exts_() {}
  ~BroadcastCalculate() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
        op->value.as<StringImm>()->value.find("vec_single_") != std::string::npos) {
      auto body = op->body;

      Array<NodeRef> stores, loads;
      StmtInfo ifInfo, forInfo;

      GetStoreAndLoads(body, stores, loads);
      CHECK_EQ(stores.size(), 1);
      CHECK_GE(loads.size(), 1);

      GetIfForInfo(body, ifInfo, forInfo);
      auto srcInfoList = GetComputationInfo(loads, forInfo);
      auto dstInfoList = GetComputationInfo(stores, forInfo);
      if (CheckBroadcast(dstInfoList, srcInfoList)) {
        Map<Var, Var> old2new;
        auto vars = GetVarsInExpr(srcInfoList[0]->index_);
        for (auto e : vars) {
          old2new.Set(e, Var(e->name_hint));
        }
        auto newForInfo = forInfo.Copy();
        for (size_t i = 0; i != forInfo.vars_.size(); ++i) {
          auto forVar = forInfo.vars_[i];
          if (old2new.count(forVar) != 0) {
            newForInfo.vars_.Set(i, old2new[forVar]);

            auto op_for = forInfo.ops_[i].as<For>();
            CHECK(op_for != nullptr);
            newForInfo.ops_.Set(
              i, For::make(old2new[forVar], op_for->min, op_for->extent, op_for->for_type, op_for->device_api, body));
          }
        }
        for (auto i = newForInfo.vars_.size(); i > 0; --i) {
          if (!IsInArray(vars, forInfo.vars_[i - 1])) {
            newForInfo.RemoveItem(i - 1);
          }
        }

        auto st_op = stores[0].as<Store>();
        CHECK(st_op != nullptr);
        auto ld_op = loads[0].as<Load>();
        CHECK(ld_op != nullptr);

        // vec_single
        auto tmp_buffer = Var("vec_single_tmp_local_UB", ld_op->type);
        auto st = Store::make(tmp_buffer, st_op->value, ld_op->index, st_op->predicate);

        auto stmt = GenIfAndFor(st, ifInfo, newForInfo);

        Map<Var, Expr> tmp_map;
        for (auto e : old2new) {
          tmp_map.Set(e.first, e.second);
        }
        stmt = Substitute(stmt, tmp_map);
        stmt = AttrStmt::make(make_zero(Int(32)), op->attr_key, op->value, stmt);

        // broadcast
        auto new_ld = Load::make(ld_op->type, tmp_buffer, ld_op->index, ld_op->predicate);
        auto stmt_broad = Store::make(st_op->buffer_var, new_ld, st_op->index, st_op->predicate);
        stmt_broad = GenIfAndFor(stmt_broad, ifInfo, forInfo);
        stmt_broad = AttrStmt::make(make_zero(Int(32)), op->attr_key, Expr("vec_broadcast"), stmt_broad);

        stmt = Block::make(stmt, stmt_broad);
        stmt = Allocate::make(tmp_buffer, ld_op->type, buffer2exts_[ld_op->buffer_var], const_true(), stmt);
        stmt = AttrStmt::make(tmp_buffer, "storage_scope", Expr("local.UB"), stmt);
        return stmt;
      }
      return s;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    buffer2exts_.Set(op->buffer_var, op->extents);
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool CheckBroadcast(const StmtInfoList &dst, const StmtInfoList &src) {
    CHECK_EQ(dst.size(), 1);
    auto &dstVars = dst[0]->var_;
    bool is_equal = false;
    if (dynamic) {
      is_equal = dstVars.empty() || !Equal(GetItem(dst[0]->strides_, -1), 1);
    } else {
      is_equal = (dstVars.empty() || GetIntConst(GetItem(dst[0]->strides_, -1)) != 1);
    }
    if (is_equal) {
      return false;
    }

    auto lastDstVar = GetItem(dstVars, -1);
    for (auto &srcInfo : src) {
      if (IsInArray(srcInfo->var_, lastDstVar)) {
        return false;
      }
    }
    return true;
  }

  bool dynamic = false;
  Map<Var, Array<Expr>> buffer2exts_;
};

Stmt MultiLastAxisReductions(Stmt stmt, bool is_dynamic = false) {
  stmt = MultiLastAxisReduction().Mutate(stmt);
  stmt = BroadcastCalculate(is_dynamic).Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
