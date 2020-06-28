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
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>

#include "emit_insn/insn_info.h"

namespace akg {
namespace ir {
class BroadcastVecRewriter : public IRMutator {
 public:
  BroadcastVecRewriter() = default;
  ~BroadcastVecRewriter() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if ((op->attr_key == "pragma_emit_insn") && op->value.as<StringImm>() &&
        (op->value.as<StringImm>()->value.find("vec_binary") != std::string::npos)) {
      std::set<std::string> supportList = {"vec_binary_add", "vec_binary_mul", "vec_binary_sub"};
      if (supportList.count(op->value.as<StringImm>()->value) != 0) {
        StmtInfoList dstInfoList;
        StmtInfoList srcInfoList;
        StmtInfo ifInfo;

        GetCompactComputationInfo(op->body, dstInfoList, srcInfoList, ifInfo, forInfo, true);
        if (IsBroadcast(dstInfoList, srcInfoList) && !IsLastAxisBroadcast(dstInfoList, srcInfoList)) {
          auto dstInfo = dstInfoList[0];
          auto broadSrcInfo = srcInfoList[0];

          if (broadSrcInfo->var_.size() > srcInfoList[1]->var_.size()) {
            broadSrcInfo = srcInfoList[1];
            fullSrcIndex = 0;
          }

          bool caseNotMatch = ((broadSrcInfo->var_.size() == srcInfoList[fullSrcIndex]->var_.size()) ||
                               (srcInfoList[fullSrcIndex]->shape_.size() < 2));
          if (caseNotMatch) {
            ResetParam();
            return IRMutator::Mutate_(op, s);
          }

          auto dtype = dstInfo->dtype_;
          int lastDimShape = GetInt32Const(GetItem(broadSrcInfo->shape_, -1));
          blkSize = GetUbBlkSize(dtype);
          int fullRepeatSize = blkSize * 8;
          CHECK_NE(blkSize, 0);
          CHECK_NE(lastDimShape, 0);

          int remainDim = fullRepeatSize / lastDimShape;
          if (remainDim == 0) {
            ResetParam();
            return IRMutator::Mutate_(op, s);
          }

          int secLastDimShape = GetInt32Const(GetItem(srcInfoList[fullSrcIndex]->shape_, -2));
          bool secDimFit = (secLastDimShape % remainDim == 0);
          bool dimFit = ((broadSrcInfo->var_.size() == 1) && (lastDimShape % blkSize == 0) &&
                         (lastDimShape < fullRepeatSize) && (lastDimShape > blkSize));
          bool forLoopFit = (dstInfo->var_.size() >= 2);
          size_t idx = 0;
          if (forLoopFit) {
            bool getVar = GetIndexOfElement(forInfo.vars_, GetItem(dstInfo->var_, -2), idx);
            if (getVar) {
              forLoopFit = GetInt32Const(GetItem(forInfo.ops_, idx).as<For>()->extent) % remainDim == 0;
            } else {
              forLoopFit = false;
            }
          }

          if (secDimFit && dimFit && forLoopFit && remainDim > 1) {
            newExtent = GetInt32Const(GetItem(forInfo.ops_, idx).as<For>()->extent) / remainDim;
            tmpBuffer = VarExpr("tmp_broadcast_" + std::to_string(broadBufferCount++) + "_local_UB", dtype);
            varList = dstInfo->var_;
            strideList = dstInfo->strides_;
            SetItem(strideList, -1, Expr(1));
            SetItem(strideList, -2, Expr(fullRepeatSize));
            inAttr = true;
            auto newStmt = this->Mutate(op->body);
            newStmt = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", op->value, newStmt);

            auto outerLoopVar = Var("broad_outer_" + std::to_string(broadBufferCount));
            auto innerLoopVar = Var("broad_inner_" + std::to_string(broadBufferCount));
            auto tmpLoad = Load::make(preLoad->type, preLoad->buffer_var, innerLoopVar, preLoad->predicate);
            auto broadStmt =
              Store::make(tmpBuffer, tmpLoad, outerLoopVar * lastDimShape + innerLoopVar, preLoad->predicate);
            broadStmt = For::make(innerLoopVar, 0, lastDimShape, ForType::Serial, DeviceAPI::None, broadStmt);
            broadStmt = For::make(outerLoopVar, 0, remainDim, ForType::Serial, DeviceAPI::None, broadStmt);
            broadStmt = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("dma_copy"), broadStmt);
            broadStmt = Block::make(broadStmt, newStmt);
            broadStmt = Allocate::make(tmpBuffer, dtype, {fullRepeatSize}, const_true(), broadStmt);
            broadStmt = AttrStmt::make(tmpBuffer, "storage_scope", Expr("local.UB"), broadStmt);
            ResetParam();

            return broadStmt;
          }
        }
      }
    }
    ResetParam();

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (inAttr) {
      if (Equal(op->loop_var, GetItem(varList, -2)) && (newExtent > 0)) {
        auto newBody = this->Mutate(op->body);
        return For::make(op->loop_var, op->min, newExtent, op->for_type, op->device_api, newBody);
      } else if (Equal(op->loop_var, GetItem(varList, -1))) {
        auto newBody = this->Mutate(op->body);
        return For::make(op->loop_var, op->min, blkSize * 8, op->for_type, op->device_api, newBody);
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (inAttr) {
      auto newValue = this->Mutate(op->value);
      Expr newIndex = 0;
      for (size_t i = 0; i < varList.size(); ++i) {
        newIndex = varList[i] * strideList[i] + newIndex;
      }

      return Store::make(op->buffer_var, newValue, newIndex, op->predicate);
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    if (inAttr) {
      Expr load;
      if (currentSrcIndex == fullSrcIndex) {
        Expr newIndex = 0;
        for (size_t i = 0; i < varList.size(); ++i) {
          newIndex = varList[i] * strideList[i] + newIndex;
        }
        load = Load::make(op->type, op->buffer_var, newIndex, op->predicate);
      } else {
        CHECK(tmpBuffer.defined());
        preLoad = op;
        load = Load::make(op->type, tmpBuffer, op->index, op->predicate);
      }
      ++currentSrcIndex;

      return load;
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  void ResetParam() {
    inAttr = false;
    blkSize = 0;
    newExtent = 0;
    fullSrcIndex = 1;
    currentSrcIndex = 0;
    forInfo = StmtInfo();
    preLoad = nullptr;
    varList = {};
    strideList = {};
  }

  bool inAttr{false};
  int blkSize{0};
  int newExtent{0};
  int fullSrcIndex{1};
  int currentSrcIndex{0};
  int broadBufferCount{0};
  StmtInfo forInfo;
  Var tmpBuffer;
  const Load *preLoad{nullptr};
  Array<Var> varList;
  Array<Expr> strideList;
};

Stmt RewriteBroadcastVector(Stmt stmt) { return BroadcastVecRewriter().Mutate(stmt); }
}  // namespace ir
}  // namespace akg
