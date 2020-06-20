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
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/ir.h>
#include <tvm/tensor.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>

#include "ir_pass.h"
#include "pass/ir_util.h"
#include "pass/utils.h"

/*
 * This pass should be used before StorageFlatten.
 *
 * Example before this pass:

  realize output([0, 16], [0, 16]) {
    for (cc0, 0, 15) {
      for (cc1, 0, 15) {
        output(cc0, cc1) = input(cc0, cc1);
      }
    }
  }

  After this pass:

  realize output([0, 16], [0, 16]) {
    for (cc0, 0, 15) {
      for (cc1, 0, 16) {
        output(cc0, cc1) = select(cc1 < 15, input(cc0, cc1), output(cc0, cc1));
      }
    }
  }
 */

namespace akg {
namespace ir {
using TensorName = std::string;
using IteratorName = std::string;

const int loopExtentRequiredAlignment = 16;

class AlignLastAxisLoopExtentMutator : public IRMutator {
 public:
  Stmt Run(const Stmt stmt, const Map<Tensor, Buffer> &externBuffer) {
    realizeShape.clear();
    for (auto buffer : externBuffer) {
      realizeShape[buffer.first->op->name] = buffer.second->shape;
    }
    return Mutate(stmt);
  }

 private:
  template <class T>
  void FindIteratorsInLastAxis(const T *op) {
    CHECK(op);
    TensorName name = op->func->func_name();
    if (realizeShape.count(name) && op->args.size() >= 1) {
      CHECK(realizeShape[name].size() == op->args.size());
      int lastArgIndex = static_cast<int>(op->args.size()) - 1;
      Expr extent = realizeShape[name][lastArgIndex];
      if (extent.as<IntImm>()) {
        int realizeBound = static_cast<int>(extent.as<IntImm>()->value);
        Expr lastArg = op->args[lastArgIndex];

        PostOrderVisit(lastArg, [&, this, realizeBound](const NodeRef &node) {
          if (node.as<Variable>()) {
            IteratorName varName = node.as<Variable>()->name_hint;
            if (unalignedIterators.count(varName) > 0) {
              bool alreadyFound = iteratorFoundInLastAxis[varName];
              iteratorFoundInLastAxis[varName] = true;
              int minRealizeShape = iteratorAlignedExtent[varName];
              bool isRealizeShapeTooSmall = realizeBound < minRealizeShape;
              if (isRealizeShapeTooSmall) {
                if (alreadyFound && iteratorFoundInvalidRealizeShape.count(varName) > 0 &&
                    !iteratorFoundInvalidRealizeShape[varName]) {
                  iteratorNeedRetry[varName] = true;
                }
                iteratorFoundInvalidRealizeShape[varName] = true;
              } else {
                iteratorsToFixInProvide.insert(varName);
              }
            }
          }
        });
      }
    }
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) override {
    TensorName name = op->func->func_name();
    // save realize shape in outer scope
    bool outerRealizeExist = false;
    Array<Expr> outerRealizeShape;
    if (realizeShape.count(name) > 0) {
      outerRealizeExist = true;
      outerRealizeShape = realizeShape[name];
    }

    Array<Expr> shape;
    std::transform(op->bounds.begin(), op->bounds.end(), std::back_inserter(shape.CopyOnWrite()->data),
                   [](const Range &bound) { return (bound->extent); });
    realizeShape[name] = shape;

    Stmt stmt = IRMutator::Mutate_(op, s);

    // restore realize shape in outer scope
    realizeShape.erase(name);
    if (outerRealizeExist) {
      realizeShape[name] = outerRealizeShape;
    }

    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) override {
    if (op->call_type == Call::CallType::Halide) {
      FindIteratorsInLastAxis(op);
    }

    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    iteratorsToFixInProvide.clear();

    FindIteratorsInLastAxis(op);
    Expr newValue = IRMutator::Mutate(op->value);

    Expr joinedCondition;
    bool isFirst = true;
    for (auto it : iteratorsToFixInProvide) {
      if (iteratorFoundInvalidRealizeShape.count(it) > 0 && iteratorFoundInvalidRealizeShape[it]) {
        continue;
      }
      CHECK_GT(iteratorVarRef.count(it), 0);
      Expr condition = iteratorVarRef[it] < Expr(unalignedIterators[it]);
      if (isFirst) {
        joinedCondition = condition;
        isFirst = false;
      } else {
        joinedCondition = joinedCondition && condition;
      }
    }

    if (isFirst) {
      return Provide::make(op->func, op->value_index, newValue, op->args);
    } else {
      Stmt new_provide = Provide::make(op->func, op->value_index, newValue, op->args);
      return IfThenElse::make(joinedCondition, new_provide, Stmt());
    }
  }

  static int CeilAlign(const int a, const int b) {
    CHECK_GE(a, 0);
    CHECK_GE(b, 1);
    return (a + b - 1) / b * b;
  }

  Stmt Mutate_(const For *op, const Stmt &s) override {
    if (is_const(op->extent) && is_const(op->min)) {
      CHECK(op->extent.as<IntImm>());
      CHECK(op->min.as<IntImm>());
      int extent = static_cast<int>(op->extent.as<IntImm>()->value + op->min.as<IntImm>()->value);
      if (extent % loopExtentRequiredAlignment != 0) {
        IteratorName name = op->loop_var->name_hint;
        unalignedIterators[name] = extent;
        int alignedExtent = CeilAlign(extent, loopExtentRequiredAlignment);
        iteratorAlignedExtent[name] = alignedExtent;
        iteratorVarRef[name] = op->loop_var;
        iteratorFoundInLastAxis[name] = false;
        iteratorFoundInvalidRealizeShape[name] = false;
        iteratorNeedRetry[name] = false;

        Stmt stmt = IRMutator::Mutate_(op, s);
        const For *newOp = stmt.as<For>();

        unalignedIterators.erase(name);
        iteratorAlignedExtent.erase(name);
        iteratorVarRef.erase(name);
        bool foundInLastAxis = iteratorFoundInLastAxis[name];
        iteratorFoundInLastAxis.erase(name);
        bool foundInvalidRealizeShape = iteratorFoundInvalidRealizeShape[name];
        iteratorFoundInvalidRealizeShape.erase(name);
        bool needRetry = iteratorNeedRetry[name];
        iteratorNeedRetry.erase(name);

        if (foundInLastAxis) {
          if (!foundInvalidRealizeShape) {
            int loopExtent = alignedExtent - static_cast<int>(op->min.as<IntImm>()->value);
            CHECK(newOp);
            return For::make(op->loop_var, op->min, Expr(loopExtent), op->for_type, op->device_api, newOp->body);
          } else if (needRetry) {
            // this iterator cannot be aligned, but some if condition has been added.
            // so we need to retry without this iterator.
            return IRMutator::Mutate_(op, s);
          } else {
            // this iterator cannot be aligned, but no if condition has been added, so we can return the loop body
            return stmt;
          }
        } else {
          return stmt;
        }
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  std::unordered_map<TensorName, Array<Expr>> realizeShape;
  std::unordered_map<IteratorName, int> unalignedIterators;
  std::unordered_map<IteratorName, int> iteratorAlignedExtent;
  std::unordered_map<IteratorName, VarExpr> iteratorVarRef;
  std::unordered_map<IteratorName, bool> iteratorFoundInLastAxis;
  std::unordered_map<IteratorName, bool> iteratorFoundInvalidRealizeShape;
  std::unordered_map<IteratorName, bool> iteratorNeedRetry;
  std::unordered_set<IteratorName> iteratorsToFixInProvide;
};

Stmt AlignLastAxisLoopExtent(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer) {
  stmt = AlignLastAxisLoopExtentMutator().Run(stmt, extern_buffer);
  return stmt;
}
}  // namespace ir
}  // namespace akg
