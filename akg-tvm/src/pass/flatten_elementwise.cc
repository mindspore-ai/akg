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

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/arithmetic.h>
#include <tvm.h>
#include <ir_pass.h>
#include <pass/utils.h>
#include <pass/expr_alg_simplify.h>
namespace akg {
namespace ir {
int GetConstInt(const Expr &expr) {
  CHECK(expr.as<IntImm>() || expr.as<UIntImm>());
  if (expr.as<IntImm>()) {
    return static_cast<int>(expr.as<IntImm>()->value);
  }
  return static_cast<int>(expr.as<UIntImm>()->value);
}

bool EqualArray(const Array<Expr> argsA, const Array<Expr> argsB) {
  if (argsA.size() != argsB.size()) {
    return false;
  }

  size_t i;
  for (i = 0; i < argsA.size(); i++) {
    if (!Equal(Simplify(argsA[i]), Simplify(argsB[i]))) {
      break;
    }
  }

  return (i == argsA.size());
}

Expr PlaceholderSimplify(const Expr &e) {
  if (auto op = e.as<Div>()) {
    if (auto mul = op->a.as<Mul>()) {
      if (Equal(mul->a, op->b)) return PlaceholderSimplify(mul->b);
      if (Equal(mul->b, op->b)) return PlaceholderSimplify(mul->a);
    }
  }
  if (auto op = e.as<Mul>()) {
    if (auto div = op->b.as<Div>()) {
      if (Equal(op->a, div->b)) return PlaceholderSimplify(div->a);
    }
    if (auto div = op->a.as<Div>()) {
      if (Equal(op->b, div->b)) return PlaceholderSimplify(div->a);
    }
  }
  return Simplify(e);
}

Expr ComputeSize(const Array<Expr> &shapes) {
  Expr size = 1;
  for (auto dim : shapes) {
    size = size * dim;
  }
  return PlaceholderSimplify(size);
}

class SimpleForVisitor : public IRVisitor {
 public:
  void Visit_(const For *op) override {
    if ((forDepth_ > maxDepth_) && (depthVec_.size() == static_cast<unsigned int>(forDepth_))) {
      maxDepth_ = forDepth_;
    }
    if (!depthVec_.empty() && (((forDepth_ != 0) && (forDepth_ <= depthVec_.back())) ||
                               ((forDepth_ == 0) && (depthVec_.back() != maxDepth_)))) {
      canSimplify_ = false;
    }
    depthVec_.push_back(forDepth_);
    ++forDepth_;
    IRVisitor::Visit_(op);
    forDepth_--;
  }

  std::vector<int> depthVec_;
  int forDepth_{0};
  int maxDepth_{-1};
  bool canSimplify_{true};
};

class ShapeChecker : public IRVisitor {
 public:
  explicit ShapeChecker(const Map<Tensor, Buffer> &externBuffer) {
    for (auto buffer : externBuffer) {
      if ((buffer.first->shape.size() == 1) && is_const(buffer.first->shape[0]) &&
          (GetConstInt(buffer.first->shape[0]) == 1)) {
        scalarShapeDic_[buffer.first->op->name] = buffer.first->shape;
      } else {
        bufferShapeDic_[buffer.first->op->name] = buffer.first->shape;
        Expr size = ComputeSize(buffer.first->shape);
        if (is_const(size_) && GetConstInt(size_) == 0) {
          size_ = ComputeSize(buffer.first->shape);
        } else if (!Equal(size_, size) && !Equal(ExprSimplifier().Simplify(Sub::make(size, size_)), 0)) {
          isElementwiseOp_ = false;
        }
      }
    }
  }
  ~ShapeChecker() override = default;

  void CheckDstArgs_(const std::string &name, const Array<Expr> &index) {
    if (scalarShapeDic_.count(name)) {
      isElementwiseOp_ = false;
      return;
    }

    if (bufferShapeDic_.count(name)) {
      auto shape = bufferShapeDic_[name];
      // For Tensor
      for (size_t i = 0; i < index.size(); ++i) {
        bool isDimExtent1 =
          (is_const(index[i]) && GetConstInt(index[i]) == 0 && is_const(shape[i]) && GetConstInt(shape[i]) == 1);
        bool isForIndexMeet =
          (loopExtentDic_.count(index[i].get()) > 0 && Equal(loopExtentDic_[index[i].get()], shape[i]));
        if (!(isDimExtent1 || isForIndexMeet)) {
          isElementwiseOp_ = false;
          break;
        }
      }
    }
  }

  void CheckEqualIndex() {
    Expr iIndex;
    if (visitedIndex.size() > 0) {
      iIndex = visitedIndex[0];
      for (auto tIndex : visitedIndex) {
        if (!Equal(iIndex, tIndex)) {
          isElementwiseOp_ = false;
        }
      }
    }
  }

  Array<Expr> GenReshapeIndex(const Array<Expr> &dstArgs, const Array<Expr> &dstDims, const Array<Expr> &srcDims) {
    Expr index;
    for (size_t i = 0; i < dstDims.size(); ++i) {
      if (i == 0) {
        index = dstArgs[i];
      } else {
        index = index * dstDims[i] + dstArgs[i];
      }
    }

    std::vector<Expr> srcArgs;
    CHECK_GE(srcDims.size(), 1);
    for (int i = static_cast<int>(srcDims.size()) - 1; i >= 0; --i) {
      srcArgs.push_back(floormod(index, srcDims[i]));
      index = floordiv(index, srcDims[i]);
    }
    Array<Expr> res(srcArgs.rbegin(), srcArgs.rend());
    return res;
  }

  void Visit_(const Realize *op) final {
    Array<Expr> shape;
    for (auto temp : op->bounds) {
      shape.push_back(temp->extent);
    }
    bufferShapeDic_[op->func->func_name()] = shape;
    Expr size = ComputeSize(shape);
    if (!Equal(size, 1) && !Equal(size, size_)) {
      isElementwiseOp_ = false;
    }
    if (isElementwiseOp_) {
      IRVisitor::Visit(op->body);
    }
  }

  void Visit_(const ProducerConsumer *op) final {
    if (isElementwiseOp_) {
      if (!op->is_producer) {
        IRVisitor::Visit(op->body);
      } else {
        visitedIndex = {};
        inProducer_ = true;
        checkProvide_ = false;
        IRVisitor::Visit(op->body);
        CheckEqualIndex();
        inProducer_ = false;
        if (!checkProvide_) {
          isElementwiseOp_ = false;
        } else {
          checkProvide_ = false;
        }
      }
    }
  }

  void Visit_(const For *op) final {
    loopExtentDic_[op->loop_var.get()] = op->extent;
    forVarDic_[op->loop_var] = Range::make_by_min_extent(op->min, op->extent);
    if (isElementwiseOp_) {
      IRVisitor::Visit(op->body);
    }
    forVarDic_.erase(op->loop_var);
  }

  void Visit_(const Provide *op) final {
    inProvide_ = true;
    if (op->func.defined() && bufferShapeDic_.count(op->func->func_name()) > 0) {
      CheckDstArgs_(op->func->func_name(), op->args);
      auto index = GenIndexFromArgs(op->args, bufferShapeDic_[op->func->func_name()]);
      visitedIndex.push_back(index);
      checkProvide_ = true;
      if (isElementwiseOp_) {
        if (op->func->func_name() == "reshape" && op->value.as<Call>() && op->value.as<Call>()->func.defined() &&
            bufferShapeDic_.count(op->value.as<Call>()->func->func_name()) > 0) {
          auto reshapeArgs = GenReshapeIndex(op->args, bufferShapeDic_[op->func->func_name()],
                                             bufferShapeDic_[op->value.as<Call>()->func->func_name()]);
          if (!EqualArray(op->value.as<Call>()->args, reshapeArgs)) {
            isElementwiseOp_ = false;
          }
        } else if (op->func->func_name() == "reshape_cast" && op->value.as<Cast>() &&
                   op->value.as<Cast>()->value.as<Call>() && op->value.as<Cast>()->value.as<Call>()->func.defined() &&
                   bufferShapeDic_.count(op->value.as<Cast>()->value.as<Call>()->func->func_name()) > 0) {
          auto reshapeArgs =
            GenReshapeIndex(op->args, bufferShapeDic_[op->func->func_name()],
                            bufferShapeDic_[op->value.as<Cast>()->value.as<Call>()->func->func_name()]);
          if (!EqualArray(op->value.as<Cast>()->value.as<Call>()->args, reshapeArgs)) {
            isElementwiseOp_ = false;
          }
        } else {
          IRVisitor::Visit(op->value);
        }
      }
    }
    inProvide_ = false;
  }

  void Visit_(const Call *op) final {
    if ((inProvide_ || inProducer_) && op->func.defined() && bufferShapeDic_.count(op->func->func_name())) {
      auto args = op->args;
      auto index = GenIndexFromArgs(args, bufferShapeDic_[op->func->func_name()]);
      visitedIndex.push_back(index);
    } else if (isElementwiseOp_) {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Variable *op) final { isElementwiseOp_ = false; }

  Expr GenIndexFromArgs(const Array<Expr> args, const Array<Expr> shapes) {
    CHECK_GE(args.size(), shapes.size());
    Map<Var, Range> vrange;
    for (auto iter = forVarDic_.begin(); iter != forVarDic_.end(); ++iter) {
      vrange.Set((*iter).first, (*iter).second);
    }
    Expr indexSrc = Expr(0);
    Expr coff = Expr(1);
    for (int i = static_cast<int>(args.size() - 1); i >= 0; i--) {
      indexSrc = indexSrc + args[i] * coff;
      indexSrc = Simplify(indexSrc, vrange);
      auto newIndex = Simplify(indexSrc, vrange);
      indexSrc = newIndex;
      coff = coff * shapes[i];
    }
    return Simplify(indexSrc, vrange);
  }

  bool isElementwiseOp_{true};
  bool inProvide_{false};
  bool inProducer_{false};
  bool checkProvide_{false};
  Expr size_{0};
  std::unordered_map<const Node *, Expr> loopExtentDic_;
  std::unordered_map<std::string, Array<Expr>> bufferShapeDic_;
  std::unordered_map<std::string, Array<Expr>> scalarShapeDic_;
  std::unordered_map<Var, Range, ExprHash, ExprEqual> forVarDic_;
  Array<Expr> visitedIndex;
};

class ShapeCompacter : public IRMutator {
 public:
  explicit ShapeCompacter(const Map<Tensor, Buffer> &externBuffer) {
    for (auto buffer : externBuffer) {
      funcDic_[buffer.first->op->name] = buffer.first->op;
      if (buffer.first->shape.size() == 1 && is_const(buffer.first->shape[0]) &&
          GetConstInt(buffer.first->shape[0]) == 1) {
        scalarOpDic_[buffer.first->op->name] = buffer.first->op;
      } else {
        size_ = ComputeSize(buffer.first->shape);
      }
    }
  }
  ~ShapeCompacter() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    Stmt ret = this->Mutate(op->body);
    auto temp = op->node.as<ComputeOpNode>();
    if (temp) {
      CHECK_GT(funcDic_.count(temp->name), 0);
      return AttrStmt::make(funcDic_[temp->name], op->attr_key, op->value, ret);
    } else {
      return AttrStmt::make(op->node, op->attr_key, op->value, ret);
    }
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) override {
    Stmt ret = this->Mutate(op->body);
    if (funcDic_.count(op->func->func_name())) {
      return Realize::make(funcDic_[op->func->func_name()], op->value_index, op->type, {Range(0, size_)}, op->condition,
                           ret);

    } else {
      return ret;
    }
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) override {
    Stmt ret = this->Mutate(op->body);
    CHECK_GT(funcDic_.count(op->func->func_name()), 0);
    return ProducerConsumer::make(funcDic_[op->func->func_name()], op->is_producer, ret);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    ++forDepth_;
    auto ret = this->Mutate(op->body);
    forDepth_--;
    if (forDepth_ != 0) {
      return ret;
    } else {
      ret = For::make(newVar_, 0, size_, ForType::Serial, DeviceAPI::None, ret);
      newVar_ = Var("cc" + std::to_string(++varIndex_));
      return ret;
    }
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto value = this->Mutate(op->value);
    FunctionRef newFunc;
    if (!funcDic_.count(op->func->func_name())) {
      if (op->func.as<ComputeOpNode>()) {
        auto oldOp = op->func.as<ComputeOpNode>();
        CHECK(!oldOp->axis.empty());
        auto newIter =
          IterVarNode::make(Range(Expr(0), size_), newVar_, oldOp->axis[0]->iter_type, oldOp->axis[0]->thread_tag);
        auto newOp = ComputeOpNode::make(oldOp->name, oldOp->tag, oldOp->attrs, {newIter}, {value});
        funcDic_[oldOp->name] = newOp;
      } else if (op->func.as<PlaceholderOpNode>()) {
        auto oldOp = op->func.as<PlaceholderOpNode>();
        auto newOp = PlaceholderOpNode::make(oldOp->name, {size_}, oldOp->dtype);
        funcDic_[oldOp->name] = newOp;
      } else {
        LOG(FATAL) << "Can't deal with the type of this op func: " << op->func->func_name();
      }
    }

    newFunc = funcDic_[op->func->func_name()];
    return Provide::make(newFunc, op->value_index, value, {newVar_});
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->func) {
      FunctionRef newFunc;
      CHECK_GT(funcDic_.count(op->name), 0);
      newFunc = funcDic_[op->name];
      Array<Expr> new_args;
      if (scalarOpDic_.count(op->name)) {
        new_args.push_back(0);
      } else {
        new_args.push_back(newVar_);
      }
      return Call::make(op->type, op->name, new_args, op->call_type, newFunc);
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

 private:
  Var newVar_{Var("cc0")};
  int varIndex_{0};
  Expr size_{0};
  int forDepth_{0};
  std::unordered_map<std::string, Operation> scalarOpDic_;
  std::unordered_map<std::string, FunctionRef> funcDic_;
};

Stmt ElementwiseFlatten(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer,
                        const Map<Tensor, Buffer> &new_extern_buffer) {
  auto simpleFor = SimpleForVisitor();
  simpleFor.Visit(stmt);
  const int MinimumForLevel = 0;
  const int MinimumSize = 2;
  if (simpleFor.canSimplify_ && simpleFor.maxDepth_ >= MinimumForLevel) {
    auto shapeChecker = ShapeChecker(extern_buffer);
    if (shapeChecker.isElementwiseOp_) {
      shapeChecker.Visit(stmt);
    }
    if (shapeChecker.isElementwiseOp_ &&
        (!is_const(shapeChecker.size_) || GetConstInt(shapeChecker.size_) >= MinimumSize)) {
      auto irCompacter = ShapeCompacter(new_extern_buffer);
      stmt = irCompacter.Mutate(stmt);
      return stmt;
    }
  }

  return stmt;
}
}  // namespace ir
}  // namespace akg
