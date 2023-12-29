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
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <tvm.h>
#include <cmath>

namespace akg {
namespace ir {
class ReduceSumRewriter : public IRMutator {
 public:
  explicit ReduceSumRewriter(const Map<Tensor, Buffer> &externBuffer) {
    for (auto buffer : externBuffer) {
      bufferShapeDic_[buffer.first->op->name] = buffer.first->shape;
    }
  }
  ~ReduceSumRewriter() override = default;

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    Array<Expr> index;
    for (auto temp : op->bounds) {
      index.push_back(temp->extent);
    }
    bufferShapeDic_[op->func->func_name()] = index;
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == air::ir::attr::reduce_update) {
      Array<IterVar> iter = Downcast<Array<IterVar>>(op->node);
      const auto pro = op->body.as<Provide>();
      int64_t size = 1;
      const int64_t THRESHOLD = 2;
      for (auto iterVar : iter) {
        if (iterVar->dom->extent.as<IntImm>()) {
          size = size * iterVar->dom->extent.as<IntImm>()->value;
        } else {
          size = 0;
        }
      }

      if (pro && pro->value.as<Add>() && size > THRESHOLD && hasZeroInit_) {
        isReduceSum_ = true;
        Stmt ret = IRMutator::Mutate_(op, s);
        isReduceSum_ = false;
        return ret;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    loopVarExtendDic_[op->loop_var.get()] = op->extent;
    forOrderRec_.push_back(op);
    auto ret = IRMutator::Mutate_(op, s);
    return ret;
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    loopVarExtendDic_.clear();
    forOrderRec_.clear();
    hasZeroInit_ = false;
    pProCon = op;
    auto body = Mutate(op->body);
    if (needMutate_ && pSrcFp16 && bufferShapeDic_.count(pSrcFp16->name) > 0 && pDstFp16 &&
        bufferShapeDic_.count(pDstFp16->name) > 0 && op->func == pDstFp16->func) {
      // Add Cast for fp16 src; Reduce Add in fp32; Cast fp32 res to fp16 dst
      srcFp32 = PlaceholderOpNode::make("fp32_" + pSrcFp16->name, bufferShapeDic_[pSrcFp16->name], Float(32));
      dstFp32 = PlaceholderOpNode::make("fp32_" + pDstFp16->name, bufferShapeDic_[pDstFp16->name], Float(32));
      cast_from_fp32_.insert(srcFp32);
      cast_to_fp32_.insert(dstFp32);

      Stmt srcFp16toFP32 = TensorCastFp16ToFp32();
      Stmt srcFp32Producer = ProducerConsumer::make(srcFp32, true, srcFp16toFP32);

      // Mutate the body
      mutateFlag_ = true;
      Stmt body_mutate = Mutate(op->body);
      Stmt dstFp32Producer = ProducerConsumer::make(dstFp32, true, body_mutate);
      mutateFlag_ = false;

      // For Loops for Cast Fp32 to Fp16
      Stmt dstFp32toFp16 = TensorCastFp32ToFp16();
      auto dstFp16Producer = ProducerConsumer::make(op->func, op->is_producer, dstFp32toFp16);

      Stmt block0 = Block::make(srcFp32Producer, dstFp32Producer);
      Stmt block1 = Block::make(block0, dstFp16Producer);

      // Two Realize
      Region srcRegion = MakeRegionFromShape(bufferShapeDic_[pSrcFp16->name]);
      Region dstRegion = MakeRegionFromShape(bufferShapeDic_[pDstFp16->name]);
      Stmt realizeSrcFp32 = Realize::make(srcFp32, 0, Float(32), srcRegion, const_true(1), block1);
      Stmt realizeDstFp32 = Realize::make(dstFp32, 0, Float(32), dstRegion, const_true(1), realizeSrcFp32);
      needMutate_ = false;
      return realizeDstFp32;
    }
    needMutate_ = false;
    return ProducerConsumer::make(op->func, op->is_producer, body);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (isReduceSum_ && op->value.as<Add>() && op->value.type() == Float(16)) {
      auto add = op->value.as<Add>();
      if (add->a.as<Call>() && add->b.as<Call>()) {
        auto pSrc0 = add->a.as<Call>();
        auto pSrc1 = add->b.as<Call>();
        if (op->func == pSrc0->func && EqualArray(op->args, pSrc0->args)) {
          pDstFp16 = pSrc0;
          pSrcFp16 = pSrc1;
          if (mutateFlag_) {
            Expr src0Fp32 =
              Call::make(Float(32), "fp32_" + pSrc0->name, pSrc0->args, pSrc0->call_type, dstFp32, pSrc0->value_index);
            Expr src1Fp32 =
              Call::make(Float(32), "fp32_" + pSrc1->name, pSrc1->args, pSrc1->call_type, srcFp32, pSrc1->value_index);
            Expr addFp32 = Add::make(src0Fp32, src1Fp32);
            Stmt resFp32 = Provide::make(dstFp32, op->value_index, addFp32, op->args);
            return resFp32;
          } else {
            needMutate_ = true;
            return s;
          }
        }
      }
    } else if (pProCon && op->func == pProCon->func && op->value.as<FloatImm>() &&
               (op->value.as<FloatImm>()->value == 0)) {
      if (mutateFlag_) {
        Stmt initFp32 = Provide::make(dstFp32, op->value_index, FloatImm::make(Float(32), 0.0), op->args);
        return initFp32;
      } else {
        hasZeroInit_ = true;
        return s;
      }
    }
    return s;
  }

  bool EqualArray(const Array<Expr> &a, const Array<Expr> &b) {
    bool res = false;
    if (a.size() == b.size()) {
      res = true;
      for (size_t i = 0; i < a.size(); ++i) {
        res = res && Equal(a[i], b[i]);
      }
    }
    return res;
  }

  Region MakeRegionFromShape(Array<Expr> &shape) {
    Region res;
    for (auto iter = shape.begin(); iter != shape.end(); ++iter) {
      res.push_back(Range::make_by_min_extent(Expr(0), *iter));
    }
    return res;
  }

  Stmt TensorCastFp16ToFp32() {
    CHECK(pSrcFp16);
    CHECK(pDstFp16);
    auto iSrcCall = Call::make(Float(16), pSrcFp16->name, pSrcFp16->args, Call::CallType::Halide, pSrcFp16->func, 0);
    auto iCast = Cast::make(Float(32), iSrcCall);
    auto iPro = Provide::make(srcFp32, 0, iCast, pSrcFp16->args);
    Stmt body = iPro;

    std::vector<const Variable *> argsVar;
    auto _hasVariable = [&argsVar, this](const NodeRef &op) {
      if (op->IsInstance<Variable>() && std::find(argsVar.begin(), argsVar.end(), op.as<Variable>()) == argsVar.end()) {
        argsVar.push_back(op.as<Variable>());
      }
    };
    for (auto dim : pSrcFp16->args) {
      PostOrderVisit(dim, _hasVariable);
    }
    for (auto iter = forOrderRec_.rbegin(); iter != forOrderRec_.rend(); ++iter) {
      auto index = (*iter)->loop_var;
      if (index.as<Variable>() && std::find(argsVar.begin(), argsVar.end(), index.as<Variable>()) != argsVar.end()) {
        const auto temp = index.as<Variable>();
        CHECK_GT(loopVarExtendDic_.count(temp), 0);
        body = For::make(Var(temp->name_hint), 0, loopVarExtendDic_[temp], ForType::Serial, DeviceAPI::None, body);
      }
    }

    return body;
  }

  Stmt TensorCastFp32ToFp16() {
    CHECK(pDstFp16);
    auto iSrcCall = Call::make(Float(32), dstFp32->func_name(), pDstFp16->args, Call::CallType::Halide, dstFp32, 0);
    auto iCast = Cast::make(Float(16), iSrcCall);
    auto iPro = Provide::make(pDstFp16->func, 0, iCast, pDstFp16->args);
    Stmt body = iPro;
    for (auto iter = pDstFp16->args.rbegin(); iter != pDstFp16->args.rend(); ++iter) {
      auto index = (*iter);
      if (index.as<Variable>()) {
        const auto temp = index.as<Variable>();
        CHECK_GT(loopVarExtendDic_.count(temp), 0);
        body = For::make(Var(temp->name_hint), 0, loopVarExtendDic_[temp], ForType::Serial, DeviceAPI::None, body);
      }
    }
    return body;
  }

  std::unordered_set<FunctionRef, NodeHash, NodeEqual> cast_from_fp32_;
  std::unordered_set<FunctionRef, NodeHash, NodeEqual> cast_to_fp32_;

 private:
  bool isReduceSum_{false};
  bool hasZeroInit_{false};
  bool needMutate_{false};
  bool mutateFlag_{false};
  const Call *pSrcFp16{nullptr};
  const Call *pDstFp16{nullptr};
  const ProducerConsumer *pProCon{nullptr};
  Operation srcFp32;
  Operation dstFp32;
  std::unordered_map<std::string, Array<Expr>> bufferShapeDic_;
  std::unordered_map<const Variable *, Expr> loopVarExtendDic_;
  std::vector<const For *> forOrderRec_;
};

class RedundantCastFinder : public IRVisitor {
 public:
  RedundantCastFinder(const std::unordered_set<FunctionRef, NodeHash, NodeEqual> &cast_from_fp32,
                      const std::unordered_set<FunctionRef, NodeHash, NodeEqual> &cast_to_fp32,
                      const Map<Tensor, Buffer> &extern_buffer)
      : cast_from_fp32_(cast_from_fp32), cast_to_fp32_(cast_to_fp32), extern_buffer_(extern_buffer) {}
  ~RedundantCastFinder() override = default;

  void Visit_(const Provide *op) final {
    auto cast = op->value.as<Cast>();
    if (cast != nullptr && cast->value.as<Call>()) {
      auto func = op->func;
      all_cast_[func] = cast;
      if (cast_count_.count(func) == 0) {
        cast_count_[func] = 1;
      } else {
        cast_count_[func]++;
      }
    }
  }

  void FindRedundantCast() {
    for (auto &it : all_cast_) {
      if (cast_count_[it.first] > 1) {
        all_cast_.erase(it.first);
      }
    }

    // Find fp32 -> fp16 -> fp32 before reduce sum.
    // Here, the fp16 and the second fp32 can be eliminated,
    // and the first fp32 can be directly used as input of reduce sum.
    for (auto &it : cast_from_fp32_) {
      FunctionRef cast_tmp;
      if (all_cast_.count(it) != 0) {
        auto call = all_cast_[it]->value.as<Call>();
        CHECK(call);
        cast_tmp = call->func;
        if (all_cast_.count(cast_tmp) != 0) {
          call = all_cast_[cast_tmp]->value.as<Call>();
          if (call != nullptr && call->type == Float(32)) {
            copy_[it] = call->func;
            copy_[cast_tmp] = FunctionRef();
          }
        }
      }
    }

    // Find fp32 -> fp16 -> fp32 after reduce sum.
    // Here, the fp16 and the second fp32 can be eliminated,
    // and the first fp32(output of reduce sum) can be directly used later.
    for (auto &it : cast_to_fp32_) {
      FunctionRef cast_tmp;
      for (auto &ele : all_cast_) {
        CHECK(ele.second);
        auto call = ele.second->value.as<Call>();
        if (call != nullptr && call->func == it) {
          cast_tmp = ele.first;
          break;
        }
      }
      if (!cast_tmp.defined()) {
        continue;
      }
      for (auto &ele : all_cast_) {
        CHECK(ele.second);
        auto call = ele.second->value.as<Call>();
        if (call != nullptr && call->func == cast_tmp && ele.second->type == Float(32)) {
          copy_[cast_tmp] = it;
          break;
        }
      }
    }

    // Do not eliminate real outputs
    for (const auto &it : extern_buffer_) {
      FunctionRef func = it.first->op;
      while (copy_.count(func) != 0) {
        auto value = copy_[func];
        copy_.erase(func);
        func = value;
      }
    }
  }

  std::unordered_map<FunctionRef, FunctionRef, NodeHash, NodeEqual> copy_;

 private:
  std::unordered_set<FunctionRef, NodeHash, NodeEqual> cast_from_fp32_;
  std::unordered_set<FunctionRef, NodeHash, NodeEqual> cast_to_fp32_;
  std::unordered_map<FunctionRef, const Cast *, NodeHash, NodeEqual> all_cast_;
  std::unordered_map<FunctionRef, int, NodeHash, NodeEqual> cast_count_;
  Map<Tensor, Buffer> extern_buffer_;
};

class RemoveRedundantCast : public IRMutator {
 public:
  explicit RemoveRedundantCast(const std::unordered_map<FunctionRef, FunctionRef, NodeHash, NodeEqual> &copy)
      : copy_(copy) {}
  ~RemoveRedundantCast() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    auto node = op->node.as<OperationNode>();
    if (node != nullptr) {
      for (auto &it : copy_) {
        if (node->name == it.first->func_name() && op->node.get() == it.first.get()) {
          return Mutate(op->body);
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    if (copy_.count(op->func) != 0) {
      return Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    if (op->is_producer && copy_.count(op->func) != 0) {
      return Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (copy_.count(op->func) != 0) {
      return Evaluate::make(0);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &s) final {
    if (copy_.count(op->func) != 0) {
      auto new_func = copy_[op->func];
      if (new_func.defined()) {
        return Call::make(op->type, new_func->func_name(), op->args, op->call_type, new_func, op->value_index);
      }
    }
    return s;
  }

  Expr Mutate_(const Cast *op, const Expr &s) final {
    auto call = op->value.as<Call>();
    if (call != nullptr && copy_.count(call->func) != 0) {
      auto new_func = copy_[call->func];
      if (new_func.defined()) {
        return Call::make(Float(32), new_func->func_name(), call->args, call->call_type, new_func, call->value_index);
      }
    }
    return s;
  }

 private:
  std::unordered_map<FunctionRef, FunctionRef, NodeHash, NodeEqual> copy_;
};

Stmt HalfReduceSumRewrite(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer) {
  ReduceSumRewriter rewriter(extern_buffer);
  stmt = rewriter.Mutate(stmt);

  if (rewriter.cast_from_fp32_.size() == 0 && rewriter.cast_to_fp32_.size() == 0) {
    return stmt;
  }
  RedundantCastFinder finder(rewriter.cast_from_fp32_, rewriter.cast_to_fp32_, extern_buffer);
  finder.Visit(stmt);
  finder.FindRedundantCast();

  if (finder.copy_.size() == 0) {
    return stmt;
  }
  RemoveRedundantCast remover(finder.copy_);
  stmt = RemoveNoOp(remover.Mutate(stmt));

  return stmt;
}
}  // namespace ir
}  // namespace akg
