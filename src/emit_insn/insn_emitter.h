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

#ifndef EMIT_INSN_INSN_EMITTER_H_
#define EMIT_INSN_INSN_EMITTER_H_

#include <tvm/ir.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "tvm.h"
#include "pass/storage_access.h"
#include "emit_insn/insn_info.h"

namespace akg {
namespace ir {

Stmt EmitInsnWithDynamicShapes(const Stmt &s, const Map<Tensor, Buffer> &extern_buffer);

Stmt InsnEmit(std::string insnName, const Stmt &op, bool enableBisect, bool enableCoverProtect, int commentLevel);

Stmt MadEmitter(const Stmt &op);

Stmt Im2ColEmitter(const Stmt &op, const std::unordered_map<std::string, ObjectRef> &attrs, const Buffer &src,
                   bool is_dynamic);

Stmt Im2ColEmitterL1UB(const Stmt &op, const std::unordered_map<std::string, ObjectRef> &attrs, const Buffer &src,
                       bool is_dynamic);

std::vector<size_t> SortIndexes(const std::vector<int> &v);

template <typename T>
Buffer MakeBuf(const T *mem, const Type &t, const StmtInfo &forInfo) {
  CHECK(mem);
  bool isLegalStrides = true;
  // strides
  Array<Var> vars;
  std::copy(forInfo.vars_.begin(), forInfo.vars_.end(), std::back_inserter(vars.CopyOnWrite()->data));

  auto loop_var_size = forInfo.ops_.size();
  // shape
  Array<Expr> shape;
  Array<Expr> mem_strides = ktvm::arith::DetectLinearEquation(mem->index, vars);
  if (mem_strides.empty()) {
    isLegalStrides = false;
    mem_strides.push_back(make_const(Int(32), 1));
    mem_strides.push_back(make_const(Int(32), 0));
    shape.push_back(make_const(Int(32), 1));
    loop_var_size = 1;
  }

  if (isLegalStrides) {
    if (loop_var_size == 0) {
      shape.push_back(make_const(Int(32), 1));
    } else {
      for (auto op : forInfo.ops_) {
        auto forOp = op.as<For>();
        CHECK(forOp);
        if (ExprUseVar(mem->index, forOp->loop_var)) {
          if (forOp->extent.as<IntImm>()) {
            shape.push_back(forOp->extent);
          } else {
            shape.push_back(make_const(Int(32), 1));
          }
        } else {
          shape.push_back(make_const(Int(32), 0));
        }
      }
    }
  }

  CHECK(!mem_strides.empty()) << "May get non-linear expr in DetectLinearEquation";
  Array<Expr> strides(mem_strides.begin(), mem_strides.begin() + loop_var_size);

  std::vector<int> istrides;
  for (auto i : strides) {
    if (i.as<IntImm>()) {
      istrides.push_back(GetInt32Const(i));
    }
  }

  auto idx = SortIndexes(istrides);
  Array<Expr> fshape;
  Array<Expr> fstrides;

  for (auto i : idx) {
    if (is_zero(mem->index)) {
      fstrides.push_back(make_const(Int(32), 1));
    } else {
      fstrides.push_back(strides[i]);
    }

    // if shape[i] == 0, then do not push back
    if (const auto op = shape[i].as<IntImm>()) {
      if (op->value == 0) {
        continue;
      }
    }
    fshape.push_back(shape[i]);
  }

  // make sure no empty shape for codegen
  if (fshape.empty()) {
    fshape.push_back(make_const(Int(32), 1));
  }

  if (fstrides.empty()) {
    fstrides.push_back(make_const(Int(32), 1));
  }

  Buffer buf =
    BufferNode::make(mem->buffer_var, t, fshape, fstrides, mem_strides[mem_strides.size() - 1],
                     mem->buffer_var->name_hint, GetBufScope(mem->buffer_var->name_hint), 0, 0, BufferType::kDefault);
  return buf;
}
}  // namespace ir
}  // namespace akg
#endif  // EMIT_INSN_INSN_EMITTER_H_
