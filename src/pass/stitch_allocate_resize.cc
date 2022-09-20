/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file elim_duplicate_inputs.cc
 */
#include "tvm/ir.h"
#include "tvm/ir_mutator.h"

namespace air {
namespace ir {
class StitchGPUAllocResizer : public IRMutator {
 public:
  explicit StitchGPUAllocResizer(const Map<std::string, NodeRef> gpu_stitch_buf_alloc_size) {
    for (auto& kv : gpu_stitch_buf_alloc_size) {
      gpu_stitch_buf_alloc_size_[kv.first] = kv.second.as<IntImm>()->value;
    }
  }
  Stmt Run(Stmt& stmt) { return Mutate(stmt); };

 private:
  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    std::string buf_name = op->buffer_var->name_hint;
    if (gpu_stitch_buf_alloc_size_.count(buf_name)) {
      int32_t sh = static_cast<int32_t>(gpu_stitch_buf_alloc_size_[buf_name] / op->type.bytes());
      Array<Expr> new_extent({Expr(sh)});
      gpu_stitch_buf_alloc_size_.erase(buf_name);
      Stmt stmt = Allocate::make(op->buffer_var, op->type, new_extent, op->condition, op->body,
                                 op->new_expr, op->free_function);
      return this->Mutate(stmt);
    }
    return IRMutator::Mutate_(op, s);
  }
  std::unordered_map<std::string, int32_t> gpu_stitch_buf_alloc_size_;
};

Stmt StitchGpuAllocResize(Stmt stmt, Map<std::string, NodeRef> gpu_stitch_buf_alloc_size) {
  return StitchGPUAllocResizer(gpu_stitch_buf_alloc_size).Run(stmt);
}
}  // namespace ir
}  // namespace air