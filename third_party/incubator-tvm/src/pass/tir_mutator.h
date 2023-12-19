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
 * \file tir_mutator.h
 * \brief Visit all nodes in IR and add tvm 0.8 attributes to the nodes.
 */

/*
 * 2023.12.19 - Add case where index calculation exceeds int32.
 * 2023.03.25 - Add TVM 0.8 attributes to the node and conversion pass for exporting TVM 0.8 IR.
 */

#ifndef TIR_MUTATOR_H
#define TIR_MUTATOR_H
#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_visitor.h>
#include <tvm/lowered_func.h>
#include <tvm/operation.h>
#include <tvm/tensor.h>

#include <string>
#include <unordered_map>
#include <vector>
namespace air {
namespace ir {

class IRConverter : public IRVisitor {
 public:
  void Visit_(const Allocate* op);
  void Visit_(const AttrStmt* op);
  void Visit_(const Block* op);
  void Visit_(const Call* op);
  void Visit_(const Load* op);
  void Visit_(const Store* op);
  void Visit_(const Provide* op);
  void Visit_(const Prefetch* op);
  void Visit_(const Realize* op);

  void Visit_(const FunctionRef& op);
  void Visit_(const ComputeOpNode* op);
  void Visit_(const ExternOpNode* op);
  void Visit_(const HybridOpNode* op);
  void Visit_(const PlaceholderOpNode* op);
  void Visit_(const ScanOpNode* op);
  void Visit_(const TensorComputeOpNode* op);

  void Visit_(const BufferNode* op);
  void Visit_(const IterVarNode* op);
  void Visit_(const RangeNode* op);
  void Visit_(const TensorNode* op);
  void Visit_(const TensorIntrinNode* op);

  void Visit_(const ObjectRef& op);
  void Visit_(const IntImm* op);

 private:
  void FlattenBlock(const Stmt& s, std::vector<Stmt>* res);
};
void IR_Conversion(const ObjectRef& f);
}  // namespace ir
}  // namespace air
#endif