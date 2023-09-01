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
 * \file src/ir/type.cc
 * \brief Common type system AST nodes throughout the IR.
 */

/*
 * 2023.03.25 - Add TVM 0.8 attributes to the node and conversion pass for exporting TVM 0.8 IR.
 */

#include <tvm/expr.h>
#include <tvm/node/functor.h>
#include <tvm/runtime/registry.h>
#include <tvm/type.h>
namespace air {

PrimType::PrimType(DataType dtype) {
  ObjectPtr<PrimTypeNode> n = make_object<PrimTypeNode>();
  n->dtype = dtype;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PrimTypeNode);

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<PrimTypeNode>([](const ObjectRef& ref, IRPrinter* p) {
      auto* node = static_cast<const PrimTypeNode*>(ref.get());
      p->stream << node->dtype;
    });

PointerType::PointerType(TypeAnnotationType element_type, String storage_scope) {
  ObjectPtr<PointerTypeNode> n = make_object<PointerTypeNode>();
  n->element_type = std::move(element_type);
  n->storage_scope = std::move(storage_scope);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PointerTypeNode);

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<PointerTypeNode>([](const ObjectRef& ref, IRPrinter* p) {
      auto* node = static_cast<const PointerTypeNode*>(ref.get());
      if (!node->storage_scope.empty()) {
        p->stream << node->storage_scope << " ";
      }
      p->Print(node->element_type);
      p->stream << '*';
    });

}  // namespace air
