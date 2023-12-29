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
 * \file tvm/ir/type.h
 * \brief IR/AST nodes for the unified type system in TVM.
 *
 * We use Relay's type system as the unified type system
 * throughout the stack.
 *
 * This file contains types that are common across IR variants.
 *
 * ## Relation between Type and runtime::DataType
 *
 * Besides Type, we also store a dtype field in the low-level PrimExpr.
 * runtime::DataType(dtype) provides coarse grained type information
 * during compile time and runtime. It is eagerly built in
 * low-level expression construction and can be used for
 * quick type checking in the low-level IR.
 * For example, when an Expr's dtype is int32,
 * we know for sure that its type is also int32.
 *
 * On the other hand, Type provides more fine grained information.
 * For example, a low level expression can have DataType::Handle() as
 * its dtype and MemRef[float32] as its type.
 * Types are usually lazily constructed via type checking,
 * so they may not readily be available during IR construction.
 *
 * The unified Type serves as a common bridge across IR dialects.
 * For example, we require all the functions to have a type signature,
 * which allow us to build cross dialect function calls.
 */

/*
 * 2023.03.25 - Add TVM 0.8 attributes to the node and conversion pass for exporting TVM 0.8 IR.
 */

#ifndef TVM_IR_TYPE_H_
#define TVM_IR_TYPE_H_

#include <tvm/dtype.h>
#include <tvm/node/container.h>
#include <tvm/node/node.h>
#include <tvm/runtime/object.h>
#include <tvm/version_info.h>

#include "span.h"
#include "string.h"

namespace air {

using air::runtime::String;

/*!
 * \brief Type is the base type of all types.
 *
 * Relay's type system contains following subclasses:
 *
 * - PrimType: type of primitive type values used in the low-level IR.
 * - FuncType: type of a function.
 * - TensorType: type of certain Tensor values in the expression.
 *
 * There are also advanced types to support generic(polymorphic types).
 * \sa Type
 */
class TypeNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  static constexpr const char* _type_key = "Type";
  TVM_DECLARE_BASE_OBJECT_INFO(TypeNode, Object);
};

/*!
 * \brief Managed reference to TypeNode.
 * \sa TypeNode
 */
class TypeAnnotationType : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TypeAnnotationType, ObjectRef, TypeNode);
};

/*!
 * \brief Primitive data types used in the low-level IR.
 *
 * PrimType represents POD-values and handles that are
 * not automatically managed by the runtime.
 *
 * \sa PrimType
 */
class PrimTypeNode : public TypeNode {
 public:
  /*!
   * \brief The corresponding dtype field.
   */
  DataType dtype;

  void VisitAttrs(AttrVisitor* v) { v->Visit("dtype", &dtype); }

  static constexpr const char* _type_key = "PrimType";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimTypeNode, TypeNode);
};

/*
 * \brief Managed reference to PrimTypeNode.
 * \sa PrimTypeNode
 */
class PrimType : public TypeAnnotationType {
 public:
  /*!
   * \brief Constructor
   * \param dtype The corresponding dtype.
   */
  TVM_DLL explicit PrimType(DataType dtype);

  TVM_DEFINE_OBJECT_REF_METHODS(PrimType, TypeAnnotationType, PrimTypeNode);
};

/*!
 * \brief Low-level raw pointer type.
 *
 *  PointerType represents type hints in the TIR to be
 *  passed to the final code generator.
 *
 *  PointerType should not occur in the high-level analysis.
 *
 * \sa PointerType
 */
class PointerTypeNode : public TypeNode {
 public:
  /*!
   * \brief The type of the element which the pointer points to.
   */
  TypeAnnotationType element_type;
  /*!
   * \brief The storage scope of the pointer
   */
  String storage_scope;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("element_type", &element_type);
    v->Visit("storage_scope", &storage_scope);
  }

  static constexpr const char* _type_key = "PointerType";
  TVM_DECLARE_FINAL_OBJECT_INFO(PointerTypeNode, TypeNode);
};

/*
 * \brief Managed reference to PointerTypeNode.
 * \sa PointerTypeNode
 */
class PointerType : public TypeAnnotationType {
 public:
  /*!
   * \brief Constructor
   * \param element_type The type of the element which the pointer points to.
   * \param storage_scope The storage scope into which the pointer addresses
   */
  TVM_DLL explicit PointerType(TypeAnnotationType element_type, String storage_scope = "");

  TVM_DEFINE_OBJECT_REF_METHODS(PointerType, TypeAnnotationType, PointerTypeNode);
};

}  // namespace air
#endif  // TVM_IR_TYPE_H_
