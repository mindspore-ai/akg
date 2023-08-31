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
 * \file tvm/visit_attrs_util.h
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
#ifndef VISIT_ATTRS_UTIL_H_
#define VISIT_ATTRS_UTIL_H_
#include "node/container.h"
#include "string.h"
namespace air {

void VisitMap(const Map<std::string, NodeRef>& source_map, Map<air::runtime::String, NodeRef>& dist_map);

}  // namespace air
#endif