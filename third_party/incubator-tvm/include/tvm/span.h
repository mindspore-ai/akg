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
 * \file tvm/ir/span.h
 * \brief Span information for debugging purposes.
 */

/*
 * 2023.03.25 - Add TVM 0.8 attributes to the node and conversion pass for exporting TVM 0.8 IR.
 */

#ifndef TVM_IR_SPAN_H_
#define TVM_IR_SPAN_H_

#include <tvm/node/node.h>
#include <tvm/runtime/object.h>

#include "string.h"

namespace air {
/*!
 * \brief Span information for debugging purposes
 */
class Span;
/*!
 * \brief Stores locations in frontend source that generated a node.
 */
class SpanNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v) {}
  static constexpr const char* _type_key = "Span";
  TVM_DECLARE_FINAL_OBJECT_INFO(SpanNode, Object);
};

class Span : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Span, ObjectRef, SpanNode);
};
}  // namespace air
#endif