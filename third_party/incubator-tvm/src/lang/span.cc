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
 * \file span.cc
 * \brief The span data structure.
 */

/*
 * 2023.03.25 - Add TVM 0.8 attributes to the node and conversion pass for exporting TVM 0.8 IR.
 */

#include <tvm/expr.h>
#include <tvm/node/functor.h>
#include <tvm/runtime/registry.h>
#include <tvm/span.h>

#include <algorithm>
namespace air {

TVM_REGISTER_NODE_TYPE(SpanNode);

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
    .set_dispatch<SpanNode>([](const ObjectRef& ref, IRPrinter* p) { p->stream << "Span()"; });
} //namespace air
