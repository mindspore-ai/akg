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

#ifndef INCLUDE_AKG_TVM_H_
#define INCLUDE_AKG_TVM_H_

#include <tvm/api_registry.h>
#include <tvm/base.h>
#include <tvm/buffer.h>
#include <tvm/build_module.h>
#include <tvm/codegen.h>
#include <tvm/dtype.h>
#include <tvm/expr.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/lowered_func.h>
#include <tvm/node/container.h>
#include <tvm/node/node.h>
#include <tvm/operation.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/schedule.h>
#include <tvm/schedule_pass.h>
#include <tvm/tensor.h>

namespace akg {
using ktvm::Array;
using ktvm::AttrVisitor;
using ktvm::Bool;
using ktvm::Buffer;
using ktvm::BufferNode;
using ktvm::BufferType;
using ktvm::BuildConfig;
using ktvm::ComputeOpNode;
using ktvm::Expr;
using ktvm::ExprEqual;
using ktvm::ExprHash;
using ktvm::ExprNode;
using ktvm::Float;
using ktvm::Handle;
using ktvm::Int;
using ktvm::Integer;
using ktvm::IntImm;
using ktvm::IRPrinter;
using ktvm::IterVar;
using ktvm::IterVarNode;
using ktvm::LoweredFunc;
using ktvm::LoweredFuncNode;
using ktvm::Map;
using ktvm::Node;
using ktvm::NodeEqual;
using ktvm::NodeHash;
using ktvm::NodePtr;
using ktvm::NodeRef;
using ktvm::Operation;
using ktvm::OperationNode;
using ktvm::PlaceholderOpNode;
using ktvm::Range;
using ktvm::RangeNode;
using ktvm::Region;
using ktvm::Schedule;
using ktvm::Stmt;
using ktvm::StmtNode;
using ktvm::StrMapNode;
using ktvm::Target;
using ktvm::Tensor;
using ktvm::TensorNode;
using ktvm::Type;
using ktvm::UInt;
using ktvm::Var;
using ktvm::VarExpr;
using ktvm::Variable;

using ktvm::ir::Add;
using ktvm::ir::Allocate;
using ktvm::ir::And;
using ktvm::ir::Any;
using ktvm::ir::AssertStmt;
using ktvm::ir::AttrStmt;
using ktvm::ir::BinaryOpNode;
using ktvm::ir::Block;
using ktvm::ir::Broadcast;
using ktvm::ir::Call;
using ktvm::ir::Cast;
using ktvm::ir::CmpOpNode;
using ktvm::ir::CommReducer;
using ktvm::ir::CommReducerNode;
using ktvm::ir::DeviceAPI;
using ktvm::ir::Div;
using ktvm::ir::EQ;
using ktvm::ir::Evaluate;
using ktvm::ir::FloatImm;
using ktvm::ir::FloorDiv;
using ktvm::ir::FloorMod;
using ktvm::ir::For;
using ktvm::ir::ForType;
using ktvm::ir::Free;
using ktvm::ir::FunctionBaseNode;
using ktvm::ir::FunctionRef;
using ktvm::ir::GE;
using ktvm::ir::GT;
using ktvm::ir::IfThenElse;
using ktvm::ir::IRMutator;
using ktvm::ir::IRVisitor;
using ktvm::ir::LE;
using ktvm::ir::Let;
using ktvm::ir::LetStmt;
using ktvm::ir::Load;
using ktvm::ir::LT;
using ktvm::ir::Max;
using ktvm::ir::Min;
using ktvm::ir::Mod;
using ktvm::ir::Mul;
using ktvm::ir::NE;
using ktvm::ir::Not;
using ktvm::ir::Or;
using ktvm::ir::Prefetch;
using ktvm::ir::ProducerConsumer;
using ktvm::ir::Provide;
using ktvm::ir::Ramp;
using ktvm::ir::Realize;
using ktvm::ir::Reduce;
using ktvm::ir::Select;
using ktvm::ir::Shuffle;
using ktvm::ir::Store;
using ktvm::ir::StringImm;
using ktvm::ir::Sub;
using ktvm::ir::UIntImm;

using ktvm::runtime::GetObjectPtr;
using ktvm::runtime::GetRef;
using ktvm::runtime::Module;
using ktvm::runtime::Object;
using ktvm::runtime::ObjectEqual;
using ktvm::runtime::ObjectHash;
using ktvm::runtime::ObjectPtr;
using ktvm::runtime::ObjectRef;
using ktvm::runtime::PackedFunc;
using ktvm::runtime::TVMArgs;
using ktvm::runtime::TVMRetValue;

using ktvm::const_false;
using ktvm::const_true;
using ktvm::Downcast;
using ktvm::is_const;
using ktvm::is_zero;
using ktvm::make_const;
using ktvm::make_node;
using ktvm::ir::CanonicalSimplify;
using ktvm::ir::Compare;
using ktvm::ir::Equal;
using ktvm::ir::PostOrderVisit;
using ktvm::ir::RemoveNoOp;
using ktvm::ir::Simplify;
using ktvm::ir::Substitute;
}  // namespace akg
#endif  // INCLUDE_AKG_TVM_H_
