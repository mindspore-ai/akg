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
using air::Array;
using air::AttrVisitor;
using air::Bool;
using air::Buffer;
using air::BufferNode;
using air::BufferType;
using air::BuildConfig;
using air::ComputeOpNode;
using air::Expr;
using air::ExprEqual;
using air::ExprHash;
using air::ExprNode;
using air::Float;
using air::Handle;
using air::Int;
using air::Integer;
using air::IntImm;
using air::IRPrinter;
using air::IterVar;
using air::IterVarNode;
using air::LoweredFunc;
using air::LoweredFuncNode;
using air::Map;
using air::Node;
using air::NodeEqual;
using air::NodeHash;
using air::NodePtr;
using air::NodeRef;
using air::Operation;
using air::OperationNode;
using air::PlaceholderOpNode;
using air::Range;
using air::RangeNode;
using air::Region;
using air::Schedule;
using air::Stmt;
using air::StmtNode;
using air::StrMapNode;
using air::Target;
using air::Tensor;
using air::TensorNode;
using air::Type;
using air::UInt;
using air::Var;
using air::VarExpr;
using air::Variable;

using air::ir::Add;
using air::ir::Allocate;
using air::ir::And;
using air::ir::Any;
using air::ir::AssertStmt;
using air::ir::AttrStmt;
using air::ir::BinaryOpNode;
using air::ir::Block;
using air::ir::Broadcast;
using air::ir::Call;
using air::ir::Cast;
using air::ir::CmpOpNode;
using air::ir::CommReducer;
using air::ir::CommReducerNode;
using air::ir::DeviceAPI;
using air::ir::Div;
using air::ir::EQ;
using air::ir::Evaluate;
using air::ir::FloatImm;
using air::ir::FloorDiv;
using air::ir::FloorMod;
using air::ir::For;
using air::ir::ForType;
using air::ir::Free;
using air::ir::FunctionBaseNode;
using air::ir::FunctionRef;
using air::ir::GE;
using air::ir::GT;
using air::ir::IfThenElse;
using air::ir::IRMutator;
using air::ir::IRVisitor;
using air::ir::LE;
using air::ir::Let;
using air::ir::LetStmt;
using air::ir::Load;
using air::ir::LT;
using air::ir::Max;
using air::ir::Min;
using air::ir::Mod;
using air::ir::Mul;
using air::ir::NE;
using air::ir::Not;
using air::ir::Or;
using air::ir::Prefetch;
using air::ir::ProducerConsumer;
using air::ir::Provide;
using air::ir::Ramp;
using air::ir::Realize;
using air::ir::Reduce;
using air::ir::Select;
using air::ir::Shuffle;
using air::ir::Store;
using air::ir::StringImm;
using air::ir::Sub;
using air::ir::UIntImm;

using air::runtime::GetObjectPtr;
using air::runtime::GetRef;
using air::runtime::Module;
using air::runtime::Object;
using air::runtime::ObjectEqual;
using air::runtime::ObjectHash;
using air::runtime::ObjectPtr;
using air::runtime::ObjectRef;
using air::runtime::PackedFunc;
using air::runtime::TVMArgs;
using air::runtime::TVMRetValue;

using air::const_false;
using air::const_true;
using air::Downcast;
using air::is_const;
using air::is_zero;
using air::make_const;
using air::make_node;
using air::ir::CanonicalSimplify;
using air::ir::Compare;
using air::ir::Equal;
using air::ir::PostOrderVisit;
using air::ir::RemoveNoOp;
using air::ir::Simplify;
using air::ir::Substitute;
}  // namespace akg

namespace air {
namespace ir {
/** Substitute variables with the given pointer with the replacement
 * expression within expr. */
Expr substitute(const Variable *var, Expr replacement, Expr expr);

/** Substitute variables with the given pointer with the replacement
 * expression within stmt. */
Stmt substitute(const Variable *var, Expr replacement, Stmt stmt);

inline Expr substitute(const VarExpr &var, const Expr replacement, const Expr expr) {
  return substitute(var.get(), replacement, expr);
}

inline Stmt substitute(const VarExpr &var, const Expr replacement, const Stmt stmt) {
  return substitute(var.get(), replacement, stmt);
}

/** Substitute variables with pointers in the map. */
// @{
Expr substitute(const std::map<const Variable *, Expr> &replacements, Expr expr);
Stmt substitute(const std::map<const Variable *, Expr> &replacements, Stmt stmt);
// @}

/** Substitute expressions for other expressions. */
// @{
Expr substitute(Expr find, Expr replacement, Expr expr);
Stmt substitute(Expr find, Expr replacement, Stmt stmt);
// @}

/* align_partition.cc needs to call this function from tvm */
Stmt AppendStmts(const Stmt &a, const Stmt &b);

/* simplify_passes_cce.cc needs to call this function from tvm */
bool ExprUseVars(const Expr &expr, const std::unordered_set<const Variable *> &vars);

/*!
 * \brief partition loops in the stmt
 * \param stmt The stmt to do loop partition
 * \param split_const_loop flag to enable partition for const loop
 * \param remove_div_mod removes the division and modulo in the indexing of a tensor by partitioning the loop
 * \param partition_conv: whether to partition the convolution or not
 * \return Transformed stmt.
 */
Stmt LoopPartitionCCE(Stmt stmt, bool split_const_loop, bool remove_div_mod = false, bool partition_conv = false);
}  // namespace ir
}  // namespace air
#endif  // INCLUDE_AKG_TVM_H_
