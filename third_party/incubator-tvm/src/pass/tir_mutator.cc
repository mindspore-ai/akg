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
 * \file tir_mutator.cc
 * \brief Visit all nodes in IR and add tvm 0.8 attributes to the nodes.
 */

/*
 * 2023.12.19 - Add case where index calculation exceeds int32.
 * 2023.03.25 - Add TVM 0.8 attributes to the node and conversion pass for exporting TVM 0.8 IR.
 */
#include "tir_mutator.h"

#include "ir_util.h"

namespace air {
namespace ir {

class IntrinsicRegister{
  public:
  static IntrinsicRegister* Global();
  bool GetIntrinsicForTir(const std::string& key, std::string* value);
  IntrinsicRegister& RegisterIntrinsicForTir(const char* key, const char* value);

  private:
  std::unordered_map<std::string,std::string> intrinsic_map;
};

IntrinsicRegister* IntrinsicRegister::Global(){
    static IntrinsicRegister inst;
    return &inst;
}

IntrinsicRegister& IntrinsicRegister::RegisterIntrinsicForTir(const char* key, const char* value){
    intrinsic_map[key] = value;
  return *this;
}

bool IntrinsicRegister::GetIntrinsicForTir(const std::string& key, std::string* value){
  if(intrinsic_map.count(key)){
    if(value != nullptr){
      *value = intrinsic_map.at(key);
      return true;
    }
  }
  return false;
}

#define REGISTER_INTRINSIC_FOR_TIR(intrinsic,tir_intrinsic)\
static IntrinsicRegister TVM_STR_CONCAT(IntrinsicReg,__COUNTER__) = IntrinsicRegister::Global()->RegisterIntrinsicForTir(intrinsic,tir_intrinsic);

REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_address_of,"tir.tvm_address_of")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_if_then_else,"tir.tvm_if_then_else")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_access_ptr,"tir.tvm_access_ptr")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_static_handle,"tir.tvm_static_handle")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_context_id,"tir.tvm_context_id")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_tuple,"tir.tvm_tuple")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_struct_get,"tir.tvm_struct_get")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_handle_is_null,"tir.isnullptr")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_throw_last_error,"tir.tvm_throw_last_error")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_stack_alloca,"tir.tvm_stack_alloca")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_stack_make_shape,"tir.tvm_stack_make_shape")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_stack_make_array,"tir.tvm_stack_make_array")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_call_packed,"tir.tvm_call_packed")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_call_trace_packed,"tir.tvm_call_trace_packed")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_thread_context,"tir.tvm_thread_context")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_call_packed_lowered,"tir.tvm_call_packed_lowered")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_call_trace_packed_lowered,"tir.tvm_call_trace_packed_lowered")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_storage_sync,"tir.tvm_storage_sync")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_warp_shuffle,"tir.tvm_warp_shuffle")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_global_barrier_kinit,"tir.tvm_global_barrier_kinit")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_thread_allreduce,"tir.tvm_thread_allreduce")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_load_matrix_sync,"tir.tvm_load_matrix_sync")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_mma_sync,"tir.tvm_mma_sync")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_fill_fragment,"tir.tvm_mma_sync")
REGISTER_INTRINSIC_FOR_TIR(intrinsic::tvm_store_matrix_sync,"tir.tvm_store_matrix_sync")
REGISTER_INTRINSIC_FOR_TIR("type_annotation","tir.type_annotation")

void VisitIntrinsicOp(Call* self){
  std::string intrinsic_name;
  if(!IntrinsicRegister::Global()->GetIntrinsicForTir(self->name,&intrinsic_name)){
    intrinsic_name = self->name;
  }
  if(self->op.defined()){
    auto op_node = const_cast<IntrinsicOpNode*>(self->op.operator->());
    op_node->name = std::move(intrinsic_name);
  }else{
    auto op = IntrinsicOp(intrinsic_name);
    self->op = op;
  }
}

template <typename T>
inline void VisitArray(IRConverter* m, const Array<T>& arr) {
  for (size_t i = 0; i < arr.size(); ++i) {
    m->Visit_(arr[i]);
  }
}


inline void VisitBufferVar(const Var& var, DataType dtype) {
  auto copy_on_write_buffer = const_cast<Variable*>(var.get());
  DataType storage_dtype = (dtype == Bool() ? Int(8) : dtype);
  copy_on_write_buffer->name_hint_str = var->name_hint;
  if (var->type_annotation.defined()) {
    auto type_annotation = Downcast<PointerType>(var->type_annotation);
    auto prim_type = Downcast<PrimType>(type_annotation->element_type);
    auto copy_on_write_prim_type = const_cast<PrimTypeNode*>(static_cast<const PrimTypeNode*>(prim_type.get()));
    copy_on_write_prim_type->dtype = storage_dtype;
  } else {
    TypeAnnotationType type_annotation = PointerType(PrimType(storage_dtype));
    copy_on_write_buffer->type_annotation = type_annotation;
  }
}

void IRConverter::Visit_(const Allocate* op) {
  VisitBufferVar(op->buffer_var, op->type);
  IRVisitor::Visit_(op);
}

void IRConverter::Visit_(const AttrStmt* op) {
  this->Visit_(op->node);
  if(op->attr_key == "storage_scope" && op->node->IsInstance<Variable>()){
    auto var = const_cast<Variable*>(op->node.as<Variable>());
    if(var->type_annotation.defined()){
      auto type_annotation = const_cast<PointerTypeNode*>(var->type_annotation.as<PointerTypeNode>());
      type_annotation->storage_scope = op->value.as<StringImm>()->value;
    }else{
      auto dtype = var->type;
      DataType storage_dtype = (dtype == Bool() ? Int(8) : dtype);
      TypeAnnotationType type_annotation = PointerType(PrimType(storage_dtype),op->value.as<StringImm>()->value);
      var->type_annotation = type_annotation;
    }
  }
  IRVisitor::Visit_(op);
}

void IRConverter::Visit_(const Block* op) {
  auto copy_on_write_block = const_cast<Block*>(op);
  auto array = op->seq;
  std::vector<Stmt> seq;
  FlattenBlock(op->first, &seq);
  FlattenBlock(op->rest, &seq);
  Array<Stmt> new_seq = seq;
  copy_on_write_block->seq = std::move(new_seq);
}

void IRConverter::Visit_(const Call* op) {
  auto copy_on_write_call = const_cast<Call*>(op);
  if(op->call_type == Call::CallType::PureIntrinsic){
    Expr strimm = StringImm::make("tir."+op->name);
    Array<Expr> args_with_op_name;
    args_with_op_name.push_back(strimm);
    for(auto& item:op->args){
      args_with_op_name.push_back(item);
    }
    copy_on_write_call->args = args_with_op_name;
    auto op = IntrinsicOp("tir.call_extern");
    copy_on_write_call->op = op;
  }else if(op->call_type==Call::CallType::Intrinsic){
    VisitIntrinsicOp(copy_on_write_call);
  }
  IRVisitor::Visit_(op);
}

void IRConverter::Visit_(const Load* op) {
  VisitBufferVar(op->buffer_var, op->type);
  IRVisitor::Visit_(op);
}
void IRConverter::Visit_(const Store* op) {
  VisitBufferVar(op->buffer_var, op->value.type());
  IRVisitor::Visit_(op);
}

void IRConverter::Visit_(const Provide* op) { IRVisitor::Visit_(op); }
void IRConverter::Visit_(const Prefetch* op) { IRVisitor::Visit_(op); }
void IRConverter::Visit_(const Realize* op) { IRVisitor::Visit_(op); }

void IRConverter::Visit_(const FunctionRef& op) {
  if (op->IsInstance<PlaceholderOpNode>()) {
    this->Visit_(op.as<PlaceholderOpNode>());
  } else if (op->IsInstance<TensorComputeOpNode>()) {
    this->Visit_(op.as<TensorComputeOpNode>());
  } else if (op->IsInstance<ComputeOpNode>()) {
    this->Visit_(op.as<ComputeOpNode>());
  } else if (op->IsInstance<ScanOpNode>()) {
    this->Visit_(op.as<ScanOpNode>());
  } else if (op->IsInstance<ExternOpNode>()) {
    this->Visit_(op.as<ExternOpNode>());
  }
}

void IRConverter::Visit_(const ComputeOpNode* op) {
  VisitArray(this, op->body);
}

void IRConverter::Visit_(const ExternOpNode* op) {
  VisitArray(this, op->inputs);
  VisitArray(this, op->input_placeholders);
  VisitArray(this, op->output_placeholders);
  this->Visit_(op->body);
}

void IRConverter::Visit_(const HybridOpNode* op) {
  this->Visit_(op->body);
}

void IRConverter::Visit_(const PlaceholderOpNode* op) {
return;
}

void IRConverter::Visit_(const ScanOpNode* op) {
  VisitArray(this, op->init);
  VisitArray(this, op->update);
  VisitArray(this, op->state_placeholder);
  VisitArray(this, op->inputs);
}

void IRConverter::IRConverter::Visit_(const TensorComputeOpNode* op) {
  VisitArray(this, op->inputs);
}

void IRConverter::Visit_(const BufferNode* op){
  VisitBufferVar(op->data, op->dtype);
}

void IRConverter::Visit_(const IterVarNode* op) {
  this->Visit_(op->dom);
  this->Visit_(op->var);
}

void IRConverter::Visit_(const RangeNode* op) {
  this->Visit_(op->min);
  this->Visit_(op->extent);
}

void IRConverter::Visit_(const TensorNode* op) {this->Visit_(op->op); }

void IRConverter::Visit_(const TensorIntrinNode* op) {
  this->Visit_(op->op);
  VisitArray(this, op->inputs);
  VisitArray(this, op->scalar_params);
  this->Visit_(op->body);
  this->Visit_(op->reduce_init);
  this->Visit_(op->reduce_update);
}

void IRConverter::Visit_(const IntImm* op) {
  auto int_op = const_cast<IntImm*>(op);
  const int64_t int_max = 2147483647;
  const int64_t block_max = 64;
  int64_t upper_limit = int_max / block_max;
  if (int_op->value >  upper_limit) {
    DataType new_type = Int(64, 1);
    int_op->type = new_type;
  }
}

void IRConverter::Visit_(const ObjectRef& f) {
  if (!f.defined()) {
    return;
  }
  if (f.get()->IsInstance<ExprNode>()) {
    this->Visit(GetRef<Expr>(f.as<ExprNode>()));
  } else if (f.get()->IsInstance<StmtNode>()) {
    this->Visit(GetRef<Stmt>(f.as<StmtNode>()));
  } else if (f.get()->IsInstance<BufferNode>()) {
    this->Visit_(f.as<BufferNode>());
  } else if (f.get()->IsInstance<IterVarNode>()) {
    this->Visit_(f.as<IterVarNode>());
  } else if (f.get()->IsInstance<TensorNode>()) {
    this->Visit_(f.as<TensorNode>());
  } else if (f.get()->IsInstance<FunctionBaseNode>()) {
    this->Visit_(GetRef<FunctionRef>(f.as<FunctionBaseNode>()));
  }
}

void IRConverter::FlattenBlock(const Stmt& s, std::vector<Stmt>* res) {
  if (const Block* op = s.as<Block>()) {
    FlattenBlock(op->first, res);
    FlattenBlock(op->rest, res);
  } else {
    this->Visit(s);
    res->emplace_back(s);
  }
}

void IR_Conversion(const ObjectRef& f) { IRConverter().Visit_(f); }

}  // namespace ir
}  // namespace air