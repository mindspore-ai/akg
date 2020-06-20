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

#include <tvm/ir.h>
#include <tvm/tensor.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>

namespace akg {
class DumpCVisitor : public IRVisitor {
 public:
  DumpCVisitor() = default;
  ~DumpCVisitor() override = default;
  std::string run(const Stmt &stmt, const Array<Buffer> &_extern_buffer_, bool is_in_cdiff_mode,
                  bool is_after_emit_insn, bool is_after_storage_flatten) {
    is_in_cdiff_mode_ = is_in_cdiff_mode;
    is_after_emit_insn_ = is_after_emit_insn;
    is_after_storage_flatten_ = is_after_storage_flatten;
    extern_buffer_ = _extern_buffer_;
    indent_level_ = 0;
    realized_names_.clear();
    realized_dim_.clear();
    PrintFuncHeader();
    Visit(stmt);
    PrintFuncFooter();
    PrintGlobalVars();
    PrintMain();
    return out_.str();
  }

 private:
  void PrintFuncHeaderTwoVersions() {
    PrintIndent();
    out_ << "static void cpp_kernel(";
    bool is_first = true;
    for (auto buffer : extern_buffer_) {
      if (is_first) {
        is_first = false;
      } else {
        out_ << ", ";
      }

      bool enable_cdiff = is_in_cdiff_mode_ && !is_after_storage_flatten_;
      if (enable_cdiff) {
        out_ << "Buffer& ";
      } else {
        out_ << buffer->dtype << " ";
      }

      bool binds_use_basic_ptr = is_after_emit_insn_;
      if (binds_use_basic_ptr) {
        out_ << "*";
      }

      out_ << buffer->name;

      if (!enable_cdiff && !binds_use_basic_ptr) {
        for (auto bound : buffer->shape) {
          out_ << "[" << bound << "]";
        }
      }

      realized_names_.insert(buffer->name);
      realized_dim_[buffer->name] = buffer->shape.size();
    }
    out_ << ")" << std::endl;
  }

  void PrintFuncHeader() {
    out_ << std::endl;
    PrintFuncHeaderTwoVersions();
    PrintIndent();
    out_ << "{" << std::endl;
    indent_level_++;
  }

  void PrintFuncFooter() {
    indent_level_--;
    PrintIndent();
    out_ << "}" << std::endl;
    out_ << std::endl;
  }

  void PrintInitEnv() {
    auto signals_to_capture = {"SIGSEGV", "SIGBUS", "SIGINT", "SIGHUP", "SIGPIPE", "SIGSTOP"};
    for (const auto &signal : signals_to_capture) {
      PrintIndent();
      out_ << "signal(" << signal << ", signal_handler);" << std::endl;
    }
    out_ << std::endl;
  }

  void PrintTrackedBuf(const Buffer &buffer, const std::string &name) {
    out_ << "Buffer " << name << "(" << '"' << buffer->name << '"' << ", "
         << "{ ";
    for (size_t dim = 0; dim < buffer->shape.size(); dim++) {
      out_ << buffer->shape[dim];
      if (dim < buffer->shape.size() - 1) {
        out_ << ", ";
      }
    }
    out_ << " })";
  }

  void PrintBufferArray(const Buffer &buffer, const std::string &name) {
    out_ << buffer->dtype << " " << name;
    for (const auto &dim : buffer->shape) {
      out_ << "[" << dim << "]";
    }
  }

  void PrintInit_tracker() {
    for (const auto &buffer : extern_buffer_) {
      std::string name = MangleTrackedBufferName(buffer->name);
      PrintIndent();
      out_ << "static ";
      PrintAlignmentAttribute();
      if (is_after_storage_flatten_) {
        PrintBufferArray(buffer, name);
      } else {
        PrintTrackedBuf(buffer, name);
      }
      out_ << ";" << std::endl;
    }
    out_ << std::endl;
  }

  void PrintInput_tensor_def() {
    for (const auto &buffer : extern_buffer_) {
      PrintIndent();
      out_ << "static " << buffer->dtype << "_t " << buffer->name;
      for (const auto &bound : buffer->shape) {
        out_ << "[" << bound << "]";
      }
      out_ << ";" << std::endl;
    }
    out_ << std::endl;
  }

  void PrintGlobalVars() {
    PrintInput_tensor_def();
    if (is_in_cdiff_mode_) {
      PrintInit_tracker();
    }
  }

  void PrintReadInput() {
    size_t count = 0;
    PrintIndent();
    out_ << "FILE *fp;" << std::endl;
    PrintIndent();
    out_ << "size_t array_size;" << std::endl;
    out_ << std::endl;

    for (auto buffer : extern_buffer_) {
      PrintIndent();
      out_ << "fp = fopen(\"in_" << std::to_string(count++) << R"(.bin", "rb");)" << std::endl;
      PrintIndent();
      out_ << "CHECK(fp);" << std::endl;
      PrintIndent();
      out_ << "array_size = sizeof(" << buffer->name << ");" << std::endl;
      PrintIndent();
      out_ << "wrapped_fread((void *)" << buffer->name << ", array_size, fp);" << std::endl;
      PrintIndent();
      out_ << "fclose(fp);" << std::endl;
      out_ << std::endl;
    }
  }

  static std::string MangleTrackedBufferName(const std::string &buffer_name) { return buffer_name + "_tracked"; }

  void PrintLaunchKernel() {
    PrintIndent();
    out_ << "launch_kernel();" << std::endl;
  }

  void PrintTrackedCallKernel(bool mangle_name = false) {
    bool binds_use_basic_ptr = is_after_emit_insn_;
    if (is_in_cdiff_mode_ && binds_use_basic_ptr) {
      PrintIndent();
      out_ << "DisableUndefinedAssignCheck();" << std::endl;
    }

    PrintIndent();
    out_ << "cpp_kernel(";
    bool is_first = true;
    for (auto buffer : extern_buffer_) {
      if (is_first) {
        is_first = false;
      } else {
        out_ << ", ";
      }

      if (binds_use_basic_ptr) {
        // cast array ptr to basic ptr
        out_ << "(" << buffer->dtype << "*)";
      }

      if (mangle_name) {
        out_ << MangleTrackedBufferName(buffer->name);
      } else {
        out_ << buffer->name;
      }
    }
    out_ << ");" << std::endl;

    if (is_in_cdiff_mode_ && binds_use_basic_ptr) {
      PrintIndent();
      out_ << "RestoreUndefinedAssignCheck();" << std::endl;
    }
    out_ << std::endl;
  }

  static std::string GetLoopVarName(size_t dim) { return "cc" + std::to_string(dim); }

  void PrintBufferLoopVar(const Buffer &buffer, size_t dim) {
    std::string loop_var = GetLoopVarName(dim);
    PrintIndent();
    out_ << "for (int " << loop_var << " = 0; " << loop_var << " < " << buffer->shape[dim] << "; "
         << "++" << loop_var << ") {" << std::endl;
  }

  void PrintBufferIndex(const Buffer &buffer) {
    for (size_t dim = 0; dim < buffer->shape.size(); dim++) {
      std::string loop_var = GetLoopVarName(dim);
      out_ << "[" << loop_var << "]";
    }
  }

  void PrintCopyData(bool is_to_tracker) {
    for (auto buffer : extern_buffer_) {
      for (size_t dim = 0; dim < buffer->shape.size(); dim++) {
        PrintBufferLoopVar(buffer, dim);
        indent_level_++;
      }

      PrintIndent();
      if (is_to_tracker) {
        out_ << MangleTrackedBufferName(buffer->name);
      } else {
        out_ << buffer->name;
      }

      PrintBufferIndex(buffer);
      out_ << " = ";

      if (is_to_tracker) {
        out_ << buffer->name;
      } else {
        out_ << MangleTrackedBufferName(buffer->name);
      }
      PrintBufferIndex(buffer);
      if (!is_to_tracker) {
        out_ << ".GetValue()";
      }
      out_ << ";";
      out_ << std::endl;

      for (size_t dim = 0; dim < buffer->shape.size(); dim++) {
        indent_level_--;
        PrintIndent();
        out_ << "}" << std::endl;
      }
    }
    out_ << std::endl;
  }

  void PrintCopyDataToTracker() { PrintCopyData(true); }

  void PrintCopyDataFromTracker() { PrintCopyData(false); }

  void PrintCallKernel() {
    if (!is_in_cdiff_mode_) {
      PrintTrackedCallKernel(false);
    } else {
      PrintCopyDataToTracker();
      PrintLaunchKernel();
      PrintTrackedCallKernel(true);
      PrintCopyDataFromTracker();
    }
  }

  void PrintWriteOutput() {
    size_t count = 0;
    for (auto buffer : extern_buffer_) {
      PrintIndent();
      out_ << "fp = fopen(\"out_" << std::to_string(count++) << R"(.bin", "wb");)" << std::endl;
      PrintIndent();
      out_ << "CHECK(fp);" << std::endl;
      PrintIndent();
      out_ << "array_size = sizeof(" << buffer->dtype << ")";
      for (auto bound : buffer->shape) {
        out_ << " * " << bound;
      }
      out_ << ";" << std::endl;
      PrintIndent();
      out_ << "wrapped_fwrite((void *)" << buffer->name << ", array_size, fp);" << std::endl;
      PrintIndent();
      out_ << "fclose(fp);" << std::endl;
      out_ << std::endl;
    }
  }

  void PrintMain() {
    out_ << std::endl;
    PrintIndent();
    out_ << "int main() {" << std::endl;
    indent_level_++;

    PrintInitEnv();
    PrintReadInput();
    PrintCallKernel();
    PrintWriteOutput();

    PrintIndent();
    out_ << "return 0;" << std::endl;
    indent_level_--;
    PrintIndent();
    out_ << "}" << std::endl;
  }

  void PrintIndent() {
    for (int i = 0; i < indent_level_; i++) {
      out_ << "  ";
    }
  }

  void BeginScope() {
    indent_level_++;
    outer_scope_realized_names_.push_back(realized_names_);
    outer_scope_realized_dim_.push_back(realized_dim_);
  }

  void EndScope() {
    indent_level_--;
    realized_names_ = outer_scope_realized_names_.back();
    outer_scope_realized_names_.pop_back();
    realized_dim_ = outer_scope_realized_dim_.back();
    outer_scope_realized_dim_.pop_back();
  }

  template <class T>
  std::string MangleFuncName(const T *op) {
    std::string name = op->func->func_name();
    if (op->func->num_outputs() != 1) {
      name += "_v" + std::to_string(op->value_index);
    }
    return name;
  }

  void PrintMulticore(const AttrStmt *op) {
    PrintIndent();
    out_ << "struct { ";
    if (is_in_cdiff_mode_) {
      out_ << "iterator_t x;";
    } else {
      out_ << "size_t x;";
    }
    out_ << " } blockIdx;" << std::endl;
    if (is_in_cdiff_mode_) {
      PrintIndent();
      out_ << "blockIdx.x.init(\"blockIdx\", 0);" << std::endl;
    }

    PrintIndent();
    out_ << "for (blockIdx.x = 0; "
         << "blockIdx.x < " << op->value << "; "
         << "blockIdx.x++) {" << std::endl;
    ++indent_level_;
  }

  void PrintMulticore_footer() {
    --indent_level_;
    PrintIndent();
    out_ << "}" << std::endl;
  }

  void Visit_(const AttrStmt *op) override {
    if (op->attr_key == "thread_extent") {
      PrintMulticore(op);
    }

    // we need to use block comment because the attribute may span multiple lines
    PrintIndent();
    out_ << "/* attr [" << op->node << "] " << op->attr_key << " = ";
    Visit(op->value);
    out_ << " */" << std::endl;

    Visit(op->body);

    if (op->attr_key == "thread_extent") {
      PrintMulticore_footer();
    }
  }

  void Visit_(const Block *op) override {
    Visit(op->first);
    Visit(op->rest);
  }

  void Visit_(const IfThenElse *op) override {
    PrintIndent();
    out_ << "if (";
    Visit(op->condition);
    out_ << ") {" << std::endl;

    BeginScope();
    Visit(op->then_case);
    EndScope();

    PrintIndent();
    out_ << "}" << std::endl;

    if (op->else_case.defined()) {
      PrintIndent();
      out_ << "else {" << std::endl;

      BeginScope();
      Visit(op->else_case);
      EndScope();

      PrintIndent();
      out_ << "}" << std::endl;
    }
  }

  void Visit_(const For *op) override {
    PrintIndent();
    if (is_in_cdiff_mode_) {
      out_ << "for (iterator_t(" << op->loop_var << ", ";
      Visit(op->min);
      out_ << ")";
    } else {
      out_ << "for (int " << op->loop_var << " = ";
      Visit(op->min);
    }
    out_ << "; " << op->loop_var << " < ";
    Visit(ktvm::ir::Simplify(op->min + op->extent));
    out_ << "; ++" << op->loop_var << ") {" << std::endl;

    BeginScope();
    Visit(op->body);
    EndScope();

    PrintIndent();
    out_ << "}" << std::endl;
  }

  void AddFlattenedDims(const std::string &name, size_t call_dims) {
    if (realized_dim_.count(name)) {
      if (realized_dim_[name] > call_dims) {
        for (size_t i = 0; i < realized_dim_[name] - call_dims; ++i) {
          out_ << "[0]";
        }
      } else if (realized_dim_[name] < call_dims) {
        LOG(FATAL) << "call dims is larger than realized dims";
      }
    }
  }

  void Visit_(const Provide *op) override {
    PrintIndent();
    std::string name = MangleFuncName<Provide>(op);
    out_ << name;
    AddFlattenedDims(name, op->args.size());
    for (const auto &arg : op->args) {
      out_ << "[";
      Visit(arg);
      out_ << "]";
    }
    out_ << " = ";
    Visit(op->value);
    out_ << ";" << std::endl;
  }

  void Visit_(const Store *op) override {
    PrintIndent();
    std::string name = op->buffer_var->name_hint;
    out_ << name;
    AddFlattenedDims(name, 1);
    out_ << "[";
    Visit(op->index);
    out_ << "] = ";
    Visit(op->value);
    out_ << ";" << std::endl;
  }

  void Visit_(const Load *op) override {
    std::string name = op->buffer_var->name_hint;
    out_ << name;
    AddFlattenedDims(name, 1);
    out_ << "[";
    Visit(op->index);
    out_ << "]";
  }

  void Visit_(const ProducerConsumer *op) override {
    if (op->is_producer) {
      PrintIndent();
      out_ << "// produce " << op->func->func_name() << " {" << std::endl;

      Visit(op->body);

      PrintIndent();
      out_ << "// } end produce " << op->func->func_name() << std::endl;
    } else {
      Visit(op->body);
    }
  }

  void PrintAlignmentAttribute() {
    if (is_after_emit_insn_) {
      constexpr int alignment = 1024;
      out_ << "__attribute__ ((aligned(" << alignment << " * sizeof(uint8)))) ";
    }
  }

  void PrintUntrackedTensorDef(const Realize *op, const std::string &name) {
    out_ << op->type << " " << name;
    for (const auto &bound : op->bounds) {
      out_ << "[";
      Visit(ktvm::ir::Simplify(bound->min + bound->extent));
      out_ << "]";
    }
    out_ << ";" << std::endl;
  }

  void PrintTrackedTensorDef(const Realize *op, const std::string &name) {
    out_ << "Buffer " << name << "(" << '"' << name << '"' << ", "
         << "{ ";
    for (size_t dim = 0; dim < op->bounds.size(); dim++) {
      out_ << ktvm::ir::Simplify(op->bounds[dim]->min + op->bounds[dim]->extent);
      if (dim < op->bounds.size() - 1) {
        out_ << ", ";
      }
    }
    out_ << " });" << std::endl;
  }

  void PrintTensorDef(const Realize *op, const std::string &name) {
    PrintIndent();
    PrintAlignmentAttribute();
    if (!is_in_cdiff_mode_ || is_after_storage_flatten_) {
      PrintUntrackedTensorDef(op, name);
    } else {
      PrintTrackedTensorDef(op, name);
    }
  }

  void Visit_(const Realize *op) override {
    std::string name = MangleFuncName<Realize>(op);
    bool realized = (realized_names_.count(name) > 0);
    if (realized) {
      PrintIndent();
      out_ << "// nested realize: " << name << std::endl;
    } else {
      realized_names_.insert(name);
      PrintTensorDef(op, name);
      realized_dim_[name] = op->bounds.size();
    }

    Visit(op->body);
  }

  void PrintUntrackedAllocateDef(const Allocate *op) {
    out_ << op->type << " " << op->buffer_var << "[";
    for (size_t dim = 0; dim < op->extents.size(); dim++) {
      Visit(op->extents[dim]);
      if (dim < op->extents.size() - 1) {
        out_ << " * ";
      }
    }
    out_ << "];" << std::endl;
  }

  void PrintTrackedAllocateDef(const Allocate *op) {
    out_ << "Buffer " << op->buffer_var->name_hint << "(" << '"' << op->buffer_var->name_hint << '"' << ", "
         << "{ ";
    for (size_t dim = 0; dim < op->extents.size(); dim++) {
      Visit(op->extents[dim]);
      if (dim < op->extents.size() - 1) {
        out_ << ", ";
      }
    }
    out_ << " });" << std::endl;
  }

  void PrintAllocateDef(const Allocate *op) {
    PrintAlignmentAttribute();
    if (!is_in_cdiff_mode_ || is_after_storage_flatten_) {
      PrintUntrackedAllocateDef(op);
    } else {
      PrintTrackedAllocateDef(op);
    }
  }

  void Visit_(const Allocate *op) override {
    PrintIndent();
    std::string name = op->buffer_var->name_hint;
    bool realized = (realized_names_.count(name) > 0);
    if (realized) {
      out_ << "// allocated ";
    } else {
      realized_names_.insert(name);
    }

    PrintAllocateDef(op);
    realized_dim_[name] = 1;

    Visit(op->body);
  }

  void Visit_(const Free *op) override {
    PrintIndent();
    out_ << "// free(" << op->buffer_var << ");" << std::endl;
  }

  void Visit_(const Evaluate *op) override {
    PrintIndent();
    Visit(op->value);
    out_ << ";" << std::endl;
  }

  void Visit_(const Call *op) override {
    if (op->call_type == Call::CallType::Halide) {  // tensor index
      std::string name = MangleFuncName<Call>(op);
      out_ << name;
      AddFlattenedDims(name, op->args.size());
      for (auto arg : op->args) {
        out_ << "[";
        Visit(arg);
        out_ << "]";
      }
    } else {  // intrinsic function call
      out_ << op->name << "(";
      for (size_t i = 0; i < op->args.size(); i++) {
        Visit(op->args[i]);
        if (i < op->args.size() - 1) {
          out_ << ", ";
        }
      }
      out_ << ")";
    }
  }

  void Visit_(const IntImm *op) override { out_ << op->value; }

  void Visit_(const UIntImm *op) override { out_ << op->value << "u"; }

  void Visit_(const FloatImm *op) override {
    const int fp16_num_bits = 16;
    if (op->type.bits() == fp16_num_bits) {
      out_ << "float16(" << op->value << ")";
    } else {
      out_ << op->value;
    }
  }

  void Visit_(const StringImm *op) override { out_ << "\"" << op->value << "\""; }

  void Visit_(const Cast *op) override {
    out_ << "(" << op->type << ")";
    Visit(op->value);
  }

  void Visit_(const Variable *op) override { out_ << op->name_hint; }

  template <class T>
  void DumpBinaryOp(const T *op, const std::string &str) {
    out_ << "(";
    Visit(op->a);
    out_ << " " << str << " ";
    Visit(op->b);
    out_ << ")";
  }

  void Visit_(const Add *op) override { DumpBinaryOp<Add>(op, "+"); }

  void Visit_(const Sub *op) override { DumpBinaryOp<Sub>(op, "-"); }

  void Visit_(const Mul *op) override { DumpBinaryOp<Mul>(op, "*"); }

  void Visit_(const Div *op) override { DumpBinaryOp<Div>(op, "/"); }

  void Visit_(const FloorDiv *op) override { DumpBinaryOp<FloorDiv>(op, "/"); }

  void Visit_(const Mod *op) override { DumpBinaryOp<Mod>(op, "%"); }

  void Visit_(const FloorMod *op) override { DumpBinaryOp<FloorMod>(op, "%"); }

  void Visit_(const Min *op) override {
    out_ << "(";
    Visit(op->a);
    out_ << " < ";
    Visit(op->b);
    out_ << " ? ";
    Visit(op->a);
    out_ << " : ";
    Visit(op->b);
    out_ << ")";
  }

  void Visit_(const Max *op) override {
    out_ << "(";
    Visit(op->a);
    out_ << " < ";
    Visit(op->b);
    out_ << " ? ";
    Visit(op->b);
    out_ << " : ";
    Visit(op->a);
    out_ << ")";
  }

  void Visit_(const EQ *op) override { DumpBinaryOp<EQ>(op, "=="); }

  void Visit_(const NE *op) override { DumpBinaryOp<NE>(op, "!="); }

  void Visit_(const LT *op) override { DumpBinaryOp<LT>(op, "<"); }

  void Visit_(const GT *op) override { DumpBinaryOp<GT>(op, ">"); }

  void Visit_(const LE *op) override { DumpBinaryOp<LE>(op, "<="); }

  void Visit_(const GE *op) override { DumpBinaryOp<GE>(op, ">="); }

  void Visit_(const And *op) override { DumpBinaryOp<And>(op, "&&"); }

  void Visit_(const Or *op) override { DumpBinaryOp<Or>(op, "||"); }

  void Visit_(const Not *op) override {
    out_ << "!";
    Visit(op->a);
  }

  void Visit_(const Select *op) override {
    out_ << "(";
    Visit(op->condition);
    out_ << " ? ";
    Visit(op->true_value);
    out_ << " : ";
    Visit(op->false_value);
    out_ << ")";
  }

  std::stringstream out_;
  int indent_level_{0};
  Array<Buffer> extern_buffer_;
  std::unordered_set<std::string> realized_names_;
  std::unordered_map<std::string, size_t> realized_dim_;
  std::vector<std::unordered_set<std::string>> outer_scope_realized_names_;
  std::vector<std::unordered_map<std::string, size_t>> outer_scope_realized_dim_;
  bool is_in_cdiff_mode_{false};
  bool is_after_emit_insn_{false};
  bool is_after_storage_flatten_{false};
};

static bool IsAfterEmitInsn(const Stmt &stmt) {
  bool found_tvm_access_ptr = false;
  PostOrderVisit(stmt, [&found_tvm_access_ptr](const NodeRef &node) {
    if (auto call_node = node.as<Call>()) {
      if (call_node->name == "tvm_access_ptr" && call_node->call_type != Call::CallType::Halide) {
        found_tvm_access_ptr = true;
      }
    }
  });
  return found_tvm_access_ptr;
}

static bool IsAfterStorageFlatten(const Stmt &stmt) {
  bool found_allocate = false;
  PostOrderVisit(stmt, [&found_allocate](const NodeRef &node) {
    if (node.as<Allocate>()) {
      found_allocate = true;
    }
  });
  return found_allocate;
}

static bool IsInCdiffMode() {
  const char *runtime_mode = std::getenv("RUNTIME_MODE");
  if (runtime_mode == nullptr) {
    return false;
  }
  std::string runtime_mode_str = runtime_mode;
  return runtime_mode_str == "cdiff";
}

std::string DumpC(const Stmt &stmt, const Array<Buffer> &extern_buffer_) {
  std::string c_code =
    DumpCVisitor().run(stmt, extern_buffer_, IsInCdiffMode(), IsAfterEmitInsn(stmt), IsAfterStorageFlatten(stmt));
  return c_code;
}
}  // namespace akg
