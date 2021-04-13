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
#include <tvm/ir_visitor.h>
#include <tvm/node/serialization.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

#include "build_module.h"
#include "ir_pass.h"
#include "schedule_pass.h"
#include "codegen/pass_mgr.h"
#include "composite/util.h"

namespace akg {
AttrMap global_attrs;
Array<NodeRef> g_external_call_name;

Tensor CreatePlaceholder(const NodeRef &arg) {
  auto n = air::make_node<PlaceholderOpNode>();

  if (auto var_node = arg.as<Variable>()) {
    n->name = var_node->name_hint;
    n->shape = Array<Expr>{GetRef<Expr>(var_node)};
    n->dtype = var_node->type;
  } else if (auto buffer_node = arg.as<BufferNode>()) {
    n->name = buffer_node->name;
    Expr size = std::accumulate(buffer_node->shape.begin(), buffer_node->shape.end(), Expr(1),
                                [](const Expr &mul, const Expr &e) { return mul * e; });
    n->shape = Array<Expr>{air::ir::Simplify(size)};
    n->dtype = buffer_node->dtype;
  } else if (auto tensor_node = arg.as<TensorNode>()) {
    n->name = tensor_node->op->name;
    Expr size = std::accumulate(tensor_node->shape.begin(), tensor_node->shape.end(), Expr(1),
                                [](const Expr &mul, const Expr &e) { return mul * e; });
    n->shape = Array<Expr>{air::ir::Simplify(size)};
    n->dtype = tensor_node->dtype;
  } else {
    LOG(FATAL) << "arg must be Tensor, Buffer or Var, but got " << arg;
  }

  return Operation(n).output(0);
}

Buffer DeclBuffer(const NodeRef &arg, const int data_alignment, const int offset_factor,
                  const std::string &pre_name = "") {
  // use default value.
  Array<Expr> strides;

  Type dtype;
  Array<Expr> shape;
  std::string name = pre_name;

  if (auto variable_node = arg.as<Variable>()) {
    if (name.empty()) {
      name = variable_node->name_hint;
    }
    shape = Array<Expr>{GetRef<Expr>(variable_node)};
    dtype = variable_node->type;
  } else if (auto buffer_node = arg.as<BufferNode>()) {
    if (name.empty()) {
      name = buffer_node->name;
    }
    shape = buffer_node->shape;
    dtype = buffer_node->dtype;
  } else if (auto tensor_node = arg.as<TensorNode>()) {
    if (name.empty()) {
      name = tensor_node->op->name;
    }
    shape = tensor_node->shape;
    dtype = tensor_node->dtype;
  } else {
    LOG(FATAL) << "args must be Tensor, Buffer or Var, but got " << arg;
  }

  auto data = Variable::make(Handle(), name);
  Expr elem_offset;
  if (offset_factor != 0) {
    elem_offset = Variable::make(shape[0].type(), name + "_elem_offset");
  }

  return BufferNode::make(data, dtype, shape, strides, elem_offset, name, "", data_alignment, offset_factor,
                          BufferType::kDefault);
}

void GetBinds(const Array<NodeRef> &args, const Map<Tensor, Buffer> &binds, const BuildConfig &config,
              Array<NodeRef> *out_args, Map<Tensor, Buffer> *out_binds) {
  for (const auto &b : binds) {
    out_binds->Set(b.first, b.second);
  }

  for (const auto &x : args) {
    if (x->IsInstance<TensorNode>()) {
      auto tensor_node = GetRef<Tensor>(x.as<TensorNode>());
      if (out_binds->find(tensor_node) == out_binds->end()) {
        auto buf = DeclBuffer(tensor_node, config->data_alignment, config->offset_factor);
        out_binds->Set(tensor_node, buf);
        out_args->push_back(buf);
      } else {
        out_args->push_back((*out_binds)[tensor_node]);
      }
    } else if (x->IsInstance<BufferNode>()) {
      out_args->push_back(x);
    } else if (x->IsInstance<Variable>()) {
      out_args->push_back(x);
    } else {
      LOG(FATAL) << "args must be Tensor, Buffer or Var, but got " << x;
    }
  }

  return;
}

void GetFlattenedBinds(const Array<NodeRef> &args, const Map<Tensor, Buffer> &binds, const BuildConfig &config,
                       Array<NodeRef> &out_args, Map<Tensor, Buffer> &out_binds, bool is_dynamic) {
  std::unordered_map<Tensor, bool> flag_binds;
  // the map aims to remove duplicate names between binds and args
  // because in-place ops (e.g. assign_add) use the same buffer as input and args, and duplicates need to be removed
  std::unordered_map<std::string, Buffer> bind_name_to_buffer_map;
  for (const auto &b : binds) {
    static_cast<void>(bind_name_to_buffer_map.emplace(b.first->op->func_name(), b.second));
  }
  for (const auto &x : args) {
    if (x->IsInstance<TensorNode>()) {
      auto tensor_node = GetRef<Tensor>(x.as<TensorNode>());
      auto tensor_name = tensor_node->op->func_name();
      CHECK_NE(bind_name_to_buffer_map.count(tensor_name), 0) << "undefined tensor " << x;
      auto bind_buffer = bind_name_to_buffer_map[tensor_name];
      flag_binds[tensor_node] = true;
      Tensor nx = CreatePlaceholder(tensor_node);
      bool find_buf = false;
      for (auto iter : out_binds) {
        Buffer buffer = iter.second;
        if (bind_buffer->name == buffer->name) {
          out_binds.Set(nx, buffer);
          find_buf = true;
          break;
        }
      }

      if (!find_buf) {
        Buffer buf = DeclBuffer(nx, config->data_alignment, config->offset_factor, bind_buffer->name);
        out_binds.Set(nx, buf);
        out_args.push_back(buf);
      } else {
        out_args.push_back(bind_buffer);
      }
    } else if (x->IsInstance<BufferNode>()) {
      out_args.push_back(x);
    } else if (x->IsInstance<Variable>()) {
      out_args.push_back(x);
    } else {
      LOG(FATAL) << "args must be Tensor, Buffer or Var";
    }
  }

  for (const auto &x : binds) {
    Tensor x_tensor = x.first;
    if (flag_binds.insert(std::pair<Tensor, bool>{x_tensor, true}).second) {
      Tensor nx = CreatePlaceholder(x_tensor);
      bool find_buf = false;
      for (auto iter : out_binds) {
        Buffer buffer = iter.second;
        if (binds[x_tensor]->name == buffer->name) {
          out_binds.Set(nx, buffer);
          find_buf = true;
        }
      }

      if (!find_buf) {
        Buffer buf = DeclBuffer(nx, config->data_alignment, config->offset_factor, binds[x_tensor]->name);
        out_binds.Set(nx, buf);
      }
    }
  }

  // Just for reshape in dynamic mode
  if (is_dynamic) {
    Tensor in_tensor, out_tensor;
    bool is_reshape = false;
    if (out_binds.size() == 2 && args.size() == 2) {
      for (auto tb : out_binds) {
        if (tb.first->op->name == "reshape" || tb.first->op->name == "reshape_cast") {
          out_tensor = tb.first;
          is_reshape = true;
        } else {
          in_tensor = tb.first;
        }
      }
    }

    if (is_reshape) {
      Map<Tensor, Buffer> new_binds;
      Array<NodeRef> new_args;
      auto n = air::make_node<PlaceholderOpNode>();
      n->name = out_tensor->op->name;
      n->shape = in_tensor->shape;
      n->dtype = out_tensor->dtype;
      Tensor ten = Operation(n).output(0);
      Buffer buf = DeclBuffer(ten, config->data_alignment, config->offset_factor, n->name);

      new_binds.Set(in_tensor, out_binds[in_tensor]);
      new_binds.Set(ten, buf);
      new_args.push_back(out_binds[in_tensor]);
      new_args.push_back(buf);
      out_binds = new_binds;
      out_args = new_args;
    }
  }
}

void RenameBinds(Map<Tensor, Buffer> &binds, const BuildConfig &config, Array<NodeRef> &tensor_args_list,
                 Array<NodeRef> &buffer_args_list, Map<Tensor, Tensor> &tensor_replace) {
  std::unordered_map<std::string, int> tensor_name_count;
  std::set<std::string> tensor_name;
  Map<Tensor, Buffer> out_binds;
  Map<Buffer, Buffer> buffer_replace;
  bool rename_flag = false;

  // count the number of times for binds name, if op->name's count greater than 1, need rename op->name
  for (const auto &x : binds) {
    ++tensor_name_count[x.first->op->name];
  }

  // if binds' name conflict, firstly rename tensor_name, then construct new mappings, finally set to out_binds
  for (const auto &x : binds) {
    const auto &old_tensor = x.first;
    const auto &old_buffer = x.second;
    if (tensor_name_count[old_tensor->op->name] > 1) {
      int idx = 0;
      std::string new_name = old_tensor->op->name;
      std::string extend;
      do {
        extend = "_rename_" + std::to_string(++idx);
      } while (tensor_name.count(new_name + extend) != 0);
      new_name.append(extend);
      tensor_name.insert(new_name);
      auto cop = old_tensor->op.as<air::ComputeOpNode>();
      CHECK(cop != nullptr);
      Tensor new_tensor = air::ComputeOpNode::make(new_name, cop->tag, cop->attrs, cop->axis, cop->body).output(0);
      tensor_replace.Set(old_tensor, new_tensor);
      if (buffer_replace.count(old_buffer) == 0) {
        auto new_buffer = DeclBuffer(new_tensor, config->data_alignment, config->offset_factor, new_name);
        buffer_replace.Set(old_buffer, new_buffer);
        out_binds.Set(new_tensor, new_buffer);
      }
      rename_flag = true;
    }
  }
  // if there is no conflict in binds name, just do out_binds = binds
  // else need use new_buffer to replace old_buffer to insert out_binds
  auto UpdateOutBinds = [&](Map<Tensor, Buffer> &out_binds) -> Map<Tensor, Buffer> & {
    for (const auto &it : binds) {
      const auto &tensor_node = it.first;
      const auto &buffer_node = it.second;
      if (tensor_name_count[tensor_node->op->name] == 1) {
        if (buffer_replace.count(buffer_node) > 0) {
          out_binds.Set(tensor_node, buffer_replace[buffer_node]);
        } else {
          out_binds.Set(tensor_node, buffer_node);
        }
      }
    }
    return out_binds;
  };

  // traverse the list of tensor_args, according to tensor_node to update tensor_args_list
  auto UpdateArgsByTensor = [&tensor_args_list, &tensor_replace]() {
    Array<NodeRef> new_tensor_args_list;
    for (const auto &x : tensor_args_list) {
      if (x->IsInstance<TensorNode>()) {
        Tensor tensor_node = GetRef<Tensor>(x.as<TensorNode>());
        if (tensor_replace.count(tensor_node) != 0) {
          new_tensor_args_list.push_back(tensor_replace[tensor_node]);
        } else {
          new_tensor_args_list.push_back(tensor_node);
        }
      } else {
        new_tensor_args_list.push_back(x);
      }
    }
    return new_tensor_args_list;
  };

  // traverse the list of buffer_args, according to buffer_node to update buffer_args_list
  auto UpdateArgsByBuffer = [&buffer_args_list, &buffer_replace]() {
    Array<NodeRef> new_buffer_args_list;
    for (const auto &x : buffer_args_list) {
      if (x->IsInstance<BufferNode>()) {
        Buffer buffer_node = GetRef<Buffer>(x.as<BufferNode>());
        if (buffer_replace.count(buffer_node) != 0) {
          new_buffer_args_list.push_back(buffer_replace[buffer_node]);
        } else {
          new_buffer_args_list.push_back(buffer_node);
        }
      } else {
        new_buffer_args_list.push_back(x);
      }
    }
    return new_buffer_args_list;
  };

  // if rename tensor_name, need to update tensor_args and buffer_args
  if (rename_flag) {
    tensor_args_list = UpdateArgsByTensor();
    buffer_args_list = UpdateArgsByBuffer();
    binds = UpdateOutBinds(out_binds);
  }
  return;
}

void FixParametricBinds(const Map<Tensor, Buffer> &binds, const Array<NodeRef> &in_args, const BuildConfig &config,
                        Map<Tensor, Buffer> *out_binds, Array<NodeRef> *out_args) {
  Expr H = 0;
  Expr W = 0;
  Expr PT = 0;
  Expr PB = 0;
  Expr PL = 0;
  Expr PR = 0;
  Expr KH = 0;
  Expr KW = 0;
  Expr SH = 0;
  Expr SW = 0;
  Expr CI1 = 0;
  std::string feature = "input_1_1";
  std::string kernel = "input_1_2";
  std::string bias = "input_1_3";
  std::string output = "output";
  Buffer feature_buffer;
  Buffer kernel_buffer;
  Buffer bias_buffer;
  Buffer output_buffer;
  for (const auto &x : in_args) {
    if (auto buf = x.as<BufferNode>()) {
      if (buf->name.find(feature) != std::string::npos) {
        feature_buffer = Downcast<Buffer>(x);
      }
      if (buf->name.find(bias) != std::string::npos) {
        bias_buffer = Downcast<Buffer>(x);
      }
      if (buf->name.find(output) != std::string::npos || buf->name.find(kernel) != std::string::npos) {
        continue;
      }
    }
    if (auto v = x.as<Variable>()) {
      if (v->name_hint == "H") {
        H = Downcast<Var>(x);
      } else if (v->name_hint == "W") {
        W = Downcast<Var>(x);
      } else if (v->name_hint == "PT") {
        PT = Downcast<Var>(x);
      } else if (v->name_hint == "PB") {
        PB = Downcast<Var>(x);
      } else if (v->name_hint == "PL") {
        PL = Downcast<Var>(x);
      } else if (v->name_hint == "PR") {
        PR = Downcast<Var>(x);
      } else if (v->name_hint == "KH") {
        KH = Downcast<Var>(x);
      } else if (v->name_hint == "KW") {
        KW = Downcast<Var>(x);
      } else if (v->name_hint == "SH") {
        SH = Downcast<Var>(x);
      } else if (v->name_hint == "SW") {
        SW = Downcast<Var>(x);
      } else if (v->name_hint == "CI1") {
        CI1 = Downcast<Var>(x);
      }
    }
  }
  for (const auto &x : binds) {
    Array<Expr> shape;
    if (x.second->name.find(output) != std::string::npos) {
      CHECK_EQ(x.second->shape.size(), 5);
      shape.push_back(x.second->shape[0]);
      shape.push_back(x.second->shape[1]);
      auto h = air::floordiv(H + PT + PB - KH, SH) + 1;
      auto w = air::floordiv(W + PL + PR - KW, SW) + 1;
      shape.push_back(h);
      shape.push_back(w);
      shape.push_back(x.second->shape[4]);
      Tensor tt = air::placeholder(shape, x.second->dtype, x.second->name);
      output_buffer = DeclBuffer(tt, config->data_alignment, config->offset_factor, x.second->name);
      out_binds->Set(tt, output_buffer);
    } else if (x.second->name.find(kernel) != std::string::npos) {
      CHECK_EQ(x.second->shape.size(), 4);
      auto n = CI1 * KH * KW;
      shape.push_back(n);
      shape.push_back(x.second->shape[1]);
      shape.push_back(x.second->shape[2]);
      shape.push_back(x.second->shape[3]);
      Tensor tt = air::placeholder(shape, x.second->dtype, x.second->name);
      kernel_buffer = DeclBuffer(tt, config->data_alignment, config->offset_factor, x.second->name);
      out_binds->Set(tt, kernel_buffer);
    } else {
      out_binds->Set(x.first, x.second);
    }
  }
  if (feature_buffer.defined()) {
    out_args->push_back(feature_buffer);
  }
  if (kernel_buffer.defined()) {
    out_args->push_back(kernel_buffer);
  }
  if (bias_buffer.defined()) {
    out_args->push_back(bias_buffer);
  }
  if (output_buffer.defined()) {
    out_args->push_back(output_buffer);
  }
  for (const auto &x : in_args) {
    if (x.as<Variable>()) {
      out_args->push_back(x);
    }
  }
}

void DumpIr(const std::string &name, const BuildConfig &config, bool lower_list) {
  global_attrs.Set(kKernelName, StringImm::make(name));
  global_attrs.Set(kDumpPassIr, air::make_const(Int(32), config->dump_pass_ir));
  if (config->dump_pass_ir) {
    std::string dump_ir_dir;
    if (global_attrs.GetStringAttr(kDumpIrDir, &dump_ir_dir)) {
      PassMgr::SetDir(dump_ir_dir);
    } else {
      PassMgr::SetDir(name);
    }
    CreateDir(PassMgr::GetDir());
    std::string dump_poly_dir;
    if (!global_attrs.GetStringAttr(kDumpPolyDir, &dump_poly_dir) || lower_list) {
      dump_poly_dir = PassMgr::GetDir() + "/poly";
      global_attrs.Set(kDumpPolyDir, StringImm::make(dump_poly_dir));
      CreateDir(dump_poly_dir);
    }
  }
}

NodeRef LowerStmt(Schedule sch, const Array<NodeRef> &in_args, const Array<NodeRef> &shape_vars,
                  const std::string &name, const Map<Tensor, Buffer> &in_binds,
                  const Map<std::string, NodeRef> &in_attrs, bool simple_mode, bool polyhedral, bool tuning,
                  const std::string &target, const BuildConfig &config, Array<NodeRef> *args,
                  Array<NodeRef> *arg_list_0, Map<Tensor, Buffer> *binds, Map<Tensor, Buffer> *binds_0,
                  bool lower_list) {
  CHECK(sch.defined()) << "sch is not defined.";
  CHECK(!name.empty()) << "name is empty.";
  CHECK(find_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c) && c != '_'; }) == name.end())
    << "kernel name contains invalid chars: " << name;

  if (in_args.defined()) {
    *args = in_args;
  }

  if (in_binds.defined()) {
    *binds = in_binds;
  }
  if (in_attrs.defined()) {
    global_attrs = in_attrs;
  }
  PassMgr::ClearPassId();

  // dump lowerstmt
  DumpIr(name + "_0", config, lower_list);

  GetBinds(*args, *binds, config, arg_list_0, binds_0);

  // Phase 0
  Target target_platform = Target::Create(target);
  if (polyhedral && global_attrs.GetBoolAttr(kEnableAutoInline, true)) {
    akg::schedule::AutoInline(sch, target_platform, global_attrs.GetBoolAttr(kEnableCSE, false));
  }
  if (target_platform->device_type == kDLGPU && polyhedral && global_attrs.GetBoolAttr(kEnableAutoFuse, true)) {
    akg::schedule::AutoFuse(sch);
  }

  auto new_sch = sch.normalize();
  auto bounds = air::schedule::InferBound(new_sch);
  Stmt stmt = make_pass("schedule.ScheduleOps", new_sch, bounds, false);
 
  stmt = NEXT_PASS(TensorAccessRewrite, stmt);
  if (target_platform->device_type == kDLGPU) {
    if (polyhedral) {
      stmt = NEXT_PASS(ReplaceSeparator, stmt);
      stmt = NEXT_PASS(RewriteMultiValueFunc, stmt);
      Map<Tensor, Tensor> replace;
      RenameBinds(*binds_0, config, *args, *arg_list_0, replace);
      stmt = NEXT_PASS(RenameRealize, stmt, *binds_0, replace);

      Array<NodeRef> arg_list_1;
      Map<Tensor, Buffer> binds_1;
      GetFlattenedBinds(*args, *binds_0, config, arg_list_1, binds_1, false);
      Stmt stmt1 = NEXT_PASS(ElementwiseFlatten, stmt, *binds_0, binds_1);
      if (stmt1.get() != stmt.get()) {
        stmt = stmt1;
        *arg_list_0 = arg_list_1;
        *binds_0 = binds_1;
      }
      if (global_attrs.GetBoolAttr(kEnableFuseAxis, false)) {
        Array<NodeRef> fuse_axis_res = NEXT_PASS(FuseAxis, stmt, *arg_list_0, *binds_0);
        CHECK_EQ(fuse_axis_res.size(), 3);
        stmt = air::Downcast<Stmt>(fuse_axis_res[0]);
        *arg_list_0 = air::Downcast<Array<NodeRef>>(fuse_axis_res[1]);
        *binds_0 = air::Downcast<Map<Tensor, Buffer>>(fuse_axis_res[2]);
      }
      PassMgr::SetArgs(*arg_list_0);

      int level = global_attrs.GetIntAttr(kHelpTiling, -1);
      if (tuning || level > help_tiling_level["None"]) {
        if (tuning) {
          level = help_tiling_level["Tuning"];
        }
        Map<std::string, NodeRef> attrs_1 = global_attrs;
        attrs_1.Set(kDumpTuningLevel, air::make_const(Int(32), level));
        NodeRef tuning_spaces = NEXT_PASS(GenTuningSpace, stmt, target, *binds_0, attrs_1, false, new_sch);
        return tuning_spaces;
      }

      Array<NodeRef> poly_res = NEXT_PASS(AutoPoly, stmt, *binds_0, target, global_attrs, false, false, new_sch);
      CHECK_EQ(poly_res.size(), 2);
      stmt = air::Downcast<Stmt>(poly_res[0]);
      global_attrs.Set(kEnablePolySch, air::make_const(Int(32), true));
    } else {
      global_attrs.Set(kEnablePolySch, air::make_const(Int(32), false));
    }
    // Phase 1
    stmt = NEXT_PASS(RemoveFakeOp, stmt);
    stmt = NEXT_PASS(RewriteForTensorCore, stmt, new_sch, *binds_0);
    stmt = NEXT_PASS(StorageFlatten, stmt, *binds_0, 64, config->instrument_bound_checkers);
    stmt = NEXT_PASS(CanonicalSimplify, stmt);

    // Phase 2
    if (!simple_mode) {
      stmt = NEXT_PASS(LoopPartition, stmt, config->partition_const_loop);
    }
    if (config->disable_vectorize) {
      stmt = NEXT_PASS(SkipVectorize, stmt);
    } else {
      stmt = NEXT_PASS(VectorizeLoop, stmt);
    }
    stmt = NEXT_PASS(InjectVirtualThread, stmt);
    if (polyhedral && (global_attrs.GetBoolAttr(kEnableDoubleBuffer, false) 
      || global_attrs.GetBoolAttr(kEnableTransferBuffer, false))) {
      stmt = NEXT_PASS(InjectTransferBufferScope, stmt);
    }
    stmt = NEXT_PASS(InjectDoubleBuffer, stmt, config->double_buffer_split_loop, 
      global_attrs.GetBoolAttr(kEnableTransferBuffer, false));
    stmt = NEXT_PASS(StorageRewrite, stmt);

    if (target_platform->device_type == kDLGPU && polyhedral) {
      if (global_attrs.GetBoolAttr(kEnableSwizzleGPU, true)) {
        stmt = NEXT_PASS(SwizzleGPU, stmt, global_attrs);
      }
    }

    stmt = NEXT_PASS(UnrollLoop, stmt, config->auto_unroll_max_step, config->auto_unroll_max_depth,
                     config->auto_unroll_max_extent, config->unroll_explicit);

    // Phase 3
    stmt = NEXT_PASS(Simplify, stmt);
    stmt = NEXT_PASS(RemoveNoOp, stmt);
    if (config->instrument_bound_checkers) {
      stmt = NEXT_PASS(InstrumentBoundCheckers, stmt);
    }
    if (!config->disable_select_rewriting) {
      stmt = NEXT_PASS(RewriteUnsafeSelect, stmt);
    }
    if (BuildConfig::Current()->detect_global_barrier) {
      stmt = NEXT_PASS(ThreadSyncStmt, stmt, "global");
    }
    if (!global_attrs.GetBoolAttr(kEnablePolySch, false)) {
      stmt = NEXT_PASS(ThreadSyncStmt, stmt, "shared");
    }
    stmt = NEXT_PASS(ThreadSyncStmt, stmt, "warp");
    stmt = NEXT_PASS(InferFragmentStmt, stmt);
    stmt = NEXT_PASS(LowerThreadAllreduceStmt, stmt, target_platform->thread_warp_size);

    if (simple_mode) {
      return stmt;
    }
  }
  return stmt;
}
NodeRef LowerFunc(Stmt &stmt, const std::string &name, const BuildConfig &config, const Array<NodeRef> &all_args) {
  PassMgr::ClearPassId();
  // dump lowerfunc
  DumpIr(name + "_1", config, false);
  LoweredFunc lowered_func = NEXT_PASS(MakeAPI, stmt, name, all_args, 0, config->restricted_func);
  return lowered_func;
}
NodeRef Lower(Schedule sch, const Array<NodeRef> &in_args, const Array<NodeRef> &shape_vars, const std::string &name,
              const Map<Tensor, Buffer> &in_binds, const Map<std::string, NodeRef> &in_attrs, bool simple_mode,
              bool polyhedral, bool tuning, const std::string &target, const BuildConfig &config) {
  Array<NodeRef> args;
  Array<NodeRef> arg_list_0;
  Map<Tensor, Buffer> binds;
  Map<Tensor, Buffer> binds_0;
  NodeRef tmp = LowerStmt(sch, in_args, shape_vars, name, in_binds, in_attrs, simple_mode, polyhedral, tuning, target,
                          config, &args, &arg_list_0, &binds, &binds_0);
#ifdef USE_AKG_COMPILE_STUB
  CHECK(target != "cce") << "Can not enable target cce, please make sure akg/build/libakg_ext.a "
                            "downloaded successfully, then recompile the source codes";
  if (tuning || global_attrs.GetIntAttr(kHelpTiling, -1) > help_tiling_level["None"]) {
    return tmp;
  }
  Stmt stmt = Downcast<Stmt>(tmp);

  NodeRef lowered_func = LowerFunc(stmt, name, config, arg_list_0);
  return lowered_func;
#else
  Stmt stmt = Downcast<Stmt>(tmp);
  return LowerAscend(stmt, args, arg_list_0, binds, binds_0, shape_vars, name, simple_mode, polyhedral, tuning, target,
                     config);
#endif
}

void BuildForDevice(const Array<LoweredFunc> &flist, const std::string &target_name,
                    const std::string &target_host_name, Array<LoweredFunc> *out_flist,
                    air::runtime::Module *out_mdev) {
  CHECK(out_flist != nullptr) << "out_flist is nullptr.";
  CHECK(out_mdev != nullptr) << "out_mdev is nullptr.";

  Target target = Target::Create(target_name);
  DLDeviceType device_type = DLDeviceType::kDLCce;
  if (target->device_type == DLDeviceType::kDLGPU) {
    device_type = DLDeviceType::kDLGPU;
  }

  Array<LoweredFunc> fhost;
  Array<LoweredFunc> fdevice;
  for (auto func : flist) {
    if (func->func_type == air::LoweredFuncType::kMixedFunc) {
      Array<LoweredFunc> fsplits = NEXT_PASS(SplitHostDevice, func);
      fhost.push_back(fsplits[0]);
      for (size_t idx = 1; idx < fsplits.size(); idx++) {
        fdevice.push_back(fsplits[idx]);
      }
    } else if (func->func_type == air::LoweredFuncType::kHostFunc) {
      fhost.push_back(func);
    } else if (func->func_type == air::LoweredFuncType::kDeviceFunc) {
      fdevice.push_back(func);
    } else {
      LOG(FATAL) << "unknown function type " << func->func_type;
    }
  }

  if (target_name == "cuda") {
    for (size_t i = 0; i < fdevice.size(); ++i) {
      fdevice.Set(i, NEXT_PASS(LowerWarpMemory, fdevice[i], target->thread_warp_size));
    }
  }

  for (size_t i = 0; i < fhost.size(); ++i) {
    fhost.Set(i, NEXT_PASS(BindDeviceType, fhost[i], static_cast<int>(device_type)));
    fhost.Set(i, NEXT_PASS(LowerTVMBuiltin, fhost[i]));
  }

  Target target_host = Target::Create(target_host_name);

  for (size_t i = 0; i < fdevice.size(); ++i) {
    if (target_name == "cuda") {
      fdevice.Set(i, NEXT_PASS(LowerDeviceStorageAccessInfo, fdevice[i]));
    }
    fdevice.Set(i, NEXT_PASS(LowerIntrin, fdevice[i], target->target_name));
  }

  for (size_t i = 0; i < fhost.size(); ++i) {
    if (target_name == "cuda") {
      fhost.Set(i, NEXT_PASS(LowerDeviceStorageAccessInfo, fhost[i]));
    }
    fhost.Set(i, NEXT_PASS(LowerIntrin, fhost[i], target_host->target_name));
    fhost.Set(i, NEXT_PASS(CombineContextCall, fhost[i]));
  }

  for (const auto &func : fhost) {
    out_flist->push_back(func);
  }
  *out_mdev = air::codegen::Build(fdevice, target_name, g_external_call_name);
  return;
}

BuildRst BuildRstNode::make(const NodeRef &rst, const std::string &kernel_name) {
  NodePtr<BuildRstNode> node = make_node<BuildRstNode>();

  node->rst = rst;
  node->kernel_name = kernel_name;

  return BuildRst(node);
}

TVM_REGISTER_NODE_TYPE(BuildRstNode);

BuildRst BuildToFunc(const Schedule &inputs, const Array<NodeRef> &in_args, const Array<NodeRef> &shape_vars,
                     const std::string &name, const Map<Tensor, Buffer> &in_binds,
                     const Map<std::string, NodeRef> &in_attrs, bool polyhedral, const std::string &target,
                     const BuildConfig &config) {
  CHECK(inputs.defined()) << "inputs is not defined.";
  CHECK(!name.empty()) << "name is empty.";

  Array<NodeRef> args;
  if (in_args.defined()) {
    args = in_args;
  }
  Map<Tensor, Buffer> binds;
  if (in_binds.defined()) {
    binds = in_binds;
  }
  Map<std::string, NodeRef> attrs;
  if (in_attrs.defined()) {
    attrs = in_attrs;
  }

  auto rst = Lower(inputs, args, shape_vars, name, binds, attrs, false, polyhedral, false, target, config);
  return BuildRstNode::make(rst, name);
}

namespace {
void CreateCode(const std::string &code, const std::string &kernel_name, const std::string &target_name) {
  std::string file_path;
  std::string file_suffix;
  if (target_name.find("cce") != std::string::npos) {
    file_path = std::string(kMsAscendKernelPath);
    file_suffix = ".cce";
  } else if (target_name.find("cuda") != std::string::npos) {
    file_path = std::string(kMsGpuKernelPath) + "_" + std::to_string(getpid()) + "/";
    file_suffix = ".cu";
  }

  if (file_path.empty()) {
    return;
  }

  // Dump code to meta directory if it exists.
  struct stat info;
  if (stat(file_path.c_str(), &info) == 0) {
    if (info.st_mode & S_IFDIR) {
      std::string file_name = file_path + kernel_name + file_suffix;
      std::ofstream of(file_name);
      CHECK(of.is_open()) << "Failed to open " << file_name << " to dump code.";
      of << code << std::endl;
      of.close();
    }
  }
}
}  // namespace

air::runtime::Module BuildToModule(const NodeRef &ref, const std::string &target_name) {
  CHECK(!target_name.empty()) << "target_name is empty.";

  auto build_rst = Downcast<BuildRst>(ref);
  auto res = build_rst->rst;

  Array<LoweredFunc> lowered_func_list;
  if (res->IsInstance<LoweredFuncNode>()) {
    LoweredFunc lowered_func = air::Downcast<LoweredFunc>(res);
    lowered_func_list.push_back(lowered_func);
  }
  if (lowered_func_list.empty()) {
    return air::runtime::Module(nullptr);
  }

  Map<std::string, Array<LoweredFunc>> target_flist;
  target_flist.Set(target_name, lowered_func_list);

  Array<LoweredFunc> fhost_all;
  std::vector<air::runtime::Module> device_modules;

  for (auto iter : target_flist) {
    Array<LoweredFunc> out_flist;
    air::runtime::Module out_mdev;
    BuildForDevice(iter.second, iter.first, kAkgTargetHostName, &out_flist, &out_mdev);

    // Save the current lowered functions of the host and the device module.
    for (const auto &func : out_flist) {
      fhost_all.push_back(func);
    }
    device_modules.push_back(out_mdev);
  }

  // Generate a unified host module.
  air::runtime::Module mhost = air::codegen::Build(fhost_all, kAkgTargetHostName, g_external_call_name);

  // Import all modules.
  for (const auto &mdev : device_modules) {
    mhost.Import(mdev);
  }

  const char *akg_dump_code = getenv("MS_AKG_DUMP_CODE");
  if (akg_dump_code != nullptr) {
    auto mod0 = mhost->imports()[0];
    CHECK(mod0.defined());

    CreateCode(mod0->GetSource(), build_rst->kernel_name, target_name);
  }

  return mhost;
}

air::runtime::Module BuildModule(const Schedule &inputs, const Array<NodeRef> &in_args,
                                 const Array<NodeRef> &shape_vars, const std::string &target_name,
                                 const std::string &name, const Map<Tensor, Buffer> &in_binds,
                                 const Map<std::string, NodeRef> &in_attrs, bool polyhedral, const std::string &target,
                                 const BuildConfig &config) {
  auto func = BuildToFunc(inputs, in_args, shape_vars, name, in_binds, in_attrs, polyhedral, target, config);

  return BuildToModule(func, target_name);
}

TVM_REGISTER_API("_BuildModule").set_body_typed(BuildModule);
TVM_REGISTER_API("_BuildToFunc").set_body_typed(BuildToFunc);
TVM_REGISTER_API("_BuildToModule").set_body([](const TVMArgs &args, TVMRetValue *ret) {
  if (args.size() == 1) {
    *ret = BuildToModule(args[0]);
  } else if (args.size() == 2) {
    *ret = BuildToModule(args[0], args[1]);
  } else {
    LOG(FATAL) << "arg num must be 1 or 2, but given " << args.size();
  }
});

TVM_REGISTER_API("_Lower").set_body([](const TVMArgs &args, TVMRetValue *ret) {
  if (args.size() == 11) {
    NodeRef lowered_func =
      Lower(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]);
    *ret = lowered_func;
  }
});

TVM_REGISTER_API("akg.build_module.get_binds").set_body([](const TVMArgs &args, TVMRetValue *ret) {
  auto config = BuildConfig::Current();
  Array<NodeRef> inputs;
  Map<Tensor, Buffer> binds;
  if (args.size() >= 1) {
    inputs = args[0];
  } else if (args.size() >= 2) {
    inputs = args[0];
    binds = args[1];
  }

  Array<NodeRef> out_inputs;
  Map<Tensor, Buffer> out_binds;

  GetBinds(inputs, binds, config, &out_inputs, &out_binds);
  *ret = Array<NodeRef>{out_binds, out_inputs};
});
}  // namespace akg
