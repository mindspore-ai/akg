/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "tvm.h"
#include "pass/utils.h"
#include <tvm/target_info.h>
#include "../runtime/thread_storage_scope.h"

namespace akg {
namespace ir {

using air::runtime::StorageScope;
using air::runtime::ThreadScope;

class CheckBoundTensors : public IRVisitor {
 public:
  explicit CheckBoundTensors(Map<Tensor, Buffer> extern_buffer) {
    for (auto buf : extern_buffer) {
      orig_binds_.emplace(buf.first);
    }
  }

  Array<Tensor> out_of_bounds_tensors_;
 private:
  void Visit_(const AttrStmt* op) final {
    if (op->attr_key == air::ir::attr::realize_scope) {
      storage_scope_[op->node.get()] = op->value.as<StringImm>()->value;
      return IRVisitor::Visit(op->body);
    }  else if (op->attr_key == air::ir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ThreadScope ts = ThreadScope::make(iv->thread_tag);
      thread_scope_.push_back(ts);
      IRVisitor::Visit_(op);
      thread_scope_.pop_back();
      return ;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Realize* op) final {
    auto cur_tensor = Downcast<Operation>(op->func).output(op->value_index);
    if (orig_binds_.count(cur_tensor) > 0) {
      return IRVisitor::Visit_(op);
    }

    Array<Expr> shape;
    for (auto bound : op->bounds) {
      shape.push_back(bound->extent);
    }

    auto it = storage_scope_.find(op->func.get());
    if (it != storage_scope_.end()) {
      StorageScope skey;
      const std::string& strkey = it->second;
      if (strkey.length() == 0) {
        if (thread_scope_.size() != 0) {
          skey.rank = air::runtime::DefaultStorageRank(thread_scope_.back().rank);
        }
      } else {
        skey = StorageScope::make(strkey);
      }

      int32_t const_size = Allocate::constant_allocation_size(shape);
      if (skey.tag.length() != 0) {
        air::MemoryInfo info = air::GetMemoryInfo(skey.to_string());
        if (info.defined() && const_size * op->type.bits() > info->max_num_bits) {
          std::string tensor_name = op->func->func_name();
          std::string separator_str = "_local_";
          auto end = tensor_name.find(separator_str);
          if (end != std::string::npos) {
            tensor_name = tensor_name.substr(0, end);
          }

          auto new_tensor = placeholder(shape, op->type, tensor_name);
          out_of_bounds_tensors_.push_back(new_tensor);
        }
      }
    }
    IRVisitor::Visit_(op);
  }

 private:
  std::unordered_set<Tensor> orig_binds_;
  std::unordered_map<const Node*, std::string> storage_scope_;
  std::vector<ThreadScope> thread_scope_;
};

Array<NodeRef> CheckBoundTensor(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer) {
  CheckBoundTensors check_bound_tensors(extern_buffer);
  check_bound_tensors.Visit(stmt);
  return Array<NodeRef>({stmt, check_bound_tensors.out_of_bounds_tensors_});
}

}  // namespace ir
}  // namespace akg
