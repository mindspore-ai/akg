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

#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm.h>

namespace akg {
namespace ir {
class ExternalCallCollector : public IRVisitor {
 public:
  Array<NodeRef> Collect(Stmt stmt) {
    this->Visit(stmt);
    return external_call_names_;
  }

  void Visit_(const Call *op) final {
    if (op->call_type == Call::CallType::Extern && op->name.find("FL") != std::string::npos) {
      bool dup = false;
      for (auto node : external_call_names_) {
        if (node.as<StringImm>()->value == op->name) {
          dup = true;
          break;
        }
      }
      if (!dup) external_call_names_.push_back(air::ir::StringImm::make(op->name));
    }
    IRVisitor::Visit_(op);
  }

 private:
  Array<NodeRef> external_call_names_;
};

Array<NodeRef> CollectExternalCall(const Stmt &stmt) {
  return Array<NodeRef>({stmt, ExternalCallCollector().Collect(stmt)});
}
}  // namespace ir
}  // namespace akg
