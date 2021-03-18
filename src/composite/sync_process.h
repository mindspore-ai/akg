/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef AKG_SRC_COMPOSITE_SYNC_PROCESS_H_
#define AKG_SRC_COMPOSITE_SYNC_PROCESS_H_
#include <functional>
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "tvm.h"

namespace akg {
namespace ir {
class EvaluateVisitor : public IRVisitor {
 public:
  EvaluateVisitor();
  EvaluateVisitor(const std::function<bool(const Call *op)> &targe,
                  const std::vector<std::function<bool(const Call *op)>> &blacklist)
      : target_call_func_(targe), blacklist_call_funcs_(blacklist) {}
  ~EvaluateVisitor() = default;
  std::pair<bool, bool> Run(const Stmt &stmt);
  void Visit_(const Call *op) override;
  void Visit_(const Evaluate *op) override;

 private:
  std::function<bool(const Call *op)> target_call_func_;
  std::vector<std::function<bool(const Call *op)>> blacklist_call_funcs_;
  bool in_evaluate_{false};
  bool target_hit_{false};
  bool blacklist_hit_{false};
};

Stmt ProcessSyncInnerThread(const Stmt &stmt);
}  // namespace ir
}  // namespace akg
#endif  // AKG_SRC_COMPOSITE_SYNC_PROCESS_H_
