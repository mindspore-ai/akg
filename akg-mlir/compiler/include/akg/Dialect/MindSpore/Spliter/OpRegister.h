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
#ifndef COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_OPREGISTER_H_
#define COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_OPREGISTER_H_

#include <functional>
#include <string>
#include <unordered_map>
#include "OpNode.h"

namespace mlir::spliter {
using CreatorFunc = std::function<PrimOpPtr(const std::string &)>;
class OpRegistry {
 public:
  static OpRegistry &getInstance() {
    static OpRegistry instance{};
    return instance;
  }
  void registerOpFunc(const std::string &opName, const CreatorFunc &func) { creators[opName] = func; }

  PrimOpPtr newOp(const std::string &op) {
    // "OpaqueOp" is registered by default.
    return creators.find(op) == creators.end() ? creators["_opaque"](op) : creators[op](op);
  }

 private:
  OpRegistry() = default;
  ~OpRegistry() = default;

  OpRegistry(const OpRegistry &) = delete;
  OpRegistry(const OpRegistry &&) = delete;
  OpRegistry &operator=(const OpRegistry &) = delete;
  OpRegistry &operator=(const OpRegistry &&) = delete;

  std::unordered_map<std::string, CreatorFunc> creators;
};
void initAllLiteOps();
}  // namespace mlir::spliter
#endif  // COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_OPREGISTER_H_
