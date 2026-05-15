/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MFUSION_ANALYSIS_SPLIT_SPLITMODELFACTORY_H
#define MFUSION_ANALYSIS_SPLIT_SPLITMODELFACTORY_H

#include <memory>
#include <string>
#include <functional>
#include <unordered_map>

#include "mfusion/Analysis/Split/SplitModel.h"

namespace mlir {
namespace mfuse {
namespace split {

class SplitModelFactory {
 public:
  static SplitModelFactory &Instance() {
    static SplitModelFactory instance;
    return instance;
  }

  std::shared_ptr<SplitModel> createSplitModel(const std::string &processor);

  using RegFunc = std::function<std::shared_ptr<SplitModel>()>;
  void Register(const std::string &processor, const RegFunc &func) { creators_[processor] = func; }

 private:
  SplitModelFactory() = default;
  ~SplitModelFactory() = default;
  SplitModelFactory(const SplitModelFactory &) = delete;
  SplitModelFactory &operator=(const SplitModelFactory &) = delete;

  std::unordered_map<std::string, RegFunc> creators_;
};

class SplitModelRegister {
 public:
  SplitModelRegister(const std::string &processor, const SplitModelFactory::RegFunc &func) : func_(func) {
    SplitModelFactory::Instance().Register(processor, func);
  }
  ~SplitModelRegister() = default;

 protected:
  SplitModelFactory::RegFunc func_;
};

#define SPLIT_MODEL_JOIN(x, y) x##y
#define SPLIT_MODEL_UNIQUE_NAME(prefix, cnt) SPLIT_MODEL_JOIN(prefix, cnt)
#define SPLIT_MODEL_REGISTER(processor, cls, ...)                                                          \
  static const mlir::mfuse::split::SplitModelRegister SPLIT_MODEL_UNIQUE_NAME(split_model_, __COUNTER__)(  \
    processor, [__VA_ARGS__]() noexcept {                                                                  \
      return std::static_pointer_cast<mlir::mfuse::split::SplitModel>(std::make_shared<cls>(__VA_ARGS__)); \
    })

}  // namespace split
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_SPLIT_SPLITMODELFACTORY_H
