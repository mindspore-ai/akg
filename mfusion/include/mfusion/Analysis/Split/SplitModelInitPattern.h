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

#ifndef MFUSION_ANALYSIS_SPLIT_SPLITMODELINITPATTERN_H
#define MFUSION_ANALYSIS_SPLIT_SPLITMODELINITPATTERN_H

#include "mfusion/Analysis/Split/SplitModel.h"

namespace mlir {
namespace mfuse {
namespace split {

class DVMSplitModel : public SplitModel {
 public:
  DVMSplitModel() = default;
  ~DVMSplitModel() = default;

 protected:
  void initFusePatterns() override;
};

class AKGSplitModel : public SplitModel {
 public:
  AKGSplitModel() = default;
  ~AKGSplitModel() = default;

 protected:
  void initFusePatterns() override;
};

class BishengSplitModel : public SplitModel {
 public:
  BishengSplitModel() = default;
  ~BishengSplitModel() = default;

 protected:
  void initFusePatterns() override;
};

}  // namespace split
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_SPLIT_SPLITMODELINITPATTERN_H
