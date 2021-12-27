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
#ifndef COMPOSITE_EMITTER_H_
#define COMPOSITE_EMITTER_H_

#include "composite/utils/util.h"

namespace akg {
class Emitter : public IRVisitor {
 public:
  explicit Emitter(BuildOpt &opt) : opt_(opt) {}

 private:
  void Visit_(const AttrStmt *op) override;
  void Visit_(const Provide *op) override;
  void ProcessTupleGetItem(const Provide *op, const Array<NodeRef> &inputs);
  Array<NodeRef> GetRealInputs(const Array<Expr> &inputs);
  void EmitTopi(const Provide *op, const Array<NodeRef> &real_inputs);
  const PackedFunc *GetTopiFunc();
  void EmitAssign(Tensor &t, const Expr &input);
  void CollectNoinlineCandidate(const Array<NodeRef> &real_inputs, const Tensor &t);

 private:
  BuildOpt &opt_;
  std::string op_name_;
  Map<std::string, NodeRef> op_attrs_;
  std::map<FunctionRef, Array<Tensor>> array_result_;
  int assign_count_{0};
};

}  // namespace akg
#endif  // COMPOSITE_EMITTER_H_
