/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef POLY_TRANSFER_STMT_H_
#define POLY_TRANSFER_STMT_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * Transfer specified statements out of outer-most band's filter nodes if they are previously
 * recorded as transfer_stmt.
 */
class TransferStmt : public SchedulePass {
 public:
  TransferStmt(ScopInfo &scop_info, PassInfo &pass_info) : scop_info_(scop_info), pass_info_(pass_info) {
    pass_name_ = __FUNCTION__;
  }
  ~TransferStmt(){};

  virtual isl::schedule Run(isl::schedule sch);

 private:
  ScopInfo &scop_info_;
  PassInfo &pass_info_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_TRANSFER_STMT_H_
