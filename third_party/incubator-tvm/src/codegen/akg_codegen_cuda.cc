/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * 2021.7.9 - Add new file.
 */

/*
 * \file akg_codegen_cuda.cc
 */

#include "codegen_cuda.h"

#include <tvm/base.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>
#include <cmath>
#include <vector>
#include <string>
#include "common/common_util.h"
#include <tvm/ir_pass.h>
#include "literal/cuda_half_t.h"
#include "codegen_cuda.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace air {
namespace codegen {

void CodeGenCUDA::VisitExpr_(const Mod* op, std::ostream& os) {
  if (op->type.is_float()) {
    os << "fmod" << '(';
    this->PrintExpr(op->a, os);
    os << ", ";
    this->PrintExpr(op->b, os);
    os << ')';
  } else {
    os << '(';
    this->PrintExpr(op->a, os);
    os << ' ' << '%' << ' ';
    this->PrintExpr(op->b, os);
    os << ')';
  }
}

}  // namespace codegen
}  // namespace air
