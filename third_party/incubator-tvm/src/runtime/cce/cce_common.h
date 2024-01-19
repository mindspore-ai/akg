/*!
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

/*!
 * \file cce_common.h
 * \brief Common utilties for cce
 */

/*!
 * 2019.12.30 - Add file cce_common.h.
 * 2023.4.21 - Include cce_wrapper.h.
 * 2024.1.24 - Change rt*** to aclrt***.
 */

#ifndef TVM_RUNTIME_CCE_CCE_COMMON_H_
#define TVM_RUNTIME_CCE_CCE_COMMON_H_

#include <tvm/runtime/packed_func.h>
#include <runtime/workspace_pool.h>
#include <string>
#include "cce_wrapper.h"
#include "runtime/cce/cce_acl.h"

namespace air {
namespace runtime {
inline const char* CceGetErrorString(aclError e) {
  switch (e) {
    case ACL_SUCCESS:
      return "success";
    default:
      return "Unknow cce error code";
  }
}

#define CCE_CALL(func)                                                                                  \
  {                                                                                                     \
    aclError e = (func);                                                                               \
    CHECK(e == ACL_SUCCESS) << "Cce runtime error: errno=" << e << ", info=" << CceGetErrorString(e); \
  }

/*! \brief Thread local workspace */
class CceThreadEntry {
 public:
  /*! \brief The cce stream */
  aclrtStream stream{nullptr};
  /*! \brief thread local pool */
  air::runtime::WorkspacePool pool;
  /*! \brief profiting handle */
#ifdef USE_CCE_PROFILING
  void* profcfghandle{nullptr};
#endif
  /*! \brief constructor */
  CceThreadEntry();
  /*! \brief destructor */
  ~CceThreadEntry() {}
  // get the threadlocal workspace
  static CceThreadEntry* ThreadLocal();
};


}  // namespace runtime
}  // namespace air
#endif  // TVM_RUNTIME_CCE_CCE_COMMON_H_
