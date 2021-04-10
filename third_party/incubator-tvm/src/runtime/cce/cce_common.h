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
 */

#ifndef TVM_RUNTIME_CCE_CCE_COMMON_H_
#define TVM_RUNTIME_CCE_CCE_COMMON_H_

#include <tvm/runtime/packed_func.h>
#include <runtime/rt.h>
#include <runtime/workspace_pool.h>
#include <string>

namespace air {
namespace runtime {
inline const char* CceGetErrorString(rtError_t e) {
  switch (e) {
    case RT_ERROR_NONE:
      return "success";
    default:
      return "Unknow cce error code";
  }
}

#define CCE_CALL(func)                                                                                  \
  {                                                                                                     \
    rtError_t e = (func);                                                                               \
    CHECK(e == RT_ERROR_NONE) << "Cce runtime error: errno=" << e << ", info=" << CceGetErrorString(e); \
  }

/*! \brief Thread local workspace */
class CceThreadEntry {
 public:
  /*! \brief The cce stream */
  rtStream_t stream{nullptr};
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

#ifndef RT_KERNEL_LAUNCH_PARAMETRIC_SHAPE
#define RT_KERNEL_LAUNCH_PARAMETRIC_SHAPE
/**
 * @ingroup kt_kernel
 * @brief launch kernel to device
 * @param [in] stubFunc  stub function
 * @param [in] blockDim  block dimentions
 * @param [in] args  tensor arguments 64-bit address for kernel function
 * @param [in] argsSize  tensor arguments size
 * @param [in] shapes  64-bit integer arguments for kernel function 
 * @param [in] shapeSize  integer arguments size
 * @param [in] smDesc  shared memory description
 * @param [in] stream  associated stream
 * @return RT_ERROR_NONE  for ok, errno for failed
 */
#ifdef USE_KC_AIR
extern "C" rtError_t rtKernelLaunchShapes(const void *stubFunc,
					  uint32_t blockDIm,
					  void *args,
					  uint32_t argsSize,
					  int64_t *shapes,
					  uint32_t shapeSize,
					  rtSmDesc_t *smDesc,
					  rtStream_t stream);
#endif // USE_KC_AIR
#endif  // RT_KERNEL_LAUNCH_PARAMETRIC_SHAPE


}  // namespace runtime
}  // namespace air
#endif  // TVM_RUNTIME_CCE_CCE_COMMON_H_
