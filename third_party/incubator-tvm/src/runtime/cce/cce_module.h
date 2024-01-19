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
 * \file cce_module.h
 * \brief Execution handling of cce kernels
 */

/*!
 * 2019.12.30 - Add file cce_module.h.
 * 2024.1.24 - Change rt*** to aclrt***.
 */

#ifndef TVM_RUNTIME_CCE_CCE_MODULE_H_
#define TVM_RUNTIME_CCE_CCE_MODULE_H_

#include <tvm/runtime/module.h>
#include <runtime/meta_data.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
namespace air {
namespace runtime {
/*! \brief Maximum number of devices supported in CceModule */
static constexpr const int kMaxNumDevices = 32;

/*!
 * \brief create a cce module from data.
 *
 * \param data The module data, can be ccebin
 * \param fmt The format of the data, can be "ccebin"
 * \param fmap The map function information map of each function.
 * \param cce_source Optional, cce source file
 */
air::runtime::Module CceModuleCreate(std::string data, std::string fmt,
                                     std::unordered_map<std::string, air::runtime::FunctionInfo> fmap,
                                     std::string cce_source);
}  // namespace runtime
}  // namespace air
#endif  // TVM_RUNTIME_CCE_CCE_MODULE_H_
