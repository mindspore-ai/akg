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
 *  \file symbols_wrapper.h
 *  \brief symbols wrapper
 */

/*!
 * 2023.4.21 - Add file symbols_wrapper.h.
 */

#ifndef TVM_RUNTIME_SYMBOLS_WRAPPER_H_
#define TVM_RUNTIME_SYMBOLS_WRAPPER_H_

#include <dmlc/logging.h>

namespace air {
namespace runtime {


// define function type.
#define DEFINE_FUNC_PTR(func) func##Func func = nullptr

// load function ptr use dlopen and dlsym.
#define LOAD_FUNCTION_PTR(func_name)                                                        \
  func_name = reinterpret_cast<func_name##Func>(dlsym(handle_ptr, #func_name));             \
  if (func_name == nullptr) {                                                               \
    LOG(ERROR) << "load func (" << #func_name << ") from (" << library_path << ") failed!"; \
    UnLoadLibraries();                                                                      \
    return false;                                                                           \
  }

class SymbolsWrapper {
 public:
  static SymbolsWrapper *GetInstance();
  SymbolsWrapper() {}
  virtual ~SymbolsWrapper() {}
  virtual bool LoadLibraries() = 0;
  virtual bool UnLoadLibraries() = 0;

 private:
  static std::shared_ptr<SymbolsWrapper> symbols_wrapper_singleton_;
};
}  // namespace runtime
}  // namespace air

#endif  // TVM_RUNTIME_CCE_CCE_WRAPPER_H_
