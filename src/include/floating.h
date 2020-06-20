/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef INCLUDE_AKG_FLOATING_H_
#define INCLUDE_AKG_FLOATING_H_

#include <tvm/base.h>
#include <tvm/dtype.h>
#include <tvm/node/container.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/ir.h>
#include <string>
#include <algorithm>
#include <unordered_map>

namespace akg {
/*!
 * \brief Container of constant float number (FloatImm).
 *
 * This is used to store and automate type check
 * attributes that must be constant float.
 */
class Floating : public ktvm::Expr {
 public:
  Floating() : Expr() {}

  /*!
   * \brief constructor from node.
   */
  explicit Floating(ktvm::runtime::ObjectPtr<ktvm::runtime::Object> Object) : Expr(Object) {}
  /*!
   * \brief Construct integer from int value.
   */
  Floating(double value) : Expr(value) {}  // NOLINT(*)
  /*!
   * \brief Destructor.
   */
  ~Floating() {}
  /*!
   * \brief Assign an expression to integer.
   * \param other another expression.
   */
  Floating &operator=(const Floating &other) {
    data_ = other.data_;
    return *this;
  }
  /*!
   * \brief Get pointer to the internal value.
   * \return the content of the integer.
   */
  const ktvm::ir::FloatImm *operator->() const { return static_cast<const ktvm::ir::FloatImm *>(get()); }
  /*!
   * \brief convert to double
   */
  operator double() const {
    CHECK(data_ != nullptr) << " Trying get reference a null Integer";
    return (*this)->value;
  }
  /*! \brief type indicate the container type */
  using ContainerType = ktvm::ir::FloatImm;
};
}  // namespace akg

#endif  // INCLUDE_AKG_FLOATING_H_
