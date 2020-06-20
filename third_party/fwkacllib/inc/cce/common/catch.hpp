/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef CATCH_HPP_
#define CATCH_HPP_

#include <stdint.h>
#include <iostream>

#define ERROR_CODE() __catch_error_code
#define ERROR_LINE_NO() __catch_error_line_no
#define ERROR_PROC() __catch_error_line_no = __LINE__;

#define PROC                                   \
  uint32_t __catch_error_code = 0x7FFFFFCC;    \
  uint32_t __catch_error_line_no = 0xFFFFFFFF; \
  {
#define END_PROC \
  }              \
  __tabErrorCode:
#define THROW(errcode)              \
  {                                 \
    __catch_error_code = (errcode); \
    ERROR_PROC();                   \
    goto __tabErrorCode;            \
  }
#define EXEC(func)                                                    \
  {                                                                   \
    if (0 != (__catch_error_code = (func))) THROW(__catch_error_code) \
  }
#define EXEC_EX1(func, error_code)     \
  {                                    \
    if (0 != (func)) THROW(error_code) \
  }
#define EXEC_EX(func, succRet, error_code)                          \
  {                                                                 \
    if (succRet != (__catch_error_code = (func))) THROW(error_code) \
  }
#define ASSERT_EXEC(func, succRet)                                       \
  {                                                                      \
    if (succRet != (__catch_error_code = (func))) /*GO_ASSERT_FALSE();*/ \
      THROW(__catch_error_code)                                          \
  }                                                                      \
  }
#define NEW_ERROR_EXEC(errcode, func, succRet) \
  {                                            \
    if (succRet != (func)) {                   \
      THROW(errcode)                           \
    }                                          \
  }
#define JUDGE(errcode, expr) \
  {                          \
    if (!(expr)) {           \
      THROW(errcode)         \
    }                        \
  }
#define ASSERT_JUDGE(errcode, expr)       \
  {                                       \
    if (!(expr)) { /*GO_ASSERT_FALSE();*/ \
      THROW(errcode)                      \
    }                                     \
  }
#define JUDGE_FALSE(errcode, expr) \
  {                                \
    if (expr) {                    \
      THROW(errcode)               \
    }                              \
  }
#define JUDGE_CONTINUE(expr) \
  {                          \
    if (expr) {              \
      continue;              \
    }                        \
  }
#define CATCH_ERROR(errcode) if (__catch_error_code == (errcode)) {  // ERROR_LOG();
#define CATCH_ALL_ERROR {
#define END_CATCH_ERROR }
#define FINAL \
  __tabFinal:
#define END_FINAL /*GO_ASSERT_FALSE()*/ ;
#define GOTO_FINAL() goto __tabFinal;
#endif  // CATCH_HPP_
