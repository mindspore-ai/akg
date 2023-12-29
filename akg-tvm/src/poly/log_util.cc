/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "poly/log_util.h"

#include <dmlc/logging.h>

namespace akg {
namespace ir {
namespace poly {
namespace log {

////////////////////////////////////////////////////////////////////////////////
// Verbosity levels
////////////////////////////////////////////////////////////////////////////////

static Verbosity akg_poly_verbosity_level = Verbosity::silent;

Verbosity GetVerbosityLevel(void) { return akg_poly_verbosity_level; }

void SetVerbosityLevel(Verbosity level) { akg_poly_verbosity_level = level; }

////////////////////////////////////////////////////////////////////////////////
// Logging functions
////////////////////////////////////////////////////////////////////////////////

void Warn(const std::string &message) { LOG(WARNING) << text_yellow << message << text_reset; }

// Our errors should not be fatal so we still log as a warning.
void Error(const std::string &message) { LOG(WARNING) << text_red << message << text_reset; }

void Info(const std::string &message) { LOG(INFO) << message << text_reset; }

void Warn(const std::stringstream &stream) {
  const std::string &message = stream.str();
  Warn(message);
}

void Error(const std::stringstream &stream) {
  const std::string &message = stream.str();
  Error(message);
}

void Info(const std::stringstream &stream) {
  const std::string &message = stream.str();
  Info(message);
}

// clang-format off
#define _define_logging_wrappers(func) \
  void func(const Verbosity level, const std::string &message) { \
    if (akg_poly_verbosity_level >= level) { \
      func(message); \
    } \
  } \
  void func(const Verbosity level, const std::stringstream &stream) { \
    if (akg_poly_verbosity_level >= level) { \
      func(stream); \
    } \
  } \
  void func(const int level, const std::string &message) { \
    func(static_cast<Verbosity>(level), message); \
  } \
  void func(const int level, const std::stringstream &stream) { \
    func(static_cast<Verbosity>(level), stream); \
  }

_define_logging_wrappers(Info)
_define_logging_wrappers(Warn)
_define_logging_wrappers(Error)

#undef _declare_logging_functions_int_level
// clang-format on

}  // namespace log
}  // namespace poly
}  // namespace ir
}  // namespace akg
