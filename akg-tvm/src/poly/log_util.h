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
#ifndef POLY_LOG_UTIL_H_
#define POLY_LOG_UTIL_H_

#include <string>
#include <sstream>

namespace akg {
namespace ir {
namespace poly {
namespace log {

////////////////////////////////////////////////////////////////////////////////
// Verbosity levels
////////////////////////////////////////////////////////////////////////////////

enum class Verbosity {
  silent = 0,
  veryLow,
  low,
  medium,
  high,
  veryHigh,
};

Verbosity GetVerbosityLevel(void);
void SetVerbosityLevel(Verbosity level);

////////////////////////////////////////////////////////////////////////////////
// Log colors
////////////////////////////////////////////////////////////////////////////////

#ifdef AKG_POLY_LOG_WITH_COLORS
#define text_reset "\033[0m"
#define text_bold "\033[1m"
#define text_dim "\033[2m"
#define text_italic "\033[3m"
#define text_underline "\033[4m"
#define text_blink "\033[5m"
#define text_rapid_blink "\033[6m"
#define text_reverse "\033[7m"
#define text_conceal "\033[8m"

#define text_black "\033[30m"
#define text_red "\033[31m"
#define text_green "\033[32m"
#define text_yellow "\033[33m"
#define text_blue "\033[34m"
#define text_magenta "\033[35m"
#define text_cyan "\033[36m"
#define text_white "\033[37m"

#define text_bright_black "\033[90m"
#define text_bright_red "\033[91m"
#define text_bright_green "\033[92m"
#define text_bright_yellow "\033[93m"
#define text_bright_blue "\033[94m"
#define text_bright_magenta "\033[95m"
#define text_bright_cyan "\033[96m"
#define text_bright_white "\033[97m"
#else
#define text_reset ""
#define text_bold ""
#define text_dim ""
#define text_italic ""
#define text_underline ""
#define text_blink ""
#define text_rapid_blink ""
#define text_reverse ""
#define text_conceal ""

#define text_black ""
#define text_red ""
#define text_green ""
#define text_yellow ""
#define text_blue ""
#define text_magenta ""
#define text_cyan ""
#define text_white ""

#define text_bright_black ""
#define text_bright_red ""
#define text_bright_green ""
#define text_bright_yellow ""
#define text_bright_blue ""
#define text_bright_magenta ""
#define text_bright_cyan ""
#define text_bright_white ""
#endif

////////////////////////////////////////////////////////////////////////////////
// Local logging functions
////////////////////////////////////////////////////////////////////////////////

void Info(const std::string &message);
void Info(const std::stringstream &stream);
void Info(const int level, const std::string &message);
void Info(const int level, const std::stringstream &stream);
void Info(const Verbosity level, const std::string &message);
void Info(const Verbosity level, const std::stringstream &stream);

void Warn(const std::string &message);
void Warn(const std::stringstream &stream);
void Warn(const int level, const std::string &message);
void Warn(const int level, const std::stringstream &stream);
void Warn(const Verbosity level, const std::string &message);
void Warn(const Verbosity level, const std::stringstream &stream);

void Error(const std::string &message);
void Error(const std::stringstream &stream);
void Error(const int level, const std::string &message);
void Error(const int level, const std::stringstream &stream);
void Error(const Verbosity level, const std::string &message);
void Error(const Verbosity level, const std::stringstream &stream);

}  // namespace log
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_LOG_UTIL_H_
