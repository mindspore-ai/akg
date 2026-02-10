/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MFUSION_SUPPORT_LOGGING_H
#define MFUSION_SUPPORT_LOGGING_H

#include <cerrno>
#include <cstdlib>

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

// Environment variable to control minimum log level: MLOG_LEVEL
// 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR
// Only messages at level >= MLOG_LEVEL are printed.
// Example: export MLOG_LEVEL=0   -> print DEBUG and above
//          export MLOG_LEVEL=1   -> print INFO and above (default)
//          export MLOG_LEVEL=3   -> print ERROR only
// When unset, default is 1 (INFO).

// Helper macro to convert symbol to string
#define MLOG_LEVEL_TO_STRING(LEVEL) #LEVEL

// Helper class to automatically add newline at the end
struct MLogStream {
  llvm::raw_ostream &os;
  MLogStream(llvm::raw_ostream &stream) : os(stream) {}
  ~MLogStream() { os << "\n"; }
  template<typename T>
  MLogStream &operator<<(const T &value) {
    os << value;
    return *this;
  }
};

// Log level constants aligned with LLVM's logging conventions.
// DEBUG uses llvm::errs() so that MLOG_LEVEL=0 always prints (llvm::dbgs() is
// gated by LLVM debug flags and would not show when run from Python without -debug).
// - DEBUG: llvm::errs()
// - INFO: llvm::outs()
// - WARNING/ERROR: llvm::errs() (WithColor for WARNING)
namespace MLog {
  enum Level {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
  };

  // Returns minimum log level from environment MLOG_LEVEL (0-3).
  // Default 1 (INFO) when unset or invalid.
  inline int getMinLogLevel() {
    const char *p = std::getenv("MLOG_LEVEL");
    if (!p || *p == '\0') {
      return 1;
    }
    errno = 0;
    char *end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (errno != 0 || end == p || *end != '\0') {
      return 1;
    }
    if (v < 0 || v > 3) {
      return 1;
    }
    return static_cast<int>(v);
  }

  // Returns true if messages at this level should be printed.
  inline bool shouldLog(Level level) {
    return static_cast<int>(level) >= getMinLogLevel();
  }

  // Internal helper: stream for each level. DEBUG uses errs() so output is
  // visible when MLOG_LEVEL=0 without requiring LLVM -debug.
  // WARNING uses errs() to avoid returning a reference to a temporary WithColor (UB).
  inline llvm::raw_ostream &getStream(Level level) {
    switch (level) {
      case INFO:
        return llvm::outs();
      case DEBUG:
      case WARNING:
      case ERROR:
      default:
        return llvm::errs();
    }
  }
}

// Define convenience constants for backward compatibility
// These allow MLOG(INFO), MLOG(DEBUG), MLOG(WARNING), MLOG(ERROR) syntax
// Using static constexpr in anonymous namespace to avoid ODR violations
namespace {
  static constexpr MLog::Level INFO = MLog::INFO;
  static constexpr MLog::Level DEBUG = MLog::DEBUG;
  static constexpr MLog::Level WARNING = MLog::WARNING;
  static constexpr MLog::Level ERROR = MLog::ERROR;
}

// Main logging macro with file, function, line information, and log level.
// Whether a level is printed is controlled by environment variable MLOG_LEVEL (0-3).
// Usage: MLOG(INFO) << "message here";
//        MLOG(DEBUG) << "debug message";
//        MLOG(WARNING) << "warning message";
//        MLOG(ERROR) << "error message";
// Note: Newline is automatically added at the end, no need to add "\n" manually
#define MLOG(LEVEL) \
  (MLog::shouldLog(LEVEL) ? \
    (MLogStream(MLog::getStream(LEVEL)) << "[" << MLOG_LEVEL_TO_STRING(LEVEL) << "][" << __FILE__ << ":" << __LINE__ << ":" << __func__ << "] ") : \
    MLogStream(llvm::nulls()))

// Alternative DEBUG macro using LLVM_DEBUG (per-pass switch via DEBUG_TYPE and LLVM_DEBUG).
// Use MLOG(DEBUG) with MLOG_LEVEL=0 for simple global debug; use MLOG_DEBUG for pass-specific debug.
#define MLOG_DEBUG \
  LLVM_DEBUG(llvm::dbgs() << "[" << MLOG_LEVEL_TO_STRING(DEBUG) << "][" << __FILE__ << ":" << __LINE__ << ":" << __func__ << "] ")

// Convenience macros for each log level (aligned with LLVM conventions)
#define MLOG_INFO MLOG(MLog::INFO)
#define MLOG_WARNING MLOG(MLog::WARNING)
#define MLOG_ERROR MLOG(MLog::ERROR)

#endif // MFUSION_SUPPORT_LOGGING_H

