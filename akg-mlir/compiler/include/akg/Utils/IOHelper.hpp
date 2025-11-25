/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef AKG_UTILS_IOHELPER_H
#define AKG_UTILS_IOHELPER_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "llvm/Support/FileUtilities.h"

namespace mlir {
class DirUtils {
 public:
  DirUtils() = default;
  static void CheckOrCreateDirectory(const std::string &file_str) {
    llvm::ErrorOr<bool> result = llvm::sys::fs::exists(file_str);
    if (result) {      // If the check was successful
      if (!*result) {  // If the directory does not exist
        std::error_code ec = llvm::sys::fs::create_directory(file_str);
        if (ec) {
          llvm::report_fatal_error(llvm::StringRef("Error creating directory: " + ec.message()));
        }
      }       // If the directory exists, there's nothing more to do
    } else {  // If the check was unsuccessful
      llvm::report_fatal_error(llvm::StringRef("Error checking if directory exists: " + result.getError().message()));
    }
  }

  static nlohmann::json checkAndReadJson(const std::string &input_file_name) {
    nlohmann::json j;
    std::ifstream jfile(input_file_name);
    if (!jfile.good()) {
      llvm::report_fatal_error(llvm::StringRef("Error occurs when converting json to mlir: json file does not exist"));
    }
    jfile >> j;
    return j;
  }
};
}  // namespace mlir
#endif  // AKG_UTILS_IOHELPER_H

