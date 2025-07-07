/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <limits.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "akg/ExecutionEngine/AscendLaunchRuntime/AKGAscendLaunchRuntime.h"
#include "akg/ExecutionEngine/AscendLaunchRuntime/TensorDevice.h"

namespace {
// Run a shell commond (only support linux)
void RunCmd(const std::string &cmd) {
  if (cmd.empty()) {
    std::cerr << "RunCmd, cmd empty!" << std::endl;
    return;
  }

  std::cout << "cmd : " << cmd << std::endl;
  std::time_t start, end;
  start = time(nullptr);

  FILE *fp;
  fp = popen(cmd.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << cmd << " error, errno: " << errno << std::endl;
    return;
  }
  if (pclose(fp) == -1) {
    std::cerr << "pclose error, cmd: " << cmd << std::endl;
    return;
  }

  end = time(nullptr);
  double elapse = std::difftime(end, start);
  std::cout << "cmd execute complete, elapsed time: " << elapse << " s\n";
}

// Default split by space, results ignore empty string
std::vector<std::string> SplitString(const std::string &str, const std::string &split_str = " ") {
  std::vector<std::string> result;

  std::string::size_type post_start = 0;
  std::string::size_type post_end = str.find(split_str);
  while (post_end != std::string::npos) {
    result.push_back(str.substr(post_start, post_end - post_start));

    post_start = post_end + split_str.size();
    post_end = str.find(split_str, post_start);
  }

  if (post_start != str.length()) {
    result.push_back(str.substr(post_start));
  }

  return result;
}

void LinkSharedObject(const std::string &src_file, const std::string &dst_file) {
  std::string link_cmd =
    "ccec --cce-fatobj-link -fPIC -shared -o " + dst_file + " " + src_file + " -lascendcl -lruntime";
  const char *libray_path = std::getenv("LD_LIBRARY_PATH");
  if (libray_path == nullptr) {
    std::cerr << "LD_LIBRARY_PATH env var not found\n";
    return;
  }

  const auto link_lib_set = std::vector<std::string>{"libascendcl.so", "libruntime.so"};

  auto lib_paths = SplitString(std::string(libray_path), ":");
  for (auto lib_path : lib_paths) {
    char lib_realpath[PATH_MAX];
    char *res = realpath(lib_path.c_str(), lib_realpath);
    if (!res) {
      continue;
    }
    int ret = access(lib_realpath, F_OK);
    if (ret == -1) {
      std::cerr << "The lib_realpath could not be found!\n";
      return;
    }

    for (auto link_lib : link_lib_set) {
      std::string so_lib_path = lib_path + "/" + link_lib;
      char so_lib_realpath[PATH_MAX];
      res = realpath(so_lib_path.c_str(), so_lib_realpath);
      if (!res) {
        continue;
      }
      link_cmd += " -L" + lib_path;
    }
  }
  RunCmd(link_cmd);
}
}  // namespace

int main(int argc, char *argv[]) {
  // ascend-linker  kernel_func_name.so kernel_func_name
  if (argc != 3) {
    std::cerr << "Usage: exe kernel_func_name.so kernel_func_name" << std::endl;
    return 1;
  }
  // kenrel_name.so
  std::string kernelSoFileName(argv[2]);
  std::string path(argv[1]);
  // host_func = kernel_name + "_do";
  std::string kernel_name = kernelSoFileName.substr(0, kernelSoFileName.length() - 3);
  std::string hostKernelName = kernel_name + "_do";
  // input0 and input 1, and output;
  std::uint16_t input0[16] = {
    1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
  };
  std::uint16_t input1[16] = {
    1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
  };
  std::uint16_t output[16] = {0};

  auto input_tensors = std::vector<mlir::runtime::TensorDevicePtr>();
  auto input_shape_args = std::vector<std::vector<int64_t>>();
  auto data_ptr0 = (void *)input0;
  auto data_ptr1 = (void *)input1;
  auto data_output = (void *)output;
  auto nbytes = 16 * 2;  // 32 Bytes;
  input_tensors.push_back(std::make_shared<mlir::runtime::TensorDevice>(data_ptr0, nbytes, false));
  input_tensors.push_back(std::make_shared<mlir::runtime::TensorDevice>(data_ptr1, nbytes, false));
  input_tensors.push_back(std::make_shared<mlir::runtime::TensorDevice>(data_output, nbytes, true));
  std::uint32_t device_id = 0;
  auto kernel_runtime = mlir::runtime::AscendKernelRuntime(device_id);
  //TODO: adapter kernel launch input args
  //kernel_runtime.RunOpImpl(path, kernel_name, input_tensors, input_shape_args);
  for (int i = 0; i < 16; i++) {
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
