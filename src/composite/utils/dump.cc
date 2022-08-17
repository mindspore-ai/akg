/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "dump.h"

namespace akg {
void DumpHeader(std::ofstream &of, const std::string &str) {
  of << std::endl << ">>>>>>>>>> " << str << " <<<<<<<<<<" << std::endl;
}

void DumpIRAttr(const std::string &kernel_name, const IrAttrInfo &attr, size_t index) {
  if (getenv(GetDumpIRFlag().c_str()) == nullptr) return;
  std::ofstream of;
  of.open("stitch_info/" + kernel_name + "_stitch.log", std::ios::app);
  if (!of.is_open()) return;
  DumpHeader(of, "IrAttrInfo" + std::to_string(index));
  of << "dims:" << attr.dims << std::endl;
  of << "grid_dims:" << attr.grid_dims << std::endl;
  of << "block_dims:" << attr.block_dims << std::endl;
  of << "attrs:" << attr.attrs << std::endl;
  of << "broadcast_size:" << attr.broadcast_size << std::endl;
  of << "elemwise_size:" << attr.elemwise_size << std::endl;
  of.close();
}


void DumpStr2File(const std::string &file_name, const std::string &str) {
  if (getenv(GetDumpIRFlag().c_str()) == nullptr) return;
  std::ofstream of(file_name);
  if (of) {
    of << str << std::endl;
    of.close();
  }
}
void DumpStmt2File(const std::string &file_name, const Stmt &stmt) {
  if (getenv(GetDumpIRFlag().c_str()) == nullptr) return;
  std::ofstream of(file_name);
  if (of) {
    of << stmt << std::endl;
    of.close();
  }
}

void DumpBuildInfo(const BuildInfo &info) {
  if (getenv(GetDumpIRFlag().c_str()) == nullptr) return;
  auto dir_name = !info.opt.stitch
                    ? info.kernel_name + "_composite"
                    : "stitch_info/" + info.kernel_name + "_stitch_" + std::to_string(info.opt.stitch_ir_idx);
  std::ofstream of(dir_name + "/composite.log", std::ios::app);
  DumpHeader(of, "BuildInfo");
  of << info;
}
}  // namespace akg
