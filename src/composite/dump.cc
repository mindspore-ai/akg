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
  of << "switch_x_2_y:" << attr.switch_x_2_y << std::endl;
  of.close();
}

void DumpStitchInfo(const std::string &kernel_name, StitchAttrInfo &store_attr,
                    std::unordered_map<std::string, StitchBufferInfo> &stitch_buffer_map,
                    std::unordered_map<std::string, StitchBufferInfo> &buf_within_op_map,
                    std::vector<std::string> &allocate_revoke) {
  if (getenv(GetDumpIRFlag().c_str()) == nullptr) return;
  std::ofstream of;
  of.open("stitch_info/" + kernel_name + "_stitch.log", std::ios::app);
  if (!of.is_open()) return;
  DumpHeader(of, "StitchAttrInfo");
  of << "broadcast_size: " << store_attr.broadcast_size << std::endl;
  of << "type_array: ";
  for (const auto &a : store_attr.type_array) {
    std::string type;
    switch (a) {
      case StitchOpType::Elem:
        type = "Elem";
        break;
      case StitchOpType::Broadcast:
        type = "Broadcast";
        break;
      case StitchOpType::Reduce2D_X:
        type = "Reduce2D_X";
        break;
      case StitchOpType::All_Reduce:
        type = "All_Reduce";
        break;
      case StitchOpType::Reduce2D_Y:
        type = "Reduce2D_Y ";
        break;
      default:
        CHECK(0) << "Unknow stitch op type";
    }
    of << type << " ";
  }
  of << std::endl;
  of << "switch_x_2_y: " << store_attr.switch_x_2_y << std::endl;

  DumpHeader(of, "stitch_buffer_map");
  for (const auto &kv : stitch_buffer_map) {
    of << kv.first << std::endl;
    of << kv.second << std::endl;
  }
  DumpHeader(of, "buf_within_op_map");
  for (const auto &kv : buf_within_op_map) {
    of << kv.first << std::endl;
    of << kv.second << std::endl;
  }
  DumpHeader(of, "allocate_revoke");
  for (const auto &a : allocate_revoke) {
    of << a << std::endl;
  }
  of.close();
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
                    : "stitch_info/" + info.kernel_name + "_stitch_" + std::to_string(info.opt.stitch_ir_idx_);
  std::ofstream of(dir_name + "/composite.log", std::ios::app);
  DumpHeader(of, "BuildInfo");
  of << info;
}
}  // namespace akg
