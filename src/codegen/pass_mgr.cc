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

#include "codegen/pass_mgr.h"

#include <unordered_set>
#include <chrono>

#include "common/util_cce.h"

namespace akg {
void PassMgr::InitializeSubName() {
  auto pos = pass_name_.find_last_of('.');
  sub_name_ = pos == std::string::npos ? pass_name_ : pass_name_.substr(pos + 1);
}

TVMRetValue PassMgr::Run() const {
  const auto *packed_func = air::runtime::Registry::Get(pass_name_);
  CHECK(packed_func != nullptr) << "PackedFunc " << pass_name_ << " not found";

  TVMRetValue res;

  auto start_time = std::chrono::steady_clock::now();
  packed_func->CallPacked(TVMArgs(args_values_.data(), args_types_.data(), args_values_.size() - 1), &res);
  CHECK(res.type_code() != kNull) << "PassMgr " << tl_pass_id_ << "_" << sub_name_ << " result illegal.";

  if (enable_timer_) {
    auto end_time = std::chrono::steady_clock::now();
    int64_t elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    PassTimer *pass_timer = PassTimer::GetInstance();
    if (pass_timer == nullptr) {
      LOG(INFO) << "Failed to initialize PassTimer.";
    } else {
      pass_timer->AddItem(sub_name_, elapsed);
    }
  }

  tl_pass_id_++;
  return res;
}

std::string PassMgr::GetDumpIrFilePath() const {
  std::string file_name = tl_dump_ir_dir_;
  file_name.append("/")
    .append(tl_pass_id_ < 10 ? "0" + std::to_string(tl_pass_id_) : std::to_string(tl_pass_id_))
    .append("_")
    .append(sub_name_);
  return file_name;
}

void PassMgr::DumpIr(std::function<void(std::ostream &os)> print) const {
  auto file_name = GetDumpIrFilePath().append(".cc");
  std::ofstream of(file_name);
  CHECK(of.is_open()) << "Failed to open " << file_name << " to dump ir.";

  print(of);
  of.close();
}

static std::unordered_set<std::string> VectorToSet(const std::vector<std::string> &list) {
  std::unordered_set<std::string> set;
  for (const auto &s : list) {
    set.insert(s);
  }
  return set;
}

bool PassMgr::ShouldDumpC() const {
  const char *dump_c_pass_cstr = std::getenv("DUMP_C_PASS");
  if (dump_c_pass_cstr == nullptr) {
    return false;
  }
  auto dump_c_passes = common::Split(std::string(dump_c_pass_cstr), ",");
  auto dump_c_passes_set = VectorToSet(dump_c_passes);
  if (dump_c_passes_set.count(sub_name_) > 0) {
    return true;
  }
  for (const auto &pass : dump_c_passes) {
    if (pass.find(std::to_string(tl_pass_id_)) == 0) {
      return true;
    }
  }
  return false;
}

thread_local int PassMgr::tl_pass_id_ = -1;
thread_local air::BuildConfig PassMgr::tl_config_ = air::BuildConfig::Current();
thread_local std::string PassMgr::tl_dump_ir_dir_ = "ir/";
thread_local air::Array<NodeRef> PassMgr::tl_args_;
}  // namespace akg
