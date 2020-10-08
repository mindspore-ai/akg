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
#ifndef POLY_CONSTRAIN_SCHEDULE_H_
#define POLY_CONSTRAIN_SCHEDULE_H_

// STL headers
#include <vector>
#include <memory>

// Other libraries
#include "isl/cpp.h"

// AKG headers
#include "poly/log_util.h"
#include "poly/schedule_pass.h"
#include "poly/schedule_pass/scheduling_mind_trick.h"

namespace akg {
namespace ir {
namespace poly {

class ConstrainSchedule : public SchedulePass {
 public:
  ConstrainSchedule(PassInfo &pass_info, ScopInfo &scop_info);
  ~ConstrainSchedule() {}

  virtual isl::schedule Run(isl::schedule sch);
  bool IsEnabled(void);

 private:
  bool CheckSchedule(const isl::schedule &) const;

  bool KernelIsEligible(const isl::schedule &sch) const;

  void LoadMindTricks(void);
  void LoadMindTrickFromFile(const std::string &filename);

  void AddMindTrick(const std::shared_ptr<SchedulingMindTrick> &mind_trick);
  void ExtractMindTrickInfo(const std::shared_ptr<SchedulingMindTrick> &mind_trick);
  void LogMindTrick(const std::shared_ptr<SchedulingMindTrick> &mind_trick);
  void ExtractGpuConfig(const std::shared_ptr<SchedulingMindTrick> &mind_trick);
  void ExtractDisabledPasses(const std::shared_ptr<SchedulingMindTrick> &mind_trick);
  void ExtractAttrs(const std::shared_ptr<SchedulingMindTrick> &mind_trick);

  void CreateMindTrickTemplate(const isl::schedule &sch);

  void InitVerbosityLevel(void);

  std::string GpuCompilerFlagsTempfileName(void) const;
  void GpuCompilerFlagsTempfileRemove(void);
  void GpuCompilerFlagsTempfileCreate(const std::shared_ptr<SchedulingMindTrick> &mind_trick);

  PassInfo &pass_info_;
  ScopInfo &scop_info_;
  std::vector<std::shared_ptr<SchedulingMindTrick>> mind_tricks_;

  ///////////////////////////////////////////////////////////////////////////
  // MindTrick paths
  ///////////////////////////////////////////////////////////////////////////

  static std::vector<std::string> MindTricksDirectories(void);

  ///////////////////////////////////////////////////////////////////////////
  // Supported environment variables
  ///////////////////////////////////////////////////////////////////////////

  static constexpr const char *const env_string_mind_tricks_enable_ = "MS_AKG_MIND_TRICKS";
  static constexpr const char *const env_string_mind_tricks_dir_ = "MS_AKG_MIND_TRICKS_DIR";
  static constexpr const char *const env_string_mind_tricks_verbosity_ = "MS_AKG_MIND_TRICKS_VERBOSITY";
  static constexpr const char *const env_string_mind_tricks_templates_ = "MS_AKG_MIND_TRICKS_TEMPLATES";
  static constexpr const char *const env_string_mind_tricks_operator_blacklist_ =
    "MS_AKG_MIND_TRICKS_OPERATOR_BLACKLIST";

  ///////////////////////////////////////////////////////////////////////////
  // Logging
  ///////////////////////////////////////////////////////////////////////////

  int verbosity_{0};

  std::string LogPrefixText(const bool prefix = true) const;

  // clang-format off
#define declare_constrain_schedule_log_wrappers(func)                                                               \
  void func(const std::string &message, const bool prefix = true) const;                                            \
  void func(const std::stringstream &stream, const bool prefix = true) const;                                       \
  void func(const int level, const std::string &message, const bool prefix = true) const;                           \
  void func(const int level, const std::stringstream &stream, const bool prefix = true) const;                      \
  void func(const akg::ir::poly::log::Verbosity level, const std::string &message, const bool prefix = true) const; \
  void func(const akg::ir::poly::log::Verbosity level, const std::stringstream &stream, const bool prefix = true) const;

  declare_constrain_schedule_log_wrappers(Info) declare_constrain_schedule_log_wrappers(Warn)
  declare_constrain_schedule_log_wrappers(Error)

#undef declare_constrain_schedule_log_wrappers
  // clang-format on
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_CONSTRAIN_SCHEDULE_H_
