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
#include "poly/schedule_pass/constrain_schedule.h"

#include <unistd.h>
#include <cstdio>
#include <cstdlib>

#include <isl/ctx.h>
#include <isl/schedule.h>

// opendir(), readdir, closedir()
#include <sys/types.h>
#include <dirent.h>

// TVM
#include <tvm/node/node.h>
#include <tvm/node/container.h>

// Local headers
#include "poly/schedule_pass/scheduling_mind_trick.h"
#include "poly/isl_util.h"
#include "poly/log_util.h"

namespace akg {
namespace ir {
namespace poly {

////////////////////////////////////////////////////////////////////////////////
// Constructors
////////////////////////////////////////////////////////////////////////////////

ConstrainSchedule::ConstrainSchedule(PassInfo &pass_info, ScopInfo &scop_info)
    : pass_info_(pass_info), scop_info_(scop_info) {
  pass_name_ = __FUNCTION__;

  InitVerbosityLevel();
  const log::Verbosity saved_verbosity = log::GetVerbosityLevel();
  log::SetVerbosityLevel(static_cast<log::Verbosity>(verbosity_));
  LoadMindTricks();
  log::SetVerbosityLevel(saved_verbosity);
}

////////////////////////////////////////////////////////////////////////////////
// Setters
////////////////////////////////////////////////////////////////////////////////

void ConstrainSchedule::AddMindTrick(const std::shared_ptr<SchedulingMindTrick> &mind_trick) {
  mind_tricks_.push_back(mind_trick);
}

std::vector<std::string> ConstrainSchedule::MindTricksDirectories(void) {
  // We only build the list of directories, directory existence will be checked later on
  std::vector<std::string> directories;

  // We only add the directory specified via the environment.
  const char *const user_directory = std::getenv(env_string_mind_tricks_dir_);
  if (user_directory) {
    directories.push_back(user_directory);
  }
  // Add other directories here if necessary...

  return directories;
}

void ConstrainSchedule::LoadMindTrickFromFile(const std::string &filename) {
  auto mind_trick = std::make_shared<SchedulingMindTrick>(pass_info_, scop_info_, verbosity_);

  mind_trick->Load(filename);
  // Alternative:
  //
  //     std::ifstream stream(filename);
  //     stream >> *mind_trick;

  if (*mind_trick) {
    AddMindTrick(mind_trick);
  } else {
    Warn("something was wrong with mind_trick " + filename);
  }
}

void ConstrainSchedule::LoadMindTricks(void) {
  Info(log::Verbosity::low,
       text_reverse text_bright_blue " ConstrainSchedule " text_reset text_bright_blue " LoadMindTricks()" text_reset);

  // Try to load the user's mind_trick from CLI
  const std::string user_mind_trick = scop_info_.user_config_.GetMindTrick();
  if (user_mind_trick != "") {
    auto mind_trick = std::make_shared<SchedulingMindTrick>(pass_info_, scop_info_, verbosity_);
    mind_trick->Parse(user_mind_trick);
    if (*mind_trick) {
      AddMindTrick(mind_trick);
      Info(log::Verbosity::medium, text_bright_magenta "User's mind_trick:\n" + mind_trick->str());
    } else {
      Warn(log::Verbosity::veryLow, "something was wrong with user's mind_trick");
    }
  }

  // Look for mind_tricks in several directories.
  std::vector<std::string> directories = MindTricksDirectories();
  for (const std::string &directory_str : directories) {
    log::Info(log::Verbosity::medium, "looking for mind tricks in " + directory_str);
    DIR *const directory = opendir(directory_str.c_str());
    if (directory) {
      // We first store the strings in a vecttor because there is no guarantee
      // on the order fo the files
      std::vector<std::string> files;
      for (struct dirent *entry = readdir(directory); entry; entry = readdir(directory)) {
        const std::string &filename = std::string(entry->d_name);
        if (filename.length() > 5 && filename.compare(filename.length() - 5, 5, ".json") == 0)
          files.push_back(filename);
      }
      if (!files.empty()) {
        std::sort(files.begin(), files.end());
        for (const std::string &filename : files) {
          const std::string &path = directory_str + "/" + filename;
          LoadMindTrickFromFile(path);
        }
      }
      closedir(directory);
    } else {
      log::Error(log::Verbosity::medium, "could not access directory " + directory_str);
    }
  }

  std::stringstream summary;
  summary << text_cyan << pass_name_ << " has " << mind_tricks_.size();
  summary << (mind_tricks_.size() <= 1 ? " trick" : " tricks");
  summary << "up its sleeve";
  Info(log::Verbosity::low, summary);
}

////////////////////////////////////////////////////////////////////////////////
// Other
////////////////////////////////////////////////////////////////////////////////

static inline void RunInfo(const std::string &stage, const std::string &kernel_name, const isl::schedule &schedule) {
  log::Info(log::Verbosity::low,
            text_reverse text_bright_blue " ConstrainSchedule " text_reset text_bright_blue " Run() " + stage);
  log::Info(log::Verbosity::low, text_bright_blue "name: " + kernel_name);
  log::Info(log::Verbosity::low, text_bright_blue "schedule:\n" + to_block_string(schedule));
  log::Info(log::Verbosity::medium, text_bright_blue "schedule (loop nest):\n" + to_c_code_string(schedule));
}

bool ConstrainSchedule::KernelIsEligible(const isl::schedule &sch) const {
  const std::string &kernel_name = scop_info_.user_config_.GetKernelName();

  // For now, the only criterion for eligibility is the existence of a blacklist exists and
  // its containing the current kernel name
  const char *const blacklist_path = std::getenv(env_string_mind_tricks_operator_blacklist_);
  if (!blacklist_path) {
    Info(log::Verbosity::high, env_string_mind_tricks_operator_blacklist_ + std::string(" not set"));
    return true;
  }

  std::fstream file(blacklist_path, std::ios::in);
  if (!file.is_open()) {
    Warn(log::Verbosity::high, "could not open operator blacklist: " + std::string(blacklist_path));
    return true;
  }

  // Very basic analysis of the lines (no support for extra spaces, etc.)
  std::string line;
  while (getline(file, line)) {
    // Support for "comments" in the blacklist file
    if (line[0] == '#') continue;

    if (kernel_name == line) return false;
  }

  return true;
}

bool ConstrainSchedule::IsEnabled(void) {
  // PassMgrStrategy already checks this and does not even register the pass
  // if akg::ir::poly::UserConfig::GetEnableMindTrick() returns false.
  // We check once more in case it was set to true at startup and the runtime
  // later decided to change the value.
  if (!scop_info_.user_config_.GetEnableMindTrick()) {
    Info("ConstrainSchedule was disabled via akg::ir::poly::UserConfig");
    return false;
  }

  const char *const env_mind_tricks = std::getenv(env_string_mind_tricks_enable_);
  if (env_mind_tricks && std::string(env_mind_tricks) == "false") {
    Info("ConstrainSchedule was disabled via environment variable " + std::string(env_string_mind_tricks_enable_));
    return false;
  }

  return true;
}

isl::schedule ConstrainSchedule::Run(isl::schedule sch) {
  if (!IsEnabled()) return sch;

  const std::string &target = scop_info_.user_config_.GetTarget();
  const std::string &kernel_name = scop_info_.user_config_.GetKernelName();

  // We will want to restore the original value
  const log::Verbosity saved_verbosity = log::GetVerbosityLevel();
  log::SetVerbosityLevel(static_cast<log::Verbosity>(verbosity_));

  isl::schedule result = sch;

  // Make sure the constraints are available...
  // We expect this pass to be right after InitSchedule: most information is
  // initialized or computed in InitSchedule. However the constraints are
  // usually computed in ComputeSchedule (which we may disable afterwards!).
  pass_info_.constraints_ = MakeScheduleConstraints(sch, pass_info_);

  // Check whether we want to use ConstrainSchedule on this kernel
  const bool eligible = KernelIsEligible(sch);
  if (!eligible) {
    Warn("ConstrainSchedule will ignore this operator...");
    return sch;
  }

  const std::size_t total = mind_tricks_.size();
  RunInfo("input", kernel_name, sch);
  std::stringstream summary;
  summary << pass_name_ << " has " << total << " tricks up its sleeve";
  Info(log::Verbosity::low, summary);

  CreateMindTrickTemplate(sch);
  if (target == TARGET_CUDA) {
    GpuCompilerFlagsTempfileRemove();
  }

  size_t current = 0;
  for (std::shared_ptr<SchedulingMindTrick> &mind_trick : mind_tricks_) {
    const std::string name = mind_trick->GetName();
    current++;

    if (static_cast<log::Verbosity>(verbosity_) >= log::Verbosity::low) {
      std::stringstream stream;
      stream << text_reverse text_magenta " SchedulingMindTrick " text_reset text_magenta " ";
      stream << "[" << current << "/" << total << "] ";
      stream << name;

      const std::string &str = stream.str();
      Info(str, false);
    }

    const bool matches = mind_trick->Matches(sch);
    if (!matches) {
      Info(log::Verbosity::veryHigh, text_dim text_yellow + mind_trick->str());
      continue;
    }

    const bool needs_check = mind_trick->NeedsScheduleCheck();
    if (!needs_check) Info(log::Verbosity::veryLow, text_bright_yellow "MindTrick requests no schedule check!");

    const isl::schedule &candidate = mind_trick->Apply(sch);
    const bool has_schedule = mind_trick->HasSchedule();
    if (!has_schedule) {
      Warn(log::Verbosity::low, "'" + name + "': no schedule available");
      continue;
    }

    const bool valid = !needs_check || CheckSchedule(candidate);
    if (valid) {
      if (needs_check) Info(log::Verbosity::low, text_green "schedule is valid!");
      result = candidate;
      ExtractMindTrickInfo(mind_trick);
      LogMindTrick(mind_trick);
      if (target == TARGET_CUDA) {
        GpuCompilerFlagsTempfileCreate(mind_trick);
      }
      break;
    } else {
      Info(log::Verbosity::high, text_dim text_yellow + mind_trick->str());
    }
  }

  RunInfo("output", kernel_name, result);
  log::SetVerbosityLevel(saved_verbosity);

  return result;
}

void ConstrainSchedule::ExtractMindTrickInfo(const std::shared_ptr<SchedulingMindTrick> &mind_trick) {
  const std::string &target = scop_info_.user_config_.GetTarget();
  if (target == TARGET_CUDA) {
    ExtractGpuConfig(mind_trick);
  }

  ExtractDisabledPasses(mind_trick);
  ExtractAttrs(mind_trick);
}

void ConstrainSchedule::LogMindTrick(const std::shared_ptr<SchedulingMindTrick> &mind_trick) {
  const std::string kernel_name = scop_info_.user_config_.GetKernelName();
  const std::string mind_trick_name = mind_trick->GetName();
  const std::string &output = mind_trick->str();

  Info(log::Verbosity::veryLow, text_reverse text_bright_blue " ConstrainSchedule ", false);
  Info(log::Verbosity::veryLow, text_bright_blue "using schedule from \'" + mind_trick_name + "\'");
  Info(log::Verbosity::medium, text_dim text_green + mind_trick->str());

  scop_info_.user_config_.SetConstrainedSchedulingOutput(output);
}

void ConstrainSchedule::ExtractGpuConfig(const std::shared_ptr<SchedulingMindTrick> &mind_trick) {
  const std::string blocks = mind_trick->GetGpuBlocks();
  const std::string threads = mind_trick->GetGpuThreads();
  if (blocks != "" && threads != "") {
    scop_info_.user_config_.SetBlockConfig(blocks);
    scop_info_.user_config_.SetThreadConfig(threads);
  }
}

void ConstrainSchedule::ExtractDisabledPasses(const std::shared_ptr<SchedulingMindTrick> &mind_trick) {
  // We always want to disable ComputeSchedule when using a mind_trick!
  disabled_passes_.insert("ComputeSchedule");
  // Then maybe disable other passes...
  const std::set<std::string> &passes = mind_trick->GetDisabledPasses();
  disabled_passes_.insert(passes.begin(), passes.end());
}

void ConstrainSchedule::CreateMindTrickTemplate(const isl::schedule &sch) {
  const char *const env_templates = std::getenv(env_string_mind_tricks_templates_);
  if (!env_templates || std::string(env_templates) != "true") {
    return;
  }

  const std::string kernel_name = scop_info_.user_config_.GetKernelName();
  const std::string &filename = "mindtrick-template_" + kernel_name + ".json";

  std::ofstream output(filename);
  output << SchedulingMindTrick::TemplateString(scop_info_, sch);
  output.close();
}

void ConstrainSchedule::ExtractAttrs(const std::shared_ptr<SchedulingMindTrick> &mind_trick) {
  const air::Map<std::string, air::NodeRef> &attrs = mind_trick->GetAttrs();
  scop_info_.user_config_.SetAttrs(attrs);
}

void ConstrainSchedule::InitVerbosityLevel(void) {
#ifdef AKG_CONSTRAIN_SCHEDULE_VERBOSITY
  {
    constexpr int preprocessor_verbosity = AKG_CONSTRAIN_SCHEDULE_VERBOSITY;
    if (preprocessor_verbosity >= 0) verbosity_ = preprocessor_verbosity;
  }
#endif
  {
    const char *const env_verbosity_string = std::getenv(env_string_mind_tricks_verbosity_);
    if (env_verbosity_string) {
      const int env_verbosity = std::stoi(env_verbosity_string);
      if (env_verbosity >= 0) verbosity_ = env_verbosity;
    }
  }
  {
    const int attrs_verbosity = scop_info_.user_config_.GetConstrainScheduleVerbosity();
    if (attrs_verbosity >= 0) verbosity_ = attrs_verbosity;
  }
}
////////////////////////////////////////////////////////////////////////////////
// GPU Compiler flags
////////////////////////////////////////////////////////////////////////////////

std::string ConstrainSchedule::GpuCompilerFlagsTempfileName(void) const {
  std::stringstream filename_stream;
  filename_stream << ".akg_gpu_compiler_flags_" << getpid();

  const std::string &filename = filename_stream.str();
  return filename;
}

void ConstrainSchedule::GpuCompilerFlagsTempfileRemove(void) {
  const std::string &filename = GpuCompilerFlagsTempfileName();
  std::remove(filename.c_str());
}

void ConstrainSchedule::GpuCompilerFlagsTempfileCreate(const std::shared_ptr<SchedulingMindTrick> &mind_trick) {
  const std::vector<std::string> &flags = mind_trick->GetGpuCompilerFlags();
  if (flags.empty()) {
    return;
  }

  const std::string &filename = GpuCompilerFlagsTempfileName();
  std::ofstream tempfile(filename);
  for (const std::string &flag : flags) {
    tempfile << flag << std::endl;
  }
  tempfile.close();
}

////////////////////////////////////////////////////////////////////////////////
// Logging
////////////////////////////////////////////////////////////////////////////////

std::string ConstrainSchedule::LogPrefixText(const bool prefix) const {
  if (!prefix) {
    return "";
  }

  const std::string &kernel_name = scop_info_.user_config_.GetKernelName();
  const std::string &prefix_text = "'" + kernel_name + "': ";
  return prefix_text;
}

// clang-format off
#define define_constrain_schedule_log_wrappers(func)                                                                   \
  void ConstrainSchedule::func(const std::string &message, const bool prefix) const {                                  \
    const std::string &prefix_text = LogPrefixText(prefix);                                                            \
    log::func(prefix_text + message);                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  void ConstrainSchedule::func(const std::stringstream &stream, const bool prefix) const {                             \
    const std::string &message = stream.str();                                                                         \
    func(message, prefix);                                                                                             \
  }                                                                                                                    \
                                                                                                                       \
  void ConstrainSchedule::func(const int level, const std::string &message, const bool prefix) const {                 \
    const std::string &prefix_text = LogPrefixText(prefix);                                                            \
    log::func(level, prefix_text + message);                                                                           \
  }                                                                                                                    \
  void ConstrainSchedule::func(const int level, const std::stringstream &stream, const bool prefix) const {            \
    const std::string &message = stream.str();                                                                         \
    func(level, message, prefix);                                                                                      \
  }                                                                                                                    \
  void ConstrainSchedule::func(const log::Verbosity level, const std::string &message, const bool prefix) const {      \
    const std::string &prefix_text = LogPrefixText(prefix);                                                            \
    log::func(level, prefix_text + message);                                                                           \
  }                                                                                                                    \
  void ConstrainSchedule::func(const log::Verbosity level, const std::stringstream &stream, const bool prefix) const { \
    const std::string &message = stream.str();                                                                         \
    func(level, message, prefix);                                                                                      \
  }

define_constrain_schedule_log_wrappers(Info)
define_constrain_schedule_log_wrappers(Warn)
define_constrain_schedule_log_wrappers(Error)

#undef define_constrain_schedule_log_wrappers
// clang-format on

}  // namespace poly
}  // namespace ir
}  // namespace akg
