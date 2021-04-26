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
#ifndef POLY_SCHEDULING_MIND_TRICK_H_
#define POLY_SCHEDULING_MIND_TRICK_H_

// STL includes
#include <ostream>
#include <fstream>
#include <regex>

// External libraries
#include <picojson.h>
#include <isl/cpp.h>

// TVM
#include <tvm/node/container.h>
#include <tvm/node/node.h>

// Internal headers
#include "poly/isl.h"
#include "poly/log_util.h"
#include "poly/schedule_pass.h"
#include "poly/pass_info.h"
#include "poly/scop_info.h"
#include "poly/isl_influence.h"

namespace akg {
namespace ir {
namespace poly {

// Implementation notes
//
// 1. The class is non copyable.
//    We use raw pointers to isl objects because the C++ wrapped objects are
//    not practical in cases where the isl object is optional and std::optional
//    curretly is not an option.
//    Hence, we directly manage the isl objects. To avoid further complications
//    with copy-constructors, move-constructors, isl_*_free(), isl_*_copy(),
//    etc., the class is non copyable.

// data_value: tuple of <scheduling dim, coeff dim, coeff type, list of stmt name>
// using data_value = std::tuple<int, int, isl_influence_coeff_type, std::vector<std::string>>;

// single_data: tuple of <stmt name, scheduling dim, coeff dim, coeff type, value>
using single_data = std::tuple<std::string, int, int, isl_influence_coeff_type, int>;

// div_mod_data
// * if division: tuple of <stmt name, scheduling dim, divisor>
// * if modulo: tuple of <stmt name, scheduling dim,  modulo value>
using div_mod_data = std::tuple<std::string, int, int>;

class GpuConfig {
 public:
  std::vector<int> block_sizes_;
  std::vector<int> thread_sizes_;
  std::vector<int> block_dimensions_;
  std::vector<int> thread_dimensions_;
  std::vector<int> swizzle_dimensions_;
  std::vector<std::string> compiler_flags_;
};

enum class MindTrickType {
  none = 0,
  manual,
};

std::string to_string(MindTrickType t);
MindTrickType MindTrickTypeFromString(const std::string &str);

class SchedulingMindTrick {
 public:
  ///////////////////////////////////////////////////////////////////////////
  // Constructors and similar methods
  ///////////////////////////////////////////////////////////////////////////

  // Highly recommended to call the SchedulingMindTrick(PassInfo&, ScopInfo&)
  // constructor in other constructors and in derived classes constructors!
  SchedulingMindTrick(PassInfo &pass_info, ScopInfo &scop_info, int verbosity = -1);
  ~SchedulingMindTrick();

  void Load(const std::string &filename);

  // Parse JSON representation
  void Parse(const picojson::value &json);
  void Parse(const std::string &serialized_json);
  std::istream &Parse(std::istream &streamed_json);

  // Non copyable
  SchedulingMindTrick(const SchedulingMindTrick &) = delete;
  SchedulingMindTrick &operator=(const SchedulingMindTrick &) = delete;

  ///////////////////////////////////////////////////////////////////////////
  // MindTrick state
  ///////////////////////////////////////////////////////////////////////////

  explicit operator bool() const;

  ///////////////////////////////////////////////////////////////////////////
  // MindTrick use
  ///////////////////////////////////////////////////////////////////////////

  bool Matches(const isl::schedule &sch) const;
  isl::schedule Apply(const isl::schedule &sch);

  ///////////////////////////////////////////////////////////////////////////
  // I/O
  ///////////////////////////////////////////////////////////////////////////

  std::string str(void) const;
  std::ostream &Output(std::ostream &stream) const;
  std::istream &Input(std::istream &stream);

  friend std::ostream &operator<<(std::ostream &stream, const SchedulingMindTrick &mind_trick);
  friend std::istream &operator>>(std::istream &stream, SchedulingMindTrick &mind_trick);
  friend std::string to_string(const SchedulingMindTrick &mind_trick);

  static std::string TemplateString(ScopInfo &scop_info, const isl::schedule &schedule,
                                    MindTrickType type = MindTrickType::none);

  ///////////////////////////////////////////////////////////////////////////
  // MindTrick metadata
  ///////////////////////////////////////////////////////////////////////////

  void SetName(const std::string &name);
  std::string GetName(void) const;
  std::string GetTarget(void) const;

  ///////////////////////////////////////////////////////////////////////////
  // Misc. attributes
  ///////////////////////////////////////////////////////////////////////////

  bool HasSchedule(void) const;
  bool NeedsScheduleCheck(void) const;
  const air::Map<std::string, air::NodeRef> GetAttrs(void) const;

  ///////////////////////////////////////////////////////////////////////////
  // GPU Mapping
  ///////////////////////////////////////////////////////////////////////////

  void GuessGpuConfig(void);
  std::string GetGpuBlocks(void) const;
  std::string GetGpuThreads(void) const;
  std::vector<std::string> GetGpuCompilerFlags(void) const;

  ///////////////////////////////////////////////////////////////////////////
  // Pass toggling
  ///////////////////////////////////////////////////////////////////////////

  const std::set<std::string> &GetDisabledPasses(void) const;

  ///////////////////////////////////////////////////////////////////////////
  // Mind Trick Type
  ///////////////////////////////////////////////////////////////////////////

  MindTrickType GetType(void) const;
  void SetType(MindTrickType type);

 protected:
  ///////////////////////////////////////////////////////////////////////////
  // JSON parsing
  ///////////////////////////////////////////////////////////////////////////

  static picojson::value maybe(const picojson::value &node, const std::string &key);
  static picojson::value maybe(const picojson::object &node, const std::string &key);

  static std::vector<int> to_int_vector(const picojson::value &node);
  static std::vector<int> to_int_vector(const picojson::array &node);
  static std::vector<std::string> to_string_vector(const picojson::value &node);
  static std::vector<std::string> to_string_vector(const picojson::array &node);

  void ParseName(const picojson::value &node);
  void ParseDisabledPasses(const picojson::value &node);
  void ParsePattern(const picojson::value &node);
  void ParseOperatorName(const picojson::value &node);
  void ParseDomain(const picojson::value &node);
  void ParseSchedule(const picojson::value &node);
  void ParseGpuInfo(const picojson::value &node);
  void ParseExplicitTree(const picojson::value &node);
  void ParseCheckSchedule(const picojson::value &node);
  void ParseAttrs(const picojson::value &node);
  void ParseVerbosity(const picojson::value &node);
  void ParseSoftConstraints(const picojson::value &node);

  ///////////////////////////////////////////////////////////////////////////
  // Schedule
  ///////////////////////////////////////////////////////////////////////////

  isl::schedule GetSchedule(void) const;

  ///////////////////////////////////////////////////////////////////////////
  // Schedule suggestion
  ///////////////////////////////////////////////////////////////////////////

  void BuildSuggestedSchedule(void);

  // Various helpers to build the suggested schedule
  __isl_give isl_schedule *ComputeScheduleSuggestion(void);
  __isl_give isl_schedule *PrepareMappingOuterBand(__isl_take isl_schedule *schedule, GpuConfig &info);
  __isl_give isl_schedule_node *SplitSwizzleDim(__isl_take isl_schedule_node *band, int dimension);

  ///////////////////////////////////////////////////////////////////////////
  // Soft constraints
  ///////////////////////////////////////////////////////////////////////////

  void CollectSoftConstraintsData(std::string stmt_name, unsigned int sched_dim, int coeff_dim,
                                  isl_influence_coeff_type coeff_type, std::string coeff_vec_i);
  void BuildSoftConstraints();
  void BuildInfluenceList(std::vector<single_data> singles);
  void BuildInfluenceEqualList(std::map<std::string, std::vector<single_data>> linked);

  void BuildInfluencedSchedule(void);
  void IslInfluenceToggle(bool toggle);

  __isl_give isl_schedule *AdjustSchedule(__isl_take isl_schedule *schedule, const std::vector<div_mod_data> &modulos,
                                          const std::vector<div_mod_data> &divisions);

  // Misc helpers
  std::vector<std::string> split_string(std::string str, std::string delim);

  ///////////////////////////////////////////////////////////////////////////
  // Internal data
  ///////////////////////////////////////////////////////////////////////////

  // AKG info
  PassInfo &pass_info_;
  ScopInfo &scop_info_;

  bool correctly_parsed_{false};

  std::string operator_{""};
  std::string pattern_{""};
  isl_schedule *explicit_tree_{nullptr};

  isl_union_set *domain_{nullptr};
  isl_schedule *suggested_schedule_{nullptr};
  std::string suggested_schedule_string_{""};
  std::vector<std::string> suggested_schedule_vector_;

  std::vector<std::tuple<std::string, std::vector<int>>> post_transformations_;

  std::vector<single_data> singles_;
  std::map<std::string, std::vector<single_data>> linked_;
  std::vector<div_mod_data> modulos_;
  std::vector<div_mod_data> divisions_;
  isl_influence_list *influence_list_{nullptr};
  isl_influence_equal_list *influence_equal_list_{nullptr};
  std::string parse_soft_constraints_log_str_{""};
  isl_schedule *influenced_schedule_{nullptr};

  bool check_schedule_{true};

  std::string target_{""};
  std::string name_{"unnamed mind_trick"};

  GpuConfig gpu_info_;

  air::Map<std::string, air::NodeRef> attrs_;

  std::set<std::string> disabled_passes_;

  MindTrickType type_{MindTrickType::manual};

  int verbosity_{0};

  std::string LogPrefixText(const bool prefix = true) const;

  // clang-format off
#define declare_scheduling_mind_trick_log_wrappers(func)                                                            \
  void func(const std::string &message, const bool prefix = true) const;                                            \
  void func(const std::stringstream &stream, const bool prefix = true) const;                                       \
  void func(const int level, const std::string &message, const bool prefix = true) const;                           \
  void func(const int level, const std::stringstream &stream, const bool prefix = true) const;                      \
  void func(const akg::ir::poly::log::Verbosity level, const std::string &message, const bool prefix = true) const; \
  void func(const akg::ir::poly::log::Verbosity level, const std::stringstream &stream, const bool prefix = true) const;

  declare_scheduling_mind_trick_log_wrappers(Info)
  declare_scheduling_mind_trick_log_wrappers(Warn)
  declare_scheduling_mind_trick_log_wrappers(Error)

#undef declare_scheduling_mind_trick_log_wrappers

 private :
  // Non copyable
  SchedulingMindTrick();
  // clang-format on
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_SCHEDULING_MIND_TRICK_H_
