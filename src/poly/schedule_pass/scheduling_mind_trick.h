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

namespace akg {
namespace ir {
namespace poly {

///////////////////////////////////////////////////////////////////////////
// Supported environment variables
///////////////////////////////////////////////////////////////////////////

static constexpr const char *const env_string_mind_tricks_enable_ = "MS_AKG_MIND_TRICKS";
static constexpr const char *const env_string_mind_tricks_dir_ = "MS_AKG_MIND_TRICKS_DIR";
static constexpr const char *const env_string_mind_tricks_verbosity_ = "MS_AKG_MIND_TRICKS_VERBOSITY";
static constexpr const char *const env_string_mind_tricks_templates_ = "MS_AKG_MIND_TRICKS_TEMPLATES";
static constexpr const char *const env_string_mind_tricks_operator_blacklist_ = "MS_AKG_MIND_TRICKS_OPERATOR_BLACKLIST";

static constexpr const char *const env_string_mind_tricks_autogen_ = "MS_AKG_MIND_TRICKS_AUTOGEN";
static constexpr const char *const env_string_mind_tricks_autogen_swizzle_ = "MS_AKG_MIND_TRICKS_AUTOGEN_SWIZZLE";

////////////////////////////////////////////////////////////////////////////////
// Miscellaneous useful types or classes for SchedulingMindTrick
////////////////////////////////////////////////////////////////////////////////

enum class SoftToken {
  INVALID_TOKEN = -1,
  OPEN_BRACKET,       // 0
  CLOSE_BRACKET,      // 1
  OPEN_PARENTHESIS,   // 2
  CLOSE_PARENTHESIS,  // 3
  COMMA,              // 4
  QUESTION_MARK,      // 5
  MINUS,              // 6
  DIVISION,           // 7
  MODULO,             // 8
  DIGIT,              // 9
};

// single_data: tuple of <stmt name, scheduling dim, coeff dim, coeff type, value>
using single_data = std::tuple<std::string, int, int, isl_influence_coeff_type, int>;

// div_mod_data
// tuple of <stmt name, scheduling dim, pair of <token type, value>>
// token type can either be modulo or division
using div_mod_data = std::tuple<std::string, int, std::pair<SoftToken, int>>;

////////////////////////////////////////////////////////////////////////////////
// GpuConfig
////////////////////////////////////////////////////////////////////////////////

class GpuConfig {
 public:
  std::vector<int> block_sizes_;
  std::vector<int> thread_sizes_;
  std::vector<int> block_dimensions_;
  std::vector<int> thread_dimensions_;
  std::vector<int> swizzle_dimensions_;
  std::vector<std::string> compiler_flags_;
  bool automap_{false};
  bool was_automapped_{false};
  bool has_swizzle_dim_{false};

  bool CanPrepareMappingOuterBand(const isl::schedule &schedule) const;
  void OffsetThreadDimensions(int offset);
};

////////////////////////////////////////////////////////////////////////////////
// MindTrickType
////////////////////////////////////////////////////////////////////////////////

enum class MindTrickType {
  none = 0,
  manual,
  autogen,
};

std::string to_string(MindTrickType t);
MindTrickType MindTrickTypeFromString(const std::string &str);

////////////////////////////////////////////////////////////////////////////////
// SchedulingMindTrick
////////////////////////////////////////////////////////////////////////////////

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
  void Parse(const picojson::value &root);
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
  const std::string &GetName(void) const;
  const std::string &GetTarget(void) const;

  ///////////////////////////////////////////////////////////////////////////
  // Misc. attributes
  ///////////////////////////////////////////////////////////////////////////

  bool HasSchedule(void) const;
  bool NeedsScheduleCheck(void) const;
  const air::Map<std::string, air::NodeRef> GetAttrs(void) const;

  ///////////////////////////////////////////////////////////////////////////
  // GPU Mapping
  ///////////////////////////////////////////////////////////////////////////

  GpuConfig ExtractGpuConfig(const isl::schedule &schedule, const GpuConfig &info);
  std::string GetGpuBlocks(void) const;
  std::string GetGpuThreads(void) const;
  const std::vector<std::string> &GetGpuCompilerFlags(void) const;
  bool HasGpuSwizzleDim(void) const;

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
  // Soft constraints parser utils
  ///////////////////////////////////////////////////////////////////////////

  SoftToken GetSoftToken(const char &token) const;
  bool HasValidNextToken(const char &token1, const char &token2) const;
  void FlushSoftData();
  bool CheckSoftExpression(const std::string &expr);
  std::pair<int, isl_influence_coeff_type> DetermineCoeffType(unsigned int incr, unsigned int nb_vars,
                                                              unsigned int nb_params) const;

  ///////////////////////////////////////////////////////////////////////////
  // Schedule
  ///////////////////////////////////////////////////////////////////////////

  isl::schedule GetSchedule(void) const;

  ///////////////////////////////////////////////////////////////////////////
  // Schedule suggestion
  ///////////////////////////////////////////////////////////////////////////

  void BuildSuggestedSchedule(const isl::schedule &initial);
  isl::schedule ComputeScheduleSuggestion(const isl::schedule &initial);

  ///////////////////////////////////////////////////////////////////////////
  // GPU specific utilities
  ///////////////////////////////////////////////////////////////////////////

  isl::schedule GpuPostProcessSchedule(const isl::schedule &schedule, GpuConfig &info);

  void GpuPrepareMappingOuterBandFindSizes(const isl::schedule &schedule, GpuConfig &info);
  isl::schedule_node_band GpuPrepareMappingOuterBandTrustUser(const isl::schedule_node_band &band,
                                                              const GpuConfig &info);
  isl::schedule GpuPrepareMappingOuterBand(const isl::schedule &schedule, GpuConfig &info);

  isl::schedule_node_band DetectAndSplitSwizzleDim(const isl::schedule_node_band &band, GpuConfig &info);
  isl::schedule_node_band SplitSwizzleDim(const isl::schedule_node_band &band, GpuConfig &info, int dimension);

  isl::schedule_node_band GpuStripmineUniqueCoincidentDimension(const isl::schedule_node_band &band, int &innermost,
                                                                const int thread_max);
  isl::schedule_node_band GpuAutomapThreads(const isl::schedule_node_band &band, GpuConfig &config, int &innermost,
                                            const int thread_max);
  isl::schedule_node_band GpuCollapseRemainingDimensions(const isl::schedule_node_band &band, GpuConfig &config,
                                                         int &innermost);
  isl::schedule_node_band GpuAutomapBlocks(const isl::schedule_node_band &band, GpuConfig &config, int &innermost);
  isl::schedule GpuAutomap(const isl::schedule &schedule, GpuConfig &config);
  bool GpuShouldAutomap(const GpuConfig &config) const;

  isl::schedule GpuAutoDisablePasses(const isl::schedule &schedule, const GpuConfig &config);

  ///////////////////////////////////////////////////////////////////////////
  // Soft constraints
  ///////////////////////////////////////////////////////////////////////////

  static std::tuple<std::string, std::string> AutoGenSoftConstraints(ScopInfo &scop_info, const isl::schedule &sch);
  static std::tuple<std::string, std::string> AutoGenGPUSoftConstraints(ScopInfo &scop_info, const isl::schedule &sch);

  void CollectSoftConstraintsData(std::string stmt_name, unsigned int dim, unsigned int nb_vars, unsigned int nb_params,
                                  std::string expr);

  void BuildSoftConstraints();
  void BuildInfluenceList(std::vector<single_data> singles);
  void BuildInfluenceEqualList(std::map<std::string, std::vector<single_data>> linked);

  void BuildInfluencedSchedule(void);
  void IslInfluenceToggle(bool toggle);

  isl::schedule AdjustSchedule(const isl::schedule &schedule, const std::vector<div_mod_data> &modulos_divisions);

  ///////////////////////////////////////////////////////////////////////////
  // Miscellaneous methods
  ///////////////////////////////////////////////////////////////////////////

  std::vector<std::string> split_string(std::string str, std::string delim) const;
  int FindStripmineFactor(int val, int limit, bool greedy = false) const;
  int FindInnermostCoincidentDimension(const isl::schedule_node_band &band);

  ///////////////////////////////////////////////////////////////////////////
  // Internal data
  ///////////////////////////////////////////////////////////////////////////

  // AKG info
  PassInfo &pass_info_;
  ScopInfo &scop_info_;

  bool correctly_parsed_{false};

  std::string operator_{""};
  std::string pattern_{""};
  isl::schedule explicit_tree_;

  isl::union_set domain_;
  isl::schedule suggested_schedule_;
  std::string suggested_schedule_string_{""};
  std::vector<std::string> suggested_schedule_vector_;

  std::vector<std::tuple<std::string, std::vector<int>>> post_transformations_;

  std::vector<single_data> singles_;
  std::map<std::string, std::vector<single_data>> linked_;

  std::vector<div_mod_data> modulos_divisions_;
  isl_influence_list *influence_list_{nullptr};
  isl_influence_equal_list *influence_equal_list_{nullptr};
  std::string parse_soft_constraints_log_str_{""};
  isl::schedule influenced_schedule_;

  bool check_schedule_{true};

  std::string target_{""};
  std::string name_{"unnamed mind_trick"};

  GpuConfig gpu_info_;

  air::Map<std::string, air::NodeRef> attrs_;

  std::set<std::string> disabled_passes_;

  MindTrickType type_{MindTrickType::manual};

  int verbosity_{0};

  std::string LogPrefixText(const bool prefix = true) const;
  void DebugSchedule(const isl::schedule &schedule, const std::string &message = "",
                     const log::Verbosity level = log::Verbosity::high) const;

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


std::tuple<std::string, std::string> AutoGenAscend910SoftConstraints(const ScopInfo &scop_info,
                                                                     const isl::schedule &sch);

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_SCHEDULING_MIND_TRICK_H_
