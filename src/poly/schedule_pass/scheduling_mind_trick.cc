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
#include "scheduling_mind_trick.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <functional>
#include <string.h>
#include <regex>

// TVM
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/dtype.h>

#include "poly/isl_util.h"
#include "poly/log_util.h"
#include "poly/gpu_emit/gpu_isl_emitter.h"

namespace akg {
namespace ir {
namespace poly {
///////////////////////////////////////////////////////////////////////////
// Supported environment variables
///////////////////////////////////////////////////////////////////////////

static constexpr const char *const kEnvStringMindTricksAutogenSwizzle = "MS_AKG_MIND_TRICKS_AUTOGEN_SWIZZLE";

////////////////////////////////////////////////////////////////////////////////
// Local "implementation-detail" variables
////////////////////////////////////////////////////////////////////////////////

static constexpr int gpu_thread_max_ = 1024;

////////////////////////////////////////////////////////////////////////////////
// Additional classes for the implementation (not exposed in the header)
////////////////////////////////////////////////////////////////////////////////

class AccessInfo {
 public:
  // Quick and dirty types
  static constexpr const char *const read = "read";
  static constexpr const char *const write = "write";

  std::string statement_{""};
  std::string type_{"none"};
  isl::map access_;

  std::string data_{""};
  Type data_type_;
  int bytes_{-1};
  int swizzle_size_{-1};
  // Should be from innermost to outermost!
  std::vector<std::string> dimensions_;
  std::vector<long int> dimensions_sizes_;
  std::vector<long int> consecutive_bytes_;

  AccessInfo(ScopInfo &scop_info, const std::string &name, const std::string &type, const isl::map &access);
  std::vector<long int> ConsecutiveBytes(const std::vector<long int> &sizes) const;
  double Score(ScopInfo &scop_info, const std::string &dimension, const std::vector<std::string> &subset) const;
  void Log(const std::string &prefix = "") const;

  static int AccessSwizzleSize(const isl::map &access);
  static std::vector<long int> DimensionsBytes(const isl::map &access, const int bytes);
  static bool DimensionIsContiguous(const isl::map &access, const unsigned int dimension);
  static std::vector<isl::id> InvolvedInputDims(const isl::map &access, const unsigned int dimension);
  static std::vector<std::string> OrderInputDimsForInnermostAccess(const isl::map &access);
};

class DimensionsDecision {
 public:
  std::string statement_name_{""};
  int swizzle_size_{-1};
  long int max_consecutive_bytes_{-1};
  bool partial_swizzle_{false};
  std::vector<std::string> dimensions_;

  DimensionsDecision(void) {}
  DimensionsDecision(const std::string &statement, const std::vector<std::string> &dimensions,
                     const std::vector<AccessInfo> &info);
  int BestOutermostSwizzleDimension(const isl::set &statement, const std::string &name) const;
  float Score(void) const;
  std::map<std::string, int> DimensionsMap(const isl::set &statement) const;
  std::string DimensionConstraint(ScopInfo &scop_info, const std::vector<std::string> &names,
                                  const std::map<std::string, int> &depth, const bool swizzle, int dimension,
                                  unsigned int shift) const;
  std::string SoftConstraints(ScopInfo &scop_info, const isl::set &statement, unsigned int shift) const;
  bool operator>(const DimensionsDecision &element) const;
  void Log(const std::string &prefix = "") const;
};

class DimensionAnalysis {
 public:
  ScopInfo &scop_info_;
  isl::schedule initial_;
  std::map<std::string, std::vector<AccessInfo>> info_;
  std::map<std::string, std::vector<DimensionsDecision>> analysis_;

  DimensionAnalysis(ScopInfo &scop_info, const isl::schedule &initial);
  void ExtractAccessInfo(ScopInfo &scop_info, const std::string &type, const isl::union_map &accesses);
  void Compute(ScopInfo &scop_info);
  DimensionsDecision SelectDimensions(const std::string &statement) const;
  void Log(void) const;
};

////////////////////////////////////////////////////////////////////////////////
// MindTrickType functions
////////////////////////////////////////////////////////////////////////////////

std::string to_string(MindTrickType t) {
  std::string result = "";
  switch (t) {
    case MindTrickType::manual:
      result = "manual";
      break;
    case MindTrickType::autogen:
      result = "autogen";
      break;
    case MindTrickType::none:
    default:
      result = "none";
      break;
  }
  return result;
}

MindTrickType MindTrickTypeFromString(const std::string &str) {
  if (str == "manual") {
    return MindTrickType::manual;
  } else if (str == "autogen") {
    return MindTrickType::autogen;
  }

  return MindTrickType::none;
}

////////////////////////////////////////////////////////////////////////////////
// GpuConfig
////////////////////////////////////////////////////////////////////////////////

bool GpuConfig::CanPrepareMappingOuterBand(const isl::schedule &schedule) const {
  if (!block_sizes_.empty() || !thread_sizes_.empty()) {
    LOG(WARNING) << "CanPrepareMappingOuterBand: GpuConfig sizes are already set";
  }

  if (block_dimensions_.empty() || thread_dimensions_.empty()) {
    return false;
  }

  if (block_dimensions_[0] != 0) {
    return false;
  }

  const int block_max = *std::max_element(block_dimensions_.begin(), block_dimensions_.end());
  const int thread_min = *std::min_element(thread_dimensions_.begin(), thread_dimensions_.end());
  if (block_max >= thread_min) {
    return false;
  }

  const isl::schedule_node &root = schedule.root();
  const isl::schedule_node_band &band = root.child(0).as<isl::schedule_node_band>();
  const int size = static_cast<int>(band.n_member());
  const int thread_max = *std::max_element(thread_dimensions_.begin(), thread_dimensions_.end());
  if (size <= thread_max) {
    return false;
  }

  return true;
}

void GpuConfig::OffsetThreadDimensions(int offset) {
  const std::size_t size = thread_dimensions_.size();
  for (std::size_t i = 0; i < size; ++i) {
    thread_dimensions_[i] += offset;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Additional classes for the implementation (not exposed in the header)
////////////////////////////////////////////////////////////////////////////////

static bool ShouldAutogenSwizzleDimension(ScopInfo &scop_info) {
  bool result = false;

  const std::string &target = scop_info.user_config_.GetTarget();
  if (target == TARGET_CUDA) {
    result = true;
  }

  const char *const env_autogen_swizzle = std::getenv(kEnvStringMindTricksAutogenSwizzle);
  if (env_autogen_swizzle && std::string(env_autogen_swizzle) == "false") {
    result = false;
  }

  const char *const env_disable_swizzle = std::getenv("MS_AKG_DISABLE_SWIZZLE");
  if (env_disable_swizzle && std::string(env_disable_swizzle) == "1") {
    result = false;
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////
// AccessInfo
////////////////////////////////////////////////////////////////////////////////

AccessInfo::AccessInfo(ScopInfo &scop_info, const std::string &name, const std::string &type, const isl::map &access)
    : statement_(name), type_(type), access_(access), data_(access.tuple_id(isl_dim_out).to_str()) {
  data_type_ = scop_info.GetDtypeOf(data_);
  bytes_ = scop_info.user_config_.GetDataBytes(data_);

  dimensions_ = OrderInputDimsForInnermostAccess(access_);
  swizzle_size_ = AccessSwizzleSize(access_);

  dimensions_sizes_ = DimensionsBytes(access_, bytes_);
  consecutive_bytes_ = ConsecutiveBytes(dimensions_sizes_);
}

std::vector<long int> AccessInfo::DimensionsBytes(const isl::map &access, const int bytes) {
  std::vector<long int> result;

  // We select "best" input dimensions based on the output dimensions:
  // our output sizes are already in the order we are looking for.
  const isl::set &lexmax = access.range().lexmax();
  const unsigned int size = isl_set_dim(lexmax, isl_dim_set);
  for (unsigned int i = 0; i < size; ++i) {
    const long int value = isl_set_plain_get_num_si(lexmax, i) + 1;
    result.push_back(value);
  }
  std::reverse(result.begin(), result.end());

  return result;
}

std::vector<long int> AccessInfo::ConsecutiveBytes(const std::vector<long int> &sizes) const {
  std::vector<long int> result;

  if (sizes.empty()) {
    return result;
  }

  // Do not forget to multiply the first value to get the actual bytes count!
  result.push_back(sizes[0] * bytes_);
  const std::size_t count = sizes.size();
  for (std::size_t i = 1; i < count; ++i) {
    const int current = result[i - 1] * sizes[i];
    result.push_back(current);
  }

  return result;
}

double AccessInfo::Score(ScopInfo &scop_info, const std::string &dimension,
                         const std::vector<std::string> &subset) const {
  // Some accesses do not have any dimension!
  if (dimensions_.empty()) {
    return .0;
  }

  double result = .0;
  // The further it is in the "perfect" dimensions, the worse it is.
  const auto it = std::find(dimensions_.begin(), dimensions_.end(), dimension);
  const int position = static_cast<int>(dimensions_.size()) - std::distance(dimensions_.begin(), it);
  result += static_cast<double>(position);
  // We are currently dealing with the "ideal" innermost dimension and we want to favour writes
  if (position == static_cast<int>(dimensions_.size()) && type_ == AccessInfo::write) {
    result += 1.0;
  }

  // If we are currently deciding the innermost dimension
  if (subset.empty() && ShouldAutogenSwizzleDimension(scop_info) && dimension == dimensions_[0]) {
    const int swizzle_size = swizzle_size_;
    if (swizzle_size > 1) {
      result += static_cast<double>(swizzle_size) * 3.0;
      // Favor swizzle writes over swizzle reads?
      if (type_ == AccessInfo::write) {
        result += 2.0;
      }
    }
  }

  return result;
}

void AccessInfo::Log(const std::string &prefix) const {
  std::string dimensions_string = "{ ";
  for (auto id : dimensions_) {
    dimensions_string += id + ", ";
  }
  dimensions_string += "}";

  std::string dimensions_sizes_string = "{ ";
  for (auto value : dimensions_sizes_) {
    dimensions_sizes_string += std::to_string(value) + ", ";
  }
  dimensions_sizes_string += "}";

  std::string consecutive_bytes_string = "{ ";
  for (auto value : consecutive_bytes_) {
    consecutive_bytes_string += std::to_string(value) + ", ";
  }
  consecutive_bytes_string += "}";

  std::stringstream stream;
  stream << data_type_;
  std::string data_type_string = stream.str();

  log::Info(text_blue + prefix + data_);
  log::Info(prefix + "  access: " + access_.to_str());
  log::Info(prefix + "  I/O: " + type_);
  log::Info(prefix + "  data type: " + data_type_string);
  log::Info(prefix + "  bytes: " + std::to_string(bytes_));
  log::Info(prefix + "  swizzle_size: " + std::to_string(swizzle_size_));
  log::Info(prefix + "  best dimensions: " + dimensions_string);
  log::Info(prefix + "  sizes: " + dimensions_sizes_string);
  log::Info(prefix + "  consecutive bytes: " + consecutive_bytes_string);
}

int AccessInfo::AccessSwizzleSize(const isl::map &access) {
  constexpr int none = -1;
  const isl::set &range = access.range();
  if (range.n_dim() == 0) {
    return none;
  }

  const unsigned int innermost = range.n_dim() - 1;
  const isl::pw_aff &maxbound = range.dim_max(innermost);
  if (!isl_pw_aff_is_cst(maxbound.get()) || !maxbound.isa_aff()) {
    return none;
  }

  const isl::aff &maxbound_aff = maxbound.as_aff();
  const isl::val &maxbound_val = maxbound_aff.constant_val().add(1);

  if (!maxbound_val.ge(2)) {
    return none;
  } else if (!maxbound_val.ge(4) && maxbound_val.mod(2).eq(0)) {
    constexpr int two_elements_swizzle = 2;
    return two_elements_swizzle;
  } else if (maxbound_val.mod(4).ne(0)) {
    return none;
  }

  constexpr int four_elements_swizzle = 4;
  return four_elements_swizzle;
}

bool AccessInfo::DimensionIsContiguous(const isl::map &access, const unsigned int dimension) {
  // Implementation note:
  // We do not directly use isl::map::range_stride_info() because it does not
  // return the same information as isl::set::stride()!
  const isl::set &range = access.range();
  const isl::val &stride = range.stride(dimension);
  return stride.is_one();
}

std::vector<isl::id> AccessInfo::InvolvedInputDims(const isl::map &access, const unsigned int dimension) {
  const isl::map &drop = isl_map_drop_constraints_not_involving_dims(access, isl_dim_out, dimension, 1);
  const unsigned int input_dimensions = drop.dim(isl_dim_in);

  std::vector<isl::id> iterators;
  for (unsigned int i = 0; i < input_dimensions; ++i) {
    const bool involved = isl_map_involves_dims(drop, isl_dim_in, i, 1);
    if (involved) {
      const isl::id &id = isl_map_get_dim_id(drop, isl_dim_in, i);
      iterators.push_back(id);
    }
  }

  return iterators;
}

std::vector<std::string> AccessInfo::OrderInputDimsForInnermostAccess(const isl::map &access) {
  std::vector<std::string> result;

  // The output vector will be reversed: from innermost dimension at index 0 to outermost dimension at the last index
  // This resulting vector represents the "ideal" scheduling dimension ordering for this access.
  const unsigned int dimensions = access.range().n_dim();
  for (unsigned int i = dimensions; i-- > 0;) {
    const bool contiguous = DimensionIsContiguous(access, i);
    if (!contiguous) {
      break;
    }
    // For now, we do not handle accesses where multiple iterators are involved
    const std::vector<isl::id> &involved = InvolvedInputDims(access, i);
    if (involved.size() != 1) {
      break;
    }

    const std::string &name = involved[0].get_name();
    // To ensure the iterator is not already recorded (may happen for collapsed domains!)
    if (std::find(result.begin(), result.end(), name) == result.end()) {
      result.push_back(name);
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////
// DimensionsDecision
////////////////////////////////////////////////////////////////////////////////

DimensionsDecision::DimensionsDecision(const std::string &statement, const std::vector<std::string> &dimensions,
                                       const std::vector<AccessInfo> &info)
    : statement_name_(statement), dimensions_(dimensions) {
  std::size_t ignored = 0;
  std::vector<int> swizzle_sizes;
  for (const AccessInfo &current_info : info) {
    if (current_info.dimensions_.empty()) {
      ignored += 1;
    } else if (current_info.swizzle_size_ > 1) {
      swizzle_sizes.push_back(current_info.swizzle_size_);
    }
  }
  if (!swizzle_sizes.empty()) {
    const int swizzle_size = *std::min_element(swizzle_sizes.begin(), swizzle_sizes.end());
    if (swizzle_size > 1) {
      swizzle_size_ = swizzle_size;
      partial_swizzle_ = swizzle_sizes.size() != (info.size() - ignored);
    }
  }
}

int DimensionsDecision::BestOutermostSwizzleDimension(const isl::set &statement, const std::string &name) const {
  auto iteration_counts = [](const std::vector<std::string> &dimensions, const isl::set &lexmax) {
    std::vector<long> result;

    result.push_back(1);
    const std::size_t size = dimensions.size();
    for (std::size_t i = 1; i < size; ++i) {
      const std::string &dim = dimensions[i];
      const int position = isl_set_find_dim_by_name(lexmax, isl_dim_set, dim);
      const long size = isl_set_plain_get_num_si(lexmax, position);
      const long product = result[i - 1] * size;
      result.push_back(product);
    }

    return result;
  };
  auto select_dimension = [](const std::vector<long> &counts) {
    int result = 0;

    const long gpu_thread_max = 1024;
    const std::size_t size = counts.size();
    for (std::size_t i = 1; result <= 4 && i < size; ++i) {
      if (counts[i] <= gpu_thread_max) {
        result = i;
      }
    }

    return result;
  };

  const isl::set &lexmax = statement.lexmax();
  const std::vector<long> sizes = iteration_counts(dimensions_, lexmax);
  const int nb_vars = isl_set_dim(statement, isl_dim_set);
  int selection = nb_vars - select_dimension(sizes) - 1;
  if (selection == 0) {
    // The outermost dimension is seldom good for the outerpart of the swizzle
    selection = 1;
  }

  return selection;
}

float DimensionsDecision::Score(void) const {
  float result = .0;

  result += static_cast<float>(swizzle_size_);
  if (max_consecutive_bytes_ >= 1024) {
    result += 3.0;
  } else if (max_consecutive_bytes_ >= 128) {
    result += 1.0;
  }

  return result;
}

std::map<std::string, int> DimensionsDecision::DimensionsMap(const isl::set &statement) const {
  // Compute the scheduling dimension depth
  const int iterators = static_cast<int>(isl_set_dim(statement, isl_dim_set));
  const int decided = static_cast<int>(dimensions_.size());

  std::map<std::string, int> result;
  for (int i = 0; i < decided; ++i) {
    const std::string &current = dimensions_[i];
    const int position = iterators - i - 1;
    result[current] = position;
  }

  return result;
}

std::string DimensionsDecision::DimensionConstraint(ScopInfo &scop_info, const std::vector<std::string> &names,
                                                    const std::map<std::string, int> &depth, const bool swizzle,
                                                    int dimension, unsigned int shift) const {
  const int columns = names.size();

  std::function<std::string(int)> undecided_column;
  if (swizzle) {
    undecided_column = [&shift](int column) {
      std::string result = "?";
      result += std::to_string(shift) + std::to_string(column);
      return result;
    };
  } else {
    undecided_column = [](int column) { return "?"; };
  }

  std::string result = "";
  for (int c = 0; c < columns; ++c) {
    const std::string &current = names[c];
    if (depth.find(current) != depth.end()) {
      const int target = depth.at(current);
      const std::string &coefficient = target == dimension ? "1" : "0";
      result += coefficient;
    } else {
      result += undecided_column(c);
    }
    result += ", ";
  }
  // constant coefficient
  result += undecided_column(columns);

  // Do not forget to enclose the array
  result = "[" + result += "]";

  return result;
}

std::string DimensionsDecision::SoftConstraints(ScopInfo &scop_info, const isl::set &statement,
                                                unsigned int shift) const {
  const std::string &statement_name = statement.get_tuple_name();
  picojson::object contents;
  contents["statement"] = picojson::value(statement_name);

  const std::vector<std::string> &vars = isl_set_all_names(statement);
  const int nb_vars = isl_set_dim(statement, isl_dim_set);
  const int parameters = isl_set_dim(statement, isl_dim_param);
  picojson::array meta;
  meta.push_back(picojson::value(std::to_string(nb_vars)));
  meta.push_back(picojson::value(std::to_string(parameters)));
  contents["meta"] = picojson::value(meta);

  const int swizzle_size = swizzle_size_;
  const bool should_swizzle = ShouldAutogenSwizzleDimension(scop_info) && swizzle_size > 1;

  picojson::array coefficients;
  const std::map<std::string, int> map = DimensionsMap(statement);
  for (int i = 0; i < nb_vars; ++i) {
    const bool swizzle = should_swizzle && i == nb_vars - 1;
    const std::string &constraint = DimensionConstraint(scop_info, vars, map, swizzle, i, shift);
    if (swizzle) {
      std::string first = constraint + " (/" + std::to_string(swizzle_size) + ")";
      std::string second = constraint + " (%" + std::to_string(swizzle_size) + ")";
      coefficients.push_back(picojson::value(first));
      coefficients.push_back(picojson::value(second));
    } else {
      coefficients.push_back(picojson::value(constraint));
    }
  }

  // We should move the outermost part of the stripmined swizzle dimension if not all accesses
  // are swizzled (to try to coalesce non swizzled accesses)
  if (should_swizzle && partial_swizzle_) {
    const std::string &swizzled_dim = dimensions_[0];
    const int target = BestOutermostSwizzleDimension(statement, swizzled_dim);
    // Only move if target is a valid dimension and is not the last dimension!
    // (BestSecondaryOuterMostSchedulingDimension() should not return the last dimension as it looks in partially
    // satisfied I/O!)
    if (target >= 0 && target < nb_vars - 1) {
      const int old_position = nb_vars - 1;
      picojson::value constraint = coefficients[old_position];
      coefficients.erase(coefficients.begin() + old_position);
      coefficients.insert(coefficients.begin() + target, constraint);
    }
  }

  contents["coefficients"] = picojson::value(coefficients);

  const picojson::value &trick = picojson::value(contents);
  const std::string &result = trick.serialize();

  return result;
}

bool DimensionsDecision::operator>(const DimensionsDecision &element) const { return Score() > element.Score(); }

void DimensionsDecision::Log(const std::string &prefix) const {
  std::string dimensions_string = "{ ";
  for (auto id : dimensions_) {
    dimensions_string += id + ", ";
  }
  dimensions_string += "}";

  log::Info(prefix + text_blue "dimensions: " + dimensions_string);
  log::Info(prefix + "  score: " + std::to_string(Score()));
  log::Info(prefix + "  swizzle_size: " + std::to_string(swizzle_size_));
  log::Info(prefix + "  partial_swizzle: " + std::to_string(partial_swizzle_));
  log::Info(prefix + "  max consecutive bytes: " + std::to_string(max_consecutive_bytes_));
}

////////////////////////////////////////////////////////////////////////////////
// DimensionAnalysis
////////////////////////////////////////////////////////////////////////////////

DimensionAnalysis::DimensionAnalysis(ScopInfo &scop_info, const isl::schedule &initial)
    : scop_info_(scop_info), initial_(initial) {
  const isl::union_map &reads = scop_info.analysis_result_.GetReads();
  const isl::union_map &writes = scop_info.analysis_result_.GetWrites();
  ExtractAccessInfo(scop_info, AccessInfo::read, reads);
  ExtractAccessInfo(scop_info, AccessInfo::write, writes);
  Compute(scop_info);
  if (log::GetVerbosityLevel() >= log::Verbosity::high) {
    Log();
  }
}

void DimensionAnalysis::ExtractAccessInfo(ScopInfo &scop_info, const std::string &type,
                                          const isl::union_map &accesses) {
  const isl::map_list &list = accesses.get_map_list();
  const unsigned int size = list.size();
  for (unsigned int i = 0; i < size; ++i) {
    const isl::set &statement = list.get_at(i).domain().unwrap().domain();
    const std::string &statement_name = statement.get_tuple_id().get_name();
    const isl::map &access = list.get_at(i).flatten_domain().set_tuple_name(isl_dim_in, statement_name);

    const AccessInfo &current = AccessInfo(scop_info, statement_name, type, access);
    info_[statement_name].push_back(current);
  }
}

void DimensionAnalysis::Compute(ScopInfo &scop_info) {
  // First pass: create decisions and record as much as possible
  for (auto it : info_) {
    const std::vector<AccessInfo> &info = it.second;
    if (info.empty()) {
      continue;
    }

    const std::string statement = it.first;
    // New alternative!
    {
      const isl::set &domain = info[0].access_.domain();
      std::vector<std::string> iterators = isl_set_dim_names(domain, isl_dim_set);

      std::vector<std::string> subset;
      while (!iterators.empty()) {
        std::map<std::string, double> scores;
        for (auto iterator : iterators) {
          double score = .0;
          for (const AccessInfo &current_info : info) {
            double current_score = current_info.Score(scop_info, iterator, subset);
            score += current_score;
          }
          scores[iterator] = score;
        }

        using pair_type = decltype(scores)::value_type;
        auto score_comparator = [](const pair_type &p1, const pair_type &p2) { return p1.second < p2.second; };
        const auto best_dimension = std::max_element(scores.begin(), scores.end(), score_comparator);
        const std::string &selection = best_dimension->first;
        subset.push_back(selection);

        const auto last = std::remove(iterators.begin(), iterators.end(), selection);
        iterators.erase(last, iterators.end());
      }

      const DimensionsDecision &decision = DimensionsDecision(statement, subset, info);
      if (log::GetVerbosityLevel() >= log::Verbosity::high) {
        decision.Log();
      }
      analysis_[statement].push_back(decision);
    }
  }

  // Then we order our decisions
  for (auto it : analysis_) {
    std::sort(it.second.begin(), it.second.end(), std::greater<DimensionsDecision>());
    analysis_[it.first] = it.second;
  }
}

DimensionsDecision DimensionAnalysis::SelectDimensions(const std::string &statement) const {
  if (analysis_.find(statement) != analysis_.end() && !analysis_.at(statement).empty()) {
    // It should have been sorted so we just select the first one.
    return analysis_.at(statement)[0];
  }

  DimensionsDecision result;
  return result;
}

void DimensionAnalysis::Log(void) const {
  log::Info(text_bold text_reverse " AccessInfo ");
  for (auto it : info_) {
    const std::string &statement = it.first;
    const std::vector<AccessInfo> &info = it.second;
    log::Info(text_bold text_blue "  " + statement);
    for (auto current : info) {
      current.Log("    ");
    }
  }
  log::Info("");
  log::Info(text_bold text_reverse " DimensionsDecision ");
  for (auto it : analysis_) {
    const std::string &statement = it.first;
    const std::vector<DimensionsDecision> &analysis = it.second;
    log::Info(text_bold text_blue "  " + statement);
    for (auto current : analysis) {
      current.Log("    ");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Constructors and friends
////////////////////////////////////////////////////////////////////////////////

SchedulingMindTrick::SchedulingMindTrick(PassInfo &pass_info, ScopInfo &scop_info, int verbosity)
    : pass_info_(pass_info), scop_info_(scop_info) {
  if (verbosity >= 0) {
    verbosity_ = verbosity;
  }
}

SchedulingMindTrick::~SchedulingMindTrick() {}

void SchedulingMindTrick::Load(const std::string &filename) {
  // Set filename as the mind_trick's default name.
  // It may be overriden when parsing the file contents.
  name_ = filename;

  std::ifstream stream(filename);
  if (stream.is_open()) {
    Parse(stream);
  }
}

std::istream &SchedulingMindTrick::Parse(std::istream &streamed_json) {
  picojson::value json;

  const std::string error = picojson::parse(json, streamed_json);
  if (!error.empty()) {
    Error(error);
    correctly_parsed_ = false;
    return streamed_json;
  }

  // Parse the json representation
  correctly_parsed_ = true;
  Parse(json);

  return streamed_json;
}

void SchedulingMindTrick::Parse(const std::string &serialized_json) {
  // Parse the serialized string representation
  picojson::value json;
  const std::string error = picojson::parse(json, serialized_json);

  if (!error.empty()) {
    Error(error);
    correctly_parsed_ = false;
    return;
  }

  // Parse the json representation
  correctly_parsed_ = true;
  Parse(json);
}

picojson::value SchedulingMindTrick::maybe(const picojson::value &node, const std::string &key) {
  if (!node.is<picojson::object>()) {
    poly::log::Error("cannot parse input JSON (not an \"object\")");
    picojson::value v;
    return v;
  } else {
    return maybe(node.get<picojson::object>(), key);
  }
}

picojson::value SchedulingMindTrick::maybe(const picojson::object &node, const std::string &key) {
  if (node.find(key) != node.end()) {
    return node.at(key);
  } else {
    picojson::value v;
    return v;
  }
}

std::vector<int> SchedulingMindTrick::to_int_vector(const picojson::value &node) {
  if (!node.is<picojson::array>()) {
    std::vector<int> v;
    return v;
  } else {
    return to_int_vector(node.get<picojson::array>());
  }
}

std::vector<int> SchedulingMindTrick::to_int_vector(const picojson::array &node) {
  std::vector<int> values;
  for (auto value : node) {
    if (value.is<double>()) {
      const double &d = value.get<double>();
      values.push_back((int)d);
    } else if (value.is<std::string>()) {
      const std::string &s = value.get<std::string>();
      values.push_back(std::stoi(s));
    }
  }
  return values;
}

std::vector<std::string> SchedulingMindTrick::to_string_vector(const picojson::value &node) {
  if (!node.is<picojson::array>()) {
    std::vector<std::string> v;
    return v;
  } else {
    return to_string_vector(node.get<picojson::array>());
  }
}

std::vector<std::string> SchedulingMindTrick::to_string_vector(const picojson::array &node) {
  std::vector<std::string> values;
  for (auto value : node) {
    if (value.is<std::string>()) {
      const std::string &s = value.get<std::string>();
      values.push_back(s);
    }
  }
  return values;
}

std::vector<std::string> SchedulingMindTrick::split_string(std::string str, std::string delim) const {
  std::vector<std::string> results;
  std::size_t current, previous = 0;
  current = str.find(delim);
  while (current != std::string::npos) {
    results.push_back(str.substr(previous, current - previous));
    previous = current + 1;
    current = str.find(delim, previous);
  }
  results.push_back(str.substr(previous, current - previous));

  return results;
}

void SchedulingMindTrick::Parse(const picojson::value &root) {
  if (root.is<picojson::null>()) {
    Error("cannot parse input JSON (null)");
    correctly_parsed_ = false;
    return;
  } else if (!root.is<picojson::object>()) {
    Error("cannot parse input JSON (not an \"object\")");
    correctly_parsed_ = false;
    return;
  }

  // Parse main characteristics
  ParseName(maybe(root, "name"));
  ParseDisabledPasses(maybe(root, "disable"));
  ParseSchedule(maybe(root, "schedule"));
  ParsePattern(maybe(root, "pattern"));
  ParseOperatorName(maybe(root, "operator"));
  ParseDomain(maybe(root, "domain"));
  ParseExplicitTree(maybe(root, "tree"));
  ParseGpuInfo(maybe(root, "gpu"));
  ParseCheckSchedule(maybe(root, "check schedule"));
  ParseAttrs(maybe(root, "attrs"));
  ParseVerbosity(maybe(root, "verbosity"));
#ifdef AKG_USE_MLS
  hints_ = mls::bin::Hints(root);
#endif
}

void SchedulingMindTrick::ParseName(const picojson::value &node) {
  if (!node.is<std::string>()) {
    return;
  }

  name_ = node.get<std::string>();
}

void SchedulingMindTrick::ParseDisabledPasses(const picojson::value &node) {
  if (!node.is<picojson::array>()) {
    return;
  }

  std::set<std::string> passes;
  for (auto pass : node.get<picojson::array>()) {
    if (pass.is<std::string>()) {
      const std::string &name = pass.get<std::string>();
      passes.insert(name);
    }
  }
  disabled_passes_ = passes;
}

void SchedulingMindTrick::ParseSchedule(const picojson::value &node) {
  if (node.is<std::string>()) {
    suggested_schedule_string_ = node.get<std::string>();
  } else if (node.is<picojson::object>()) {
    suggested_schedule_string_ = node.serialize();
  } else if (node.is<picojson::array>()) {
    suggested_schedule_vector_ = to_string_vector(node);
  }
}

void SchedulingMindTrick::ParsePattern(const picojson::value &node) {
  std::string str;
  if (node.is<std::string>()) {
    pattern_ = node.get<std::string>();
  } else if (node.is<picojson::object>()) {
    pattern_ = node.serialize();
  }
}

void SchedulingMindTrick::ParseOperatorName(const picojson::value &node) {
  if (!node.is<std::string>()) {
    return;
  }
  operator_ = node.get<std::string>();
}

void SchedulingMindTrick::ParseDomain(const picojson::value &node) {
  isl::ctx ctx = scop_info_.GetCtx();

  isl::union_set domain;
  if (node.is<std::string>()) {
    const std::string &str = node.get<std::string>();
    domain = isl::union_set(ctx, str);
  } else if (node.is<picojson::array>()) {
    const std::vector<std::string> &subdomains = to_string_vector(node);
    if (!subdomains.empty()) {
      const std::size_t count = subdomains.size();
      const std::string &first_string = subdomains[0];
      domain = isl::union_set(ctx, first_string);
      for (std::size_t i = 1; i < count; ++i) {
        const std::string &str = subdomains[i];
        const isl::union_set &current = isl::union_set(ctx, str);
        domain = domain.unite(current);
      }
    }
  }

  if (domain) {
    domain_ = domain;
  }
}

void SchedulingMindTrick::ParseGpuInfo(const picojson::value &node) {
  if (!node.is<picojson::object>()) {
    return;
  }

  // Parse block info
  const picojson::value &blocks = maybe(node, "blocks");
  gpu_info_.block_dimensions_ = to_int_vector(blocks);

  // Parse thread info
  const picojson::value &threads = maybe(node, "threads");
  gpu_info_.thread_dimensions_ = to_int_vector(threads);

  const picojson::value &swizzle = maybe(node, "swizzle");
  gpu_info_.swizzle_dimensions_ = to_int_vector(swizzle);

  const picojson::value &flags = maybe(node, "compiler flags");
  gpu_info_.compiler_flags_ = to_string_vector(flags);

  const picojson::value &automap = maybe(node, "automap");
  if (automap.is<bool>()) {
    gpu_info_.automap_ = automap.get<bool>();
  }
}

void SchedulingMindTrick::ParseExplicitTree(const picojson::value &node) {
  std::string str;
  if (node.is<std::string>()) {
    str = node.get<std::string>();
  } else if (node.is<picojson::object>()) {
    str = node.serialize();
  } else {
    return;
  }

  isl::ctx ctx = scop_info_.GetCtx();
  const isl::schedule &schedule = isl::schedule(ctx, str);
  if (schedule) {
    explicit_tree_ = schedule;
  }
}

void SchedulingMindTrick::ParseCheckSchedule(const picojson::value &node) {
  if (!node.is<bool>()) {
    return;
  }

  check_schedule_ = node.get<bool>();
}

void SchedulingMindTrick::ParseAttrs(const picojson::value &node) {
  if (!node.is<picojson::object>()) {
    return;
  }

  const picojson::object &values = node.get<picojson::object>();
  for (auto current : values) {
    const std::string &key = current.first;
    const picojson::value &value = current.second;

    // Implementation notes:
    // - no 'value.is<bool>()' test because boolean values are integers in attrs.
    // - JSON only supports floating point numbers (hence picojson only provides doubles)
    // - (for now) all numbers in attrs are integers (including "boolean" values)
    //   + pattern-matching the key may be required if future changes introduce floating point values
    // - dynamic_shape and custom_tiling require more work
    //   + not implemented because not needed in mind tricks for now
    if (key == "dynamic_shape" || key == "custom_tiling") {
      Warn("\"" + key + "\" is not supported in mind tricks attrs");
      continue;
    }

    if (value.is<std::string>()) {
      const std::string &str = value.get<std::string>();
      const air::NodeRef ref = air::ir::StringImm::make(str);
      attrs_.Set(key, ref);
    } else if (value.is<double>()) {
      const double double_value = value.get<double>();
      const int integer_value = static_cast<int>(double_value);
      const air::DataType type = air::Int(32);
      const air::NodeRef ref = air::ir::IntImm::make(type, integer_value);
      attrs_.Set(key, ref);
      if (key == "bypass") {
        attrs_.Set("bypassL1", ref);
      }
    }
  }
}

void SchedulingMindTrick::ParseVerbosity(const picojson::value &node) {
  if (!node.is<double>()) {
    return;
  }

  const int verbosity = static_cast<int>(node.get<double>());
  if (verbosity >= 0) {
    verbosity_ = verbosity;
  }
}

static std::vector<int> FindPrimeFactors(const int value) {
  std::vector<int> factors;

  int val_tmp = value;
  int prime = 2;
  do {
    if (val_tmp % prime == 0) {
      factors.push_back(prime);
      val_tmp /= prime;
      prime = 2;
    } else {
      prime++;
    }
  } while (val_tmp > 1);

  return factors;
}

static std::vector<int> FindDivisors(const int value, std::function<bool(const int)> filter) {
  const std::vector<int> &factors = FindPrimeFactors(value);

  int current = 1;
  std::vector<int> divisors;
  const std::size_t count = factors.size();
  for (std::size_t i = 0; i < count && filter(current * factors[i]); ++i) {
    current *= factors[i];
    divisors.push_back(current);

    for (std::size_t j = i + 1; j < count && filter(current * factors[j]); ++j) {
      const int inner = current * factors[j];
      divisors.push_back(inner);
    }
  }

  std::sort(divisors.begin(), divisors.end());
  auto last = std::unique(divisors.begin(), divisors.end());
  divisors.erase(last, divisors.end());

  return divisors;
}

int SchedulingMindTrick::FindStripmineFactor(int size, int limit, bool greedy) const {
  const int warp_size = 32;
  auto warp_filter = [&limit](const int value) { return value <= limit && !(value % warp_size); };
  std::vector<int> divisors = FindDivisors(size, warp_filter);
  if (divisors.empty()) {
    auto relaxed_filter = [&limit](const int value) { return value <= limit; };
    divisors = FindDivisors(size, relaxed_filter);
  }

  const int threshold = 16;
  if (greedy && limit > threshold) {
    const int divisor = divisors.empty() ? -1 : divisors.back();
    const int result = (divisor < 1 || divisor < sqrt(limit)) ? limit : divisor;
    return result;
  } else {
    const int result = divisors.empty() ? 1 : divisors.back();
    return result;
  }
}

isl::schedule_node_band SchedulingMindTrick::DetectAndSplitSwizzleDim(const isl::schedule_node_band &band,
                                                                      GpuConfig &info) {
  const unsigned int dims = band.n_member();
  if (dims < 2) {
    return band;
  }

  const int innermost = static_cast<int>(dims - 1);
  if (!band.member_get_coincident(innermost)) {
    return band;
  }

  isl::schedule_node_band result = band;

  const isl::set &lexmax = isl_schedule_node_band_lexmax(band);
  const long size = isl_set_plain_get_num_si(lexmax, innermost) + 1;
  log::Info(log::Verbosity::medium, "innermost = " + std::to_string(innermost) + ", size = " + std::to_string(size));
  if (size == 2 || size == 4) {
    const std::vector<mls::bin::InfluenceOperation> operations = hints_.GetInfluence().GetOperations();

    bool is_swizzle_dim = false;
    for (auto operation : operations) {
      const mls::bin::InfluenceOperation::Type type = operation.GetType();
      if (type != mls::bin::InfluenceOperation::kModulo) {
        continue;
      }

      const int target = operation.GetDimension();
      if (target == innermost) {
        is_swizzle_dim = true;
      }
    }

    if (is_swizzle_dim) {
      result = SplitSwizzleDim(result, info, innermost);
      log::Info(log::Verbosity::medium, "swizzle dimension split");
    }
  }

  return result;
}

int SchedulingMindTrick::FindInnermostCoincidentDimension(const isl::schedule_node_band &band) {
  const unsigned int dims = band.n_member();
  int innermost = static_cast<int>(dims);
  for (; innermost-- > 0;) {
    if (band.member_get_coincident(innermost)) {
      break;
    }
  }

  Info(log::Verbosity::medium, "initial innermost: " + std::to_string(innermost));
  return innermost;
}

isl::schedule_node_band SchedulingMindTrick::GpuStripmineUniqueCoincidentDimension(const isl::schedule_node_band &band,
                                                                                   int &innermost,
                                                                                   const int thread_max) {
  const isl::set &lexmax = isl_schedule_node_band_lexmax(band);
  const long size = isl_set_plain_get_num_si(lexmax, innermost) + 1;
  const int limit = std::min(thread_max, (int)size);
  const int stripmine = FindStripmineFactor(size, limit);

  Info(log::Verbosity::medium, "stripmine unique: " + std::to_string(innermost) + " (" + std::to_string(size) + ")");
  const isl::schedule_node_band &result = isl_schedule_node_band_stripmine(band, innermost, stripmine);
  innermost += 1;

  return result;
}

isl::schedule_node_band SchedulingMindTrick::GpuAutomapThreads(const isl::schedule_node_band &band, GpuConfig &config,
                                                               int &innermost, const int thread_max) {
  isl::schedule_node_band result = band;
  // We want to keep at least one dimension for the blocks
  int thread_limit = std::max(innermost - 3, 1);
  Info(log::Verbosity::medium, "thread_limit: " + std::to_string(thread_limit));
  // Map up to 3 dimensions on threads while ensuring the product does not
  // exceed thread_max
  int free_threads = thread_max;

  auto log_progress = [&innermost, &free_threads](const std::string &prefix = "") {
    log::Info(log::Verbosity::medium, (prefix != "" ? (prefix + ": ") : prefix) + "innermost = " +
                                        std::to_string(innermost) + ", free_threads = " + std::to_string(free_threads));
  };

  log_progress("start");
  for (int dim = innermost + 1; free_threads > 1 && dim-- > thread_limit;) {
    const isl::set &lexmax = isl_schedule_node_band_lexmax(result);
    const long size = isl_set_plain_get_num_si(lexmax, dim) + 1;
    const std::string &size_str = " (" + std::to_string(size) + ")";
    if (size > free_threads && result.member_get_coincident(dim)) {
      const int limit = std::min(free_threads, (int)size);
      const int stripmine = FindStripmineFactor(size, limit);
      if (stripmine <= 1 || stripmine > free_threads || stripmine >= size) {
        Info(log::Verbosity::medium, "invalid stripmine value: " + std::to_string(dim) + size_str);
        break;
      }
      result = isl_schedule_node_band_stripmine(result, dim, stripmine);
      // Shift previously mapped dimensions because we added a new dimension!
      config.OffsetThreadDimensions(1);
      // Local variables need to be shifted as well
      dim += 1;
      thread_limit += 1;
      innermost += 1;
      // Finally record our dimension
      config.thread_dimensions_.push_back(dim);
      free_threads /= stripmine;

      log::Info(log::Verbosity::medium, "size: " + std::to_string(size));
      log::Info(log::Verbosity::medium, "stripmine: " + std::to_string(stripmine));
      log::Info(log::Verbosity::medium, "mapping: " + std::to_string(dim));
      log::Info(log::Verbosity::medium, "free_threads: " + std::to_string(free_threads));
    } else if (size <= free_threads && result.member_get_coincident(dim)) {
      config.thread_dimensions_.push_back(dim);
      free_threads /= size;
      log::Info(log::Verbosity::medium, "size: " + std::to_string(size));
      log::Info(log::Verbosity::medium, "mapping: " + std::to_string(dim));
      log::Info(log::Verbosity::medium, "free_threads: " + std::to_string(free_threads));
    } else {
      Info(log::Verbosity::medium, "too big for threads: " + std::to_string(dim) + size_str);
      break;
    }
    log::Info(log::Verbosity::medium, "---");
    log::Info(log::Verbosity::medium, "thread_limit: " + std::to_string(thread_limit));
    log::Info(log::Verbosity::medium, "innermost: " + std::to_string(innermost));
  }
  innermost -= config.thread_dimensions_.size();
  log_progress("post-thread");
  // Info(log::Verbosity::medium, "post thread innermost: " + std::to_string(innermost));

  // Special case: we reached 3 thread dimensions but there is still room for more threads...
  // if there is still room, we need to collapse dimensions (since we already maxed out thread dimensions)
  if (innermost > 0 && config.thread_dimensions_.size() <= 3 && free_threads > 1) {
    bool recheck = true;
    while (recheck) {
      if (innermost <= 0 || free_threads <= 1) {
        break;
      }

      log::Info(log::Verbosity::medium, "free_threads: " + std::to_string(free_threads));
      recheck = false;
      // 1. check if innermost is too big: stripmine it (and update innermost + offset thread dimensions!)
      {
        const isl::set &lexmax = isl_schedule_node_band_lexmax(result);
        const long size = isl_set_plain_get_num_si(lexmax, innermost) + 1;
        if (size > free_threads) {
          const int limit = std::min(free_threads, static_cast<int>(size));
          const int stripmine = FindStripmineFactor(size, limit);
          if (stripmine > 1 && stripmine < size && stripmine <= free_threads) {
            result = isl_schedule_node_band_stripmine(result, 0, stripmine);
            config.OffsetThreadDimensions(1);
            innermost += 1;
            log::Info(log::Verbosity::medium, "size: " + std::to_string(size));
            log::Info(log::Verbosity::medium, "stripmine: " + std::to_string(stripmine));
          }
        }
      }
      // 2. check if innermost can be squeezed into threads: we collapse it with the outermost thread dimension
      {
        const isl::set &lexmax = isl_schedule_node_band_lexmax(result);
        const long size = isl_set_plain_get_num_si(lexmax, innermost) + 1;
        if (size <= free_threads) {
          if (config.thread_dimensions_.size() == 3) {
            result = isl_schedule_node_band_collapse(result, innermost);
            config.OffsetThreadDimensions(-1);
          } else if (config.thread_dimensions_.size() < 3) {
            config.thread_dimensions_.push_back(innermost);
          }
          free_threads /= size;
          innermost -= 1;
          recheck = true;
        }
      }
    }
  }
  log_progress("recheck");

  // We deliberately stopped mapping threads before dimension 0 to preserve it.
  // Other dimensions can be directly mapped if they fit whereas dim 0 must absolutely be stripmined.
  constexpr size_t thread_count_ub = 3;
  if (innermost == 0 && config.thread_dimensions_.size() <= thread_count_ub && free_threads > 1) {
    const isl::set &lexmax = isl_schedule_node_band_lexmax(result);
    const long size = isl_set_plain_get_num_si(lexmax, 0) + 1;
    log::Info(log::Verbosity::medium,
              "last: free_threads=" + std::to_string(free_threads) + ", size=" + std::to_string(size));
    if (size > free_threads) {
      // This is the last ressort so we allow greedy stripmining
      const int limit = std::min(free_threads, static_cast<int>(size));
      const int stripmine = FindStripmineFactor(size, limit, true);
      log::Info(log::Verbosity::medium,
                "last: stripmine=" + std::to_string(stripmine) + ", size=" + std::to_string(size));
      if (stripmine > 1 && stripmine < size && stripmine <= free_threads) {
        result = isl_schedule_node_band_stripmine(result, 0, stripmine);
        if (config.thread_dimensions_.size() < thread_count_ub) {
          // Shift previously mapped dimensions because we added a new dimension!
          config.OffsetThreadDimensions(1);
          // In this special case, innermost is still 0 and the new thread dimension is 1
          config.thread_dimensions_.push_back(1);
        } else {
          result = isl_schedule_node_band_collapse(result, 1);
          // We do not need to offset the thread dimensions!
          // We stripmined then collapsed: dimension count did not change.
        }
        free_threads /= stripmine;
      }
    }
  }
  log_progress("end");

  return result;
}

isl::schedule_node_band SchedulingMindTrick::GpuCollapseRemainingDimensions(const isl::schedule_node_band &band,
                                                                            GpuConfig &config, int &innermost) {
  // At this point, more than 3 dimensions remain: we need to collapse some
  isl::schedule_node_band result = band;
  for (int dim = innermost; dim-- > 2;) {
    result = isl_schedule_node_band_collapse(result, dim);
  }
  // We also need to offset the thread dimensions!
  const int offset = innermost - 2;
  config.OffsetThreadDimensions(-offset);
  // Do not forget to update innermost!
  innermost = 2;

  return result;
}

isl::schedule_node_band SchedulingMindTrick::GpuAutomapBlocks(const isl::schedule_node_band &band, GpuConfig &config,
                                                              int &innermost) {
  // Note: this method does not actually need to edit the band or innermost...
  // We only use this prototype to be consistent with the couple other methods
  // used in GpuAutomap().

  Info(log::Verbosity::medium, "pre block innermost: " + std::to_string(innermost));
  const isl::set &lexmax = isl_schedule_node_band_lexmax(band);
  for (int dim = innermost + 1; dim-- > 0;) {
    const long size = isl_set_plain_get_num_si(lexmax, dim) + 1;
    const std::string &size_str = " (" + std::to_string(size) + ")";
    if (band.member_get_coincident(dim)) {
      config.block_dimensions_.push_back(dim);
      Info(log::Verbosity::medium, "mapping block: " + std::to_string(dim) + size_str);
    }
  }

  return band;
}

isl::schedule SchedulingMindTrick::GpuAutomap(const isl::schedule &schedule, GpuConfig &config) {
  config.block_dimensions_.clear();
  config.thread_dimensions_.clear();

  isl::schedule_node_band band = schedule.root().child(0).as<isl::schedule_node_band>();
  // First split the swizzle dimension into a separate schedule node before we attempt to automap
  band = DetectAndSplitSwizzleDim(band, config);

  int innermost = FindInnermostCoincidentDimension(band);
  // Note: config and innermost are passed by reference to the following methods and will be modified!
  if (innermost == 0) {
    band = GpuStripmineUniqueCoincidentDimension(band, innermost, gpu_thread_max_);
  }
  band = GpuAutomapThreads(band, config, innermost, gpu_thread_max_);
  if (innermost > 2) {
    band = GpuCollapseRemainingDimensions(band, config, innermost);
  }
  band = GpuAutomapBlocks(band, config, innermost);

  if (config.block_dimensions_.empty() || config.thread_dimensions_.empty()) {
    Warn(log::Verbosity::veryLow, "could not automap");
    return schedule;
  } else {
    std::sort(config.block_dimensions_.begin(), config.block_dimensions_.end());
    std::sort(config.thread_dimensions_.begin(), config.thread_dimensions_.end());

    config.was_automapped_ = true;
    scop_info_.user_config_.SetMindTrickGpuHasMapping(true);

    const isl::schedule &result = band.schedule();
    return result;
  }
}

bool SchedulingMindTrick::BuildInfluencedSchedule(const isl::schedule &schedule) {
#ifdef AKG_USE_MLS
  if (hints_.Empty()) {
    return false;
  }

  isl_union_map *const dependences = pass_info_.dependences_.get();
  isl_schedule *const initial_schedule = schedule.get();

  const mls::bin::Options options = MLSchedOptionsInit(scop_info_);
  if (options.ShouldLogInternalDebugging()) {
    LOG(INFO) << "MLSched v." << mls::bin::VersionString();
    LOG(INFO) << options.String();
  }

  const std::string &kernel_name = scop_info_.user_config_.GetKernelName();
  mls::bin::Scop scop(initial_schedule, dependences, hints_, options, kernel_name);
  const bool success = scop.ComputeSchedule();

  if (options.ShouldLogInternalDebugging()) {
    LOG(INFO) << scop.String(options) << std::endl;
  }
  if (!success) {
    return false;
  }

  isl::schedule result = isl::manage(scop.ToIslSchedule(schedule.ctx().get()));

  DebugSchedule(result, "Adjusted");
  result = GpuPostProcessSchedule(result, gpu_info_);

  influenced_schedule_ = result;
  return true;
#else
  return false;
#endif
}

void SchedulingMindTrick::BuildSuggestedSchedule(const isl::schedule &initial) {
  if (suggested_schedule_string_ == "" && suggested_schedule_vector_.empty()) {
    Info(log::Verbosity::medium, "will not build a suggested schedule (no suggestion available)");
    return;
  }

  const std::string &target = scop_info_.user_config_.GetTarget();

  // First compute the schedule suggestion
  isl::schedule schedule = ComputeScheduleSuggestion(initial);
  schedule = GpuPostProcessSchedule(schedule, gpu_info_);
  suggested_schedule_ = schedule;
}

isl::schedule SchedulingMindTrick::ComputeScheduleSuggestion(const isl::schedule &initial) {
  // For "suggestions", "computing" the schedule suggestion amounts to converting the input string...
  isl::ctx ctx = scop_info_.GetCtx();

  const isl::union_set &domain = domain_ ? domain_ : initial.domain();

  isl::multi_union_pw_aff partial;
  if (suggested_schedule_string_ != "") {
    partial = isl::multi_union_pw_aff(ctx, suggested_schedule_string_);
  } else if (!suggested_schedule_vector_.empty()) {
    const isl::union_pw_aff &first = isl::union_pw_aff(ctx, suggested_schedule_vector_[0]);
    isl::union_pw_aff_list list = isl::union_pw_aff_list(first);

    const std::size_t count = suggested_schedule_vector_.size();
    for (std::size_t i = 1; i < count; ++i) {
      const isl::union_pw_aff &current = isl::union_pw_aff(ctx, suggested_schedule_vector_[i]);
      list = list.add(current);
    }

    const isl::space &domain_space = domain.space();
    const int params = domain_space.dim(isl_dim_param);

    isl::space space = isl_space_set_alloc(ctx, params, count);
    space = isl_space_copy_param_names(space, domain_space);

    partial = isl::multi_union_pw_aff(space, list);
  }

  isl::schedule result = isl::schedule::from_domain(domain);
  if (partial) {
    result = result.insert_partial_schedule(partial);
  }

  return result;
}

isl::schedule_node_band SchedulingMindTrick::SplitSwizzleDim(const isl::schedule_node_band &band, GpuConfig &info,
                                                             int dimension) {
  isl::ctx ctx = band.ctx();
  const isl::id &marker = isl::id(ctx, MIND_TRICKS_PRESERVE_DIMENSION_MARKER);
  const isl::schedule_node_band &split = band.split(dimension);
  const isl::schedule_node &mark = split.child(0).insert_mark(marker);
  const isl::schedule_node_band &result = mark.parent().as<isl::schedule_node_band>();
  info.has_swizzle_dim_ = true;

  return result;
}

void SchedulingMindTrick::GpuPrepareMappingOuterBandFindSizes(const isl::schedule &schedule, GpuConfig &info) {
  if (!info.block_sizes_.empty() && !info.thread_sizes_.empty()) {
    return;
  }

  const isl::schedule_node &root = schedule.root();
  isl::schedule_node_band band = root.child(0).as<isl::schedule_node_band>();
  const isl::set &values = isl_schedule_node_band_lexmax(band);
  if (!values.is_null()) {
    if (info.block_sizes_.size() == 0) {
      info.block_sizes_ = isl_set_lexmax_extract_upper_bounds(values, info.block_dimensions_);
    }
    if (info.thread_sizes_.size() == 0) {
      info.thread_sizes_ = isl_set_lexmax_extract_upper_bounds(values, info.thread_dimensions_);
    }
  } else {
    Warn("can not retrieve blocks/threads sizes");
  }
}

isl::schedule_node_band SchedulingMindTrick::GpuPrepareMappingOuterBandTrustUser(const isl::schedule_node_band &band,
                                                                                 const GpuConfig &info) {
  const int size = static_cast<int>(band.n_member());
  const std::vector<int> &block_dimensions = info.block_dimensions_;
  const std::vector<int> &thread_dimensions = info.thread_dimensions_;

  isl::schedule_node_band result = band;
  // MappingOuterBand looks in permutable bands.
  result = result.set_permutable(1);
  // Enforce coincidence for block and thread dimensions (in between dimensions will not matter)
  for (auto dimension : block_dimensions)
    if (dimension < size) {
      result = result.member_set_coincident(dimension, 1);
    }
  for (auto dimension : thread_dimensions)
    if (dimension < size) {
      result = result.member_set_coincident(dimension, 1);
    }
  // Explicitly remove coincidence for innermost dimensions (deeper than thread dimensions)
  int innermost_start = 0;
  if (!thread_dimensions.empty()) {
    innermost_start = 1 + *std::max_element(thread_dimensions.begin(), thread_dimensions.end());
  } else if (!block_dimensions.empty()) {
    innermost_start = 1 + *std::max_element(block_dimensions.begin(), block_dimensions.end());
  }
  for (int dimension = innermost_start; dimension < size; ++dimension) {
    result = result.member_set_coincident(dimension, 0);
  }
  if (info.swizzle_dimensions_.size() == 1) {
    const int swizzle_dim = info.swizzle_dimensions_[0];
    result = SplitSwizzleDim(result, gpu_info_, swizzle_dim);
  }
  // We need to inform subsequent passes we have already decided on dimensions mapping
  scop_info_.user_config_.SetMindTrickGpuHasMapping(true);

  return result;
}

isl::schedule SchedulingMindTrick::GpuPrepareMappingOuterBand(const isl::schedule &schedule, GpuConfig &info) {
  if (!info.CanPrepareMappingOuterBand(schedule)) {
    return schedule;
  }

  // We previously decided what dimensions to map but MappingOuterBand needs sizes
  GpuPrepareMappingOuterBandFindSizes(schedule, info);

  const isl::schedule_node &root = schedule.root();
  isl::schedule_node_band band = root.child(0).as<isl::schedule_node_band>();

  // This is for debugging purposes! This will not be triggered in normal use.
  const bool trust_user = !influenced_schedule_ && type_ != MindTrickType::autogen;
  if (trust_user) {
    band = GpuPrepareMappingOuterBandTrustUser(band, info);
  }
  // Split blocks and threads into separate nodes (especially for reduce operators)
  const std::vector<int> &thread_dimensions = info.thread_dimensions_;
  const int outermost_thread = *std::min_element(thread_dimensions.begin(), thread_dimensions.end());
  band = band.split(outermost_thread);
  // This is for debugging purposes! This will not be triggered in normal use.
  if (trust_user) {
    band = band.child(0).as<isl::schedule_node_band>();
    band = band.set_permutable(1);
  }

  const isl::schedule &result = band.schedule();
  return result;
}

bool SchedulingMindTrick::GpuShouldAutomap(const GpuConfig &config) const {
  const bool autogen_automap = scop_info_.user_config_.GetMindTrickGpuAutogenAutomap();
  const bool should_automap = (type_ != MindTrickType::manual && autogen_automap) || config.automap_;
  return should_automap;
}

isl::schedule SchedulingMindTrick::GpuAutoDisablePasses(const isl::schedule &schedule, const GpuConfig &config) {
  // Note: we don't actually use the schedule, for now.
  // We just use the same prototype as other post processing methods.
  bool disable_tiling = type_ != MindTrickType::manual;
  disable_tiling = disable_tiling && config.was_automapped_;
  disable_tiling = disable_tiling && !config.block_sizes_.empty();
  disable_tiling = disable_tiling && !config.thread_sizes_.empty();
  if (disable_tiling) {
    disabled_passes_.insert("TileOuterBand");
  }

  return schedule;
}

isl::schedule SchedulingMindTrick::GpuPostProcessSchedule(const isl::schedule &schedule, GpuConfig &info) {
  const std::string &target = scop_info_.user_config_.GetTarget();
  if (target != TARGET_CUDA) {
    return schedule;
  }

  isl::schedule result = schedule;

  // We want gpu_info_.automap_ to override the value if it is true.
  const bool should_automap = GpuShouldAutomap(info);
  if (should_automap) {
    result = GpuAutomap(result, info);
  }
  result = GpuPrepareMappingOuterBand(result, info);
  result = GpuAutoDisablePasses(result, info);

  return result;
}

///////////////////////////////////////////////////////////////////////////
// Directives utils
///////////////////////////////////////////////////////////////////////////

#ifdef AKG_USE_MLS
void SchedulingMindTrick::ExtractDirectivesFromAKG(void) {
  ForTypeMap directives = scop_info_.analysis_result_.GetForTypeMap();
  std::map<std::string, std::vector<int>> serials_dir;
  std::map<std::string, std::vector<int>> vectorials_dir;
  std::map<std::string, std::vector<int>> parallels_dir;
  for (const auto &[stmt, vloop_directive] : directives) {
    std::string stmt_string = stmt.get_name();
    for (uint i = 0; i < vloop_directive.size(); ++i) {
      switch (vloop_directive[i]) {
        case ForType::Serial:
          break;
        case ForType::Invariant:
          LOG(INFO) << "invariant_for";
          serials_dir[stmt_string].push_back(i);
          break;
        case ForType::Parallel:
          LOG(INFO) << "parallel";
          parallels_dir[stmt_string].push_back(i);
          break;
        case ForType::Vectorized:
        case ForType::Swizzled:  // treat "Swizzled" like "Vectorized" for the moment
          LOG(INFO) << "vectorized";
          vectorials_dir[stmt_string].push_back(i);
          break;
        case ForType::Unrolled:
          LOG(WARNING) << "Do not treat ForType::Unrolled as a directives";
          break;
      }
    }
  }

  hints_.SetSerials(serials_dir);
  hints_.SetVectorials(vectorials_dir);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// MindTrick use
////////////////////////////////////////////////////////////////////////////////

isl::schedule SchedulingMindTrick::Apply(const isl::schedule &sch) {
  // 1. Attempt to influence a schedule.
  // 2. Attempt to build a suggested schedule.
  // 3. See if there is an explicit tree.

  // Attempt to influence a schedule
  if (!hints_.Empty()) {
    BuildInfluencedSchedule(sch);
  }
  // Only attempt to build the full suggestion if the influence failed.
  if (!influenced_schedule_) {
    BuildSuggestedSchedule(sch);
  }
  // Could neither build an influenced nor a suggested schedule and no
  // explicit tree is available...
  if (!HasSchedule()) {
    Warn(log::Verbosity::medium, "cannot apply mind trick (no schedule!)");
    return sch;
  }

  const std::string &target = scop_info_.user_config_.GetTarget();
  if (target == TARGET_CUDA) {
    if (explicit_tree_ && !suggested_schedule_ && !influenced_schedule_) {
      gpu_info_ = ExtractGpuConfig(explicit_tree_, gpu_info_);
      if (static_cast<log::Verbosity>(verbosity_) >= log::Verbosity::medium) {
        Info(name_ + ": extracted blocks: " + GetGpuBlocks());
        Info(name_ + ": extracted threads: " + GetGpuThreads());
      }
    }
  }

  // At this point, we are sure we have a schedule
  const isl::schedule &result = GetSchedule();
  return result;
}

SchedulingMindTrick::operator bool() const { return correctly_parsed_; }

bool SchedulingMindTrick::HasSchedule(void) const {
  return explicit_tree_ || suggested_schedule_ || influenced_schedule_;
}

////////////////////////////////////////////////////////////////////////////////
// I/O
////////////////////////////////////////////////////////////////////////////////

std::ostream &SchedulingMindTrick::Output(std::ostream &stream) const {
  stream << str();
  return stream;
}

std::istream &SchedulingMindTrick::Input(std::istream &stream) { return Parse(stream); }

std::ostream &operator<<(std::ostream &stream, const SchedulingMindTrick &mind_trick) {
  return mind_trick.Output(stream);
}

std::istream &operator>>(std::istream &stream, SchedulingMindTrick &mind_trick) { return mind_trick.Input(stream); }

static inline std::string indent(std::size_t level) {
  const std::size_t indent_spaces = 2;
  const std::string result(level * indent_spaces, ' ');
  return result;
}

static inline std::string escape(const char *input, char c) {
  std::stringstream stream;
  for (const char *p = input; *p != '\0'; p++) {
    if (*p == c) {
      stream << '\\' << c;
    } else {
      stream << *p;
    }
  }
  return stream.str();
}

static inline std::string escape(const std::string &input, char c) {
  return escape(input.c_str(), c);
}

static inline std::string quote(const std::string &input) { return "\"" + input + "\""; }

static inline std::string key_name(const std::string &key) { return quote(key) + ": "; }

std::string SchedulingMindTrick::str(void) const {
  std::stringstream stream;

  const std::string &sep = ",";

  stream << "{" << std::endl;
  // Basic metadata
  stream << indent(1) << key_name("name") << quote(name_) << sep << std::endl;
  if (domain_) {
    stream << indent(1) << key_name("domain");
    stream << "\"" << domain_ << "\"" << sep << std::endl;
  }

  // Target info
  if (target_ != "") {
    stream << indent(1) << key_name("target") << quote(target_) << sep << std::endl;
  }
  if (target_ == "" || target_ == TARGET_CUDA) {
    stream << indent(1) << key_name("gpu") << "{" << std::endl;
    stream << indent(2) << key_name("blocks") << quote(GetGpuBlocks()) << sep << std::endl;
    stream << indent(2) << key_name("threads") << quote(GetGpuThreads()) << sep << std::endl;
    if (!gpu_info_.compiler_flags_.empty()) {
      stream << indent(2) << key_name("compiler flags") << "[" << std::endl;
      for (const std::string &flag : gpu_info_.compiler_flags_) {
        stream << indent(3) << quote(flag) << sep << std::endl;
      }
      stream << indent(2) << "]" << sep << std::endl;
    }
    stream << indent(1) << "}" << sep << std::endl;
  }

  // Disabled passes
  if (!disabled_passes_.empty()) {
    stream << indent(1) << key_name("disable") << "[" << std::endl;
    for (const std::string &pass : disabled_passes_) {
      stream << indent(2) << quote(pass) << sep << std::endl;
    }
    stream << indent(1) << "]" << sep << std::endl;
  }

  if (suggested_schedule_string_ != "") {
    stream << indent(1) << key_name("schedule") << quote(suggested_schedule_string_) << sep << std::endl;
  }
  if (!suggested_schedule_vector_.empty()) {
    stream << indent(1) << key_name("schedule") << "[" << std::endl;
    for (auto dim : suggested_schedule_vector_) {
      stream << indent(2) << quote(dim) << sep << std::endl;
    }
    stream << indent(1) << "]" << sep << std::endl;
  }
  if (suggested_schedule_) {
    stream << indent(1) << key_name("suggested schedule (built)");
    stream << "\"" << suggested_schedule_ << "\"" << sep << std::endl;
    stream << indent(1) << key_name("suggested schedule (loop nest)") << "\"" << std::endl;
    stream << to_c_code_string(suggested_schedule_);
    stream << indent(1) << "\"" << sep << std::endl;
  }

  if (influenced_schedule_) {
    stream << indent(1) << key_name("influenced schedule (built)");
    stream << "\"" << influenced_schedule_ << "\"" << sep << std::endl;
    stream << indent(1) << key_name("  influenced schedule (loop nest)") << "\"" << std::endl;
    stream << to_c_code_string(influenced_schedule_);
    stream << indent(1) << "\"" << sep << std::endl;
  }

  if (explicit_tree_) {
    stream << indent(1) << key_name("tree") << "\"";
    stream << explicit_tree_ << "\"" << sep << std::endl;
    stream << "  explicit tree (loop nest): \n";
    stream << indent(1) << key_name("tree (loop nest)") << "\"" << std::endl;
    stream << to_c_code_string(explicit_tree_) << "\n";
    stream << indent(1) << "\"" << sep << std::endl;
  }

  if (pattern_ != "") {
    stream << indent(1) << key_name("pattern") << quote(pattern_) << sep << std::endl;
  }
  stream << indent(1) << key_name("check schedule") << (check_schedule_ ? "true" : "false") << sep << std::endl;

  stream << "}";

  return stream.str();
}

std::string to_string(const SchedulingMindTrick &mind_trick) { return mind_trick.str(); }

std::string SchedulingMindTrick::TemplateString(ScopInfo &scop_info, const isl::schedule &schedule,
                                                MindTrickType type) {
  picojson::object contents;

  // Name and operator fields
  const std::string &name = scop_info.user_config_.GetKernelName();
  contents["name"] = picojson::value("template contents for " + name);
  contents["operator"] = picojson::value(name);

  const isl::union_set &domain = schedule.domain();
  if (domain.isa_set()) {
    std::stringstream stream;
    stream << domain;
    contents["domain"] = picojson::value(stream.str());
  } else {
    const isl::set_list &list = domain.get_set_list();
    const unsigned size = list.size();

    picojson::array domain_json;
    for (unsigned i = 0; i < size; ++i) {
      std::stringstream stream;
      stream << list.at(i);
      const std::string &component = stream.str();
      const picojson::value &current = picojson::value(component);
      domain_json.push_back(current);
    }
    contents["domain"] = picojson::value(domain_json);
  }

  const std::string &sch_str = schedule.to_str();
  const std::string &pattern_string = escape(sch_str, '"');
  contents["pattern"] = picojson::value(sch_str);

  std::string auto_constraints = "";
  std::string auto_attrs = "";
  if (type == MindTrickType::autogen) {
    std::tie(auto_constraints, auto_attrs) = AutoGenSoftConstraints(scop_info, schedule);
  }
  if (auto_constraints != "") {
    picojson::value constraints;
    const std::string &error = picojson::parse(constraints, auto_constraints);
    if (error.empty()) {
      contents["soft constraints"] = constraints;
    } else {
      LOG(WARNING) << "auto_tricks json error: " << error;
    }
  }
  if (auto_attrs != "") {
    picojson::value attrs;
    const std::string &error = picojson::parse(attrs, auto_attrs);
    if (error.empty()) {
      contents["attrs"] = attrs;
    } else {
      LOG(WARNING) << "auto_attrs json error: " << error;
    }
  }
  const std::string &target = scop_info.user_config_.GetTarget();
  if (target == TARGET_CCE) {
    picojson::array disable_pass;
    disable_pass.push_back(picojson::value("GroupStatements"));
    disable_pass.push_back(picojson::value("UnGroupStatements"));
    contents["disable"] = picojson::value(disable_pass);
  }

  const picojson::value &trick = picojson::value(contents);
  const std::string &result = trick.serialize();
  log::Info(log::Verbosity::high, "json template:\n" + result);

  return result;
}

std::tuple<std::string, std::string> SchedulingMindTrick::AutoGenSoftConstraints(ScopInfo &scop_info,
                                                                                 const isl::schedule &sch) {
  const std::string &target = scop_info.user_config_.GetTarget();
  if (target == TARGET_CUDA) {
    return AutoGenGPUSoftConstraints(scop_info, sch);
  } else if (target == TARGET_CCE) {
    // AutoGenAscend910SoftConstraints in scheduling_mind_trick_ascend.cc
    return AutoGenAscend910SoftConstraints(scop_info, sch);
  }
  log::Warn("This case never happens");
  return std::make_tuple("", "");
}

std::tuple<std::string, std::string> SchedulingMindTrick::AutoGenGPUSoftConstraints(ScopInfo &scop_info,
                                                                                    const isl::schedule &sch) {
  const DimensionAnalysis &analysis = DimensionAnalysis(scop_info, sch);

  std::string constraints{""};
  const isl::set_list &statements = sch.get_domain().get_set_list();
  for (unsigned i = 0; i < statements.size(); ++i) {
    const isl::set &statement = statements.get_at(i);
    const std::string &statement_name = statement.get_tuple_name();
    const DimensionsDecision &decision = analysis.SelectDimensions(statement_name);
    if (!decision.dimensions_.empty()) {
      // Add ',\n' only if the current constraint has a predecessor
      // The final \n (without the comma!) will be added when the result string is composed
      if (constraints != "") {
        constraints += ",\n";
      }
      constraints += decision.SoftConstraints(scop_info, statement, i + 1);
    }
  }

  std::string result{""};
  if (constraints != "") {
    result += "[\n";
    // Add the final \n for legibility
    result += indent(1) + constraints + "\n";
    result += "]\n";
  }

  return std::make_tuple(result, "");
}

////////////////////////////////////////////////////////////////////////////////
// MindTrick metadata
////////////////////////////////////////////////////////////////////////////////

void SchedulingMindTrick::SetName(const std::string &name) { name_ = name; }

const std::string &SchedulingMindTrick::GetName(void) const { return name_; }

const std::string &SchedulingMindTrick::GetTarget(void) const { return target_; }

///////////////////////////////////////////////////////////////////////////
// Misc. attributes
///////////////////////////////////////////////////////////////////////////

bool SchedulingMindTrick::NeedsScheduleCheck(void) const {
  if (type_ == MindTrickType::manual && (explicit_tree_ || suggested_schedule_)) {
    return check_schedule_;
  }

  // "influenced" schedules should not need checks
  return false;
}

const air::Map<std::string, air::NodeRef> SchedulingMindTrick::GetAttrs(void) const { return attrs_; }

////////////////////////////////////////////////////////////////////////////////
// GPU Mapping
////////////////////////////////////////////////////////////////////////////////

struct extracted_config {
  std::vector<int> blocks;
  std::vector<int> threads;
};

static inline isl_bool isl_schedule_node_context_extract_gpu_config(__isl_keep isl_schedule_node *const node,
                                                                    struct extracted_config *const config) {
  isl::set context = isl::manage(isl_schedule_node_context_get_context(node));
  const isl_size set_dims = isl_set_dim(context, isl_dim_set);
  const isl_size param_dims = isl_set_dim(context, isl_dim_param);
  if (set_dims > 0 || param_dims <= 0) {
    log::Warn("Can not extract blocks/threads from this schedule node context");
    return isl_bool_true;
  }

  context = isl_set_move_dims(context, isl_dim_set, 0, isl_dim_param, 0, param_dims);

  const isl::set &lexmax = context.lexmax();
  const std::vector<std::string> block_names = {"b0", "b1", "b2"};
  const std::vector<std::string> thread_names = {"t0", "t1", "t2"};
  config->blocks = isl_set_lexmax_extract_upper_bounds(lexmax, block_names);
  config->threads = isl_set_lexmax_extract_upper_bounds(lexmax, thread_names);

  return isl_bool_true;
}

static inline isl_bool isl_extract_gpu_config(__isl_keep isl_schedule_node *const node, void *const user) {
  // This works under the follow assumptions:
  // - all blocks/threads are introduced in the same context
  // - block names are b0, b1, b2
  // - thread names are t0, t1, t2
  // - other iterators or parameters are not named b0, b1, b2, t0, t1 or t2
  //
  // In cases where multiple contexts introduce blocks/threads, we will guess
  // from the outermost context and the inner contexts will be ignored.
  // - {b,t}(0, 1, 2) -> x, y, z
  //
  // Block/thread config and mapping detection are separate steps because
  // we may have an input schedule tree where tiling has been applied
  // (and block/thread config has been decided) without the actual mapping.
  // We may sometimes want to bypass TileOuterBand but let MappingOuterBand
  // do its job

  struct extracted_config *config = (struct extracted_config *)user;
  const isl_schedule_node_type type = isl_schedule_node_get_type(node);
  if (type == isl_schedule_node_context && config->blocks.size() == 0 && config->threads.size() == 0) {
    return isl_schedule_node_context_extract_gpu_config(node, config);
  }

  return isl_bool_true;
}

GpuConfig SchedulingMindTrick::ExtractGpuConfig(const isl::schedule &schedule, const GpuConfig &info) {
  // Note: this method should not be triggered in normal use!
  // It requires undocumented features that are implemented for debugging purposes.

  // At least one of the block or the thread mapping must be missing
  if (!info.block_sizes_.empty() && !info.thread_sizes_.empty()) {
    return info;
  }

  if (!explicit_tree_) {
    return info;
  }

  const std::string target = scop_info_.user_config_.GetTarget();
  if (target != TARGET_CUDA) {
    Warn(name_ + ": extracting GPU config for target \'" + target + "\'?!");
  }

  // See if blocks/threads information is already available in the schedule
  // tree (usually, an explicit schedule tree).
  // Implementation note: no easy method exposed in the C++ wrapper, we have to use isl pointers here.
  struct extracted_config config;
  isl_schedule_foreach_schedule_node_top_down(schedule.get(), isl_extract_gpu_config, (void *)&config);

  GpuConfig result = info;
  if (result.block_sizes_.empty()) {
    result.block_sizes_ = config.blocks;
  }
  if (result.thread_sizes_.empty()) {
    result.thread_sizes_ = config.threads;
  }

  return result;
}

std::string SchedulingMindTrick::GetGpuBlocks(void) const {
  // Implementation Note:
  // (if needed) we reverse a copy of the vector because we don't know how many
  // times this method will be called!
  std::string string = "";
  const std::size_t dimensions = gpu_info_.block_sizes_.size();
  if (dimensions > 0) {
    std::vector<int> sizes = gpu_info_.block_sizes_;
    if (!explicit_tree_) {
      std::reverse(sizes.begin(), sizes.end());
    }
    string.append(std::to_string(sizes[0]));
    for (std::size_t i = 1; i < dimensions; ++i) {
      string.append(" ");
      string.append(std::to_string(sizes[i]));
    }
  }

  return string;
}

std::string SchedulingMindTrick::GetGpuThreads(void) const {
  // Implementation Note:
  // (if needed) we reverse a copy of the vector because we don't know how many
  // times this method will be called!
  std::vector<int> sizes = gpu_info_.thread_sizes_;
  if (!explicit_tree_) {
    // If we let MappingOuterBand do its job, it will map threads from bottom
    // to top as opposed to humans who would specify sizes from top to bottom.
    std::reverse(sizes.begin(), sizes.end());
  }

  std::string string = "";
  const std::size_t dimensions = sizes.size();
  if (dimensions > 0) {
    string.append(std::to_string(sizes[0]));
    for (std::size_t i = 1; i < dimensions; ++i) {
      string.append(" ");
      string.append(std::to_string(sizes[i]));
    }
  }

  return string;
}

const std::vector<std::string> &SchedulingMindTrick::GetGpuCompilerFlags(void) const {
  return gpu_info_.compiler_flags_;
}

bool SchedulingMindTrick::HasGpuSwizzleDim(void) const { return gpu_info_.has_swizzle_dim_; }

////////////////////////////////////////////////////////////////////////////////
// Disabling
////////////////////////////////////////////////////////////////////////////////

const std::set<std::string> &SchedulingMindTrick::GetDisabledPasses(void) const { return disabled_passes_; }

////////////////////////////////////////////////////////////////////////////////
// Schedule manipulation
////////////////////////////////////////////////////////////////////////////////

isl::schedule SchedulingMindTrick::GetSchedule(void) const {
// We assume the caller has first checked the result of HasSchedule()!

// Return `target` and log `message` if the target schedule is available
#define select_schedule(target, message)             \
  do {                                               \
    if (target) {                                    \
      Info(log::Verbosity::low, text_green message); \
      return target;                                 \
    }                                                \
  } while (0)

  // Priority ordered (although tricks should contain only one kind of schedule):
  select_schedule(explicit_tree_, "using explicit tree");
  select_schedule(suggested_schedule_, "using suggested schedule");
  select_schedule(influenced_schedule_, "using influenced schedule");

#undef select_schedule

  // We should not reach here!
  Error("GetSchedule() was probably called without checking the result of HasSchedule()!");
  isl::ctx ctx = scop_info_.GetCtx();
  const isl::schedule &dummy = isl::schedule(ctx, "domain: \"{ S[i]: 0 <= i < 1024 }\"");
  return dummy;
}

////////////////////////////////////////////////////////////////////////////////
// Loggging
////////////////////////////////////////////////////////////////////////////////

std::string SchedulingMindTrick::LogPrefixText(const bool prefix) const {
  if (!prefix) {
    return "";
  }

  const std::string &kernel_name = scop_info_.user_config_.GetKernelName();
  const std::string &trick_name = name_;
  const std::string &prefix_text = "'" + kernel_name + "': '" + trick_name + "': ";
  return prefix_text;
}

void SchedulingMindTrick::DebugSchedule(const isl::schedule &schedule, const std::string &message,
                                        const log::Verbosity level) const {
  // We explicitely check the verbosity level because some string conversions below are costly
  if (log::GetVerbosityLevel() < level) {
    return;
  }

  const std::string &block_string = to_block_string(schedule);
  const std::string &loop_nest = to_c_code_string(schedule);
  const std::string &schedule_prefix = message != "" ? message + " schedule:\n" : "Schedule:\n";
  const std::string &loop_nest_prefix = "Loop nest:\n";
  Info(schedule_prefix + block_string);
  Info(loop_nest_prefix + loop_nest);
}

///////////////////////////////////////////////////////////////////////////
// Mind Trick Type
///////////////////////////////////////////////////////////////////////////

MindTrickType SchedulingMindTrick::GetType(void) const { return type_; }

void SchedulingMindTrick::SetType(MindTrickType type) { type_ = type; }

// clang-format off
#define define_scheduling_mind_trick_log_wrappers(func)                                                                  \
  void SchedulingMindTrick::func(const std::string &message, const bool prefix) const {                                  \
    const std::string &prefix_text = LogPrefixText(prefix);                                                              \
    log::func(prefix_text + message);                                                                                    \
  }                                                                                                                      \
                                                                                                                         \
  void SchedulingMindTrick::func(const std::stringstream &stream, const bool prefix) const {                             \
    const std::string &message = stream.str();                                                                           \
    func(message, prefix);                                                                                               \
  }                                                                                                                      \
                                                                                                                         \
  void SchedulingMindTrick::func(const int level, const std::string &message, const bool prefix) const {                 \
    const std::string &prefix_text = LogPrefixText(prefix);                                                              \
    log::func(level, prefix_text + message);                                                                             \
  }                                                                                                                      \
  void SchedulingMindTrick::func(const int level, const std::stringstream &stream, const bool prefix) const {            \
    const std::string &message = stream.str();                                                                           \
    func(level, message, prefix);                                                                                        \
  }                                                                                                                      \
  void SchedulingMindTrick::func(const log::Verbosity level, const std::string &message, const bool prefix) const {      \
    const std::string &prefix_text = LogPrefixText(prefix);                                                              \
    log::func(level, prefix_text + message);                                                                             \
  }                                                                                                                      \
  void SchedulingMindTrick::func(const log::Verbosity level, const std::stringstream &stream, const bool prefix) const { \
    const std::string &message = stream.str();                                                                           \
    func(level, message, prefix);                                                                                        \
  }

  define_scheduling_mind_trick_log_wrappers(Info)
  define_scheduling_mind_trick_log_wrappers(Warn)
  define_scheduling_mind_trick_log_wrappers(Error)
#undef define_scheduling_mind_trick_log_wrappers
// clang-format on

}  // namespace poly
}  // namespace ir
}  // namespace akg
