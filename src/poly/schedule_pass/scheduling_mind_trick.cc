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
#include <string.h>

// TVM
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/dtype.h>

#include "poly/isl_util.h"
#include "poly/log_util.h"

namespace akg {
namespace ir {
namespace poly {

////////////////////////////////////////////////////////////////////////////////
// MindTrickType functions
////////////////////////////////////////////////////////////////////////////////

std::string to_string(MindTrickType t) {
  std::string result = "";
  switch (t) {
    case MindTrickType::manual:
      result = "manual";
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
  }

  return MindTrickType::none;
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

SchedulingMindTrick::~SchedulingMindTrick() {
  if (explicit_tree_) {
    explicit_tree_ = isl_schedule_free(explicit_tree_);
  }
  if (domain_) {
    domain_ = isl_union_set_free(domain_);
  }
  if (suggested_schedule_) {
    suggested_schedule_ = isl_schedule_free(suggested_schedule_);
  }
  if (influence_list_) {
    isl_influence_list_free(influence_list_);
    influence_list_ = nullptr;
  }
  if (influence_equal_list_) {
    isl_influence_equal_list_free(influence_equal_list_);
    influence_equal_list_ = nullptr;
  }
  if (influenced_schedule_) {
    influenced_schedule_ = isl_schedule_free(influenced_schedule_);
  }
}

void SchedulingMindTrick::Load(const std::string &filename) {
  // Set filename as the mind_trick's default name.
  // It may be overriden when parsing the file contents.
  name_ = filename;

  std::ifstream stream(filename);
  if (stream.is_open()) {
    Parse(stream);
  }
}

std::istream &SchedulingMindTrick::Parse(std::istream &stream) {
  picojson::value json;

  const std::string error = picojson::parse(json, stream);
  if (!error.empty()) {
    Error(error);
    correctly_parsed_ = false;
    return stream;
  }

  // Parse the json representation
  correctly_parsed_ = true;
  Parse(json);

  return stream;
}

void SchedulingMindTrick::Parse(const std::string &serialized) {
  // Parse the serialized string representation
  picojson::value json;
  const std::string error = picojson::parse(json, serialized);

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

std::vector<std::string> SchedulingMindTrick::split_string(std::string str, std::string delim) {
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
  ParseSoftConstraints(maybe(root, "soft constraints"));
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
  isl_ctx *const ctx = scop_info_.GetCtx().get();

  isl_union_set *set = 0;
  if (node.is<std::string>()) {
    const std::string &str = node.get<std::string>();
    const char *const c_str = str.c_str();
    set = isl_union_set_read_from_str(ctx, c_str);
  } else if (node.is<picojson::array>()) {
    std::vector<std::string> subdomains = to_string_vector(node);
    if (!subdomains.empty()) {
      const std::size_t count = subdomains.size();
      const char *const first_string = subdomains[0].c_str();
      set = isl_union_set_read_from_str(ctx, first_string);
      for (std::size_t i = 1; i < count; ++i) {
        const char *const string = subdomains[i].c_str();
        isl_union_set *const current = isl_union_set_read_from_str(ctx, string);
        set = isl_union_set_union(set, current);
      }
    }
  }

  if (set) {
    domain_ = set;
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

  const picojson::value &flags = maybe(node, "compiler flags");
  gpu_info_.compiler_flags_ = to_string_vector(flags);
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

  isl_ctx *const ctx = scop_info_.GetCtx().get();
  isl_schedule *const schedule = isl_schedule_read_from_str(ctx, str.c_str());
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

void SchedulingMindTrick::IslInfluenceToggle(bool toggle) {
  isl_ctx *const ctx = scop_info_.ctx_.get();
  if (toggle)
    akg_isl_influence_enable(ctx);
  else
    akg_isl_influence_disable(ctx);
}

void SchedulingMindTrick::CollectSoftConstraintsData(std::string stmt_name, unsigned int sched_dim, int coeff_dim,
                                                     isl_influence_coeff_type coeff_type, std::string coeff_vec_i) {
  // Cases of x
  if (std::regex_match(coeff_vec_i, std::regex("(-)?[0-9]+"))) {
    singles_.push_back(std::make_tuple(stmt_name, sched_dim, coeff_dim, coeff_type, std::stoi(coeff_vec_i)));
  } else if (std::regex_match(coeff_vec_i, std::regex("\\?[0-9]+"))) {  // Case of ?x
    std::string key = coeff_vec_i;

    linked_[key].push_back(std::make_tuple(stmt_name, sched_dim, coeff_dim, coeff_type, 0));
  } else {  // Case of ? or any other thing
    Warn(log::Verbosity::medium, name_ + ": ignoring free or ill-formed coefficient " + coeff_vec_i);
  }
}

void SchedulingMindTrick::ParseSoftConstraints(const picojson::value &node) {
  if (!node.is<picojson::array>()) {
    return;
  }

  unsigned int nb_stmt = node.get<picojson::array>().size();

  for (unsigned int stmt_num = 0; stmt_num < nb_stmt; ++stmt_num) {
    auto stmt = node.get<picojson::array>()[stmt_num];

    const picojson::value &statement = maybe(stmt, "statement");
    const picojson::value &meta = maybe(stmt, "meta");
    const picojson::value &coefficients = maybe(stmt, "coefficients");

    if (!statement.is<std::string>() || !meta.is<picojson::array>() || !coefficients.is<picojson::array>()) {
      return;
    }

    std::vector<std::string> coefficients_ = to_string_vector(coefficients);

    std::vector<int> meta_ = to_int_vector(meta);
    unsigned int nb_vars = meta_[0];
    unsigned int nb_params = meta_[1];
    unsigned int nb_dims = coefficients_.size();
    std::string stmt_name = statement.get<std::string>();

    for (unsigned int dim = 0; dim < nb_dims; ++dim) {
      // Remove whitespaces, '[', ']', '(' and ')'
      std::string str_ = std::regex_replace(coefficients_[dim], std::regex("\\[|\\]|\\(|\\)|[\\s]+"), "");

      if (str_.find("%") != std::string::npos && str_.find("/") != std::string::npos) {
        Error(name_ + ": cannot have a / and a % on the same scheduling dimension");
        return;
      }

      if (str_.find("%") != std::string::npos) {
        auto tmp = split_string(str_, "%");
        int mod_value = std::stoi(tmp[1]);
        modulos_.push_back(std::make_tuple(stmt_name, dim, mod_value));
        str_ = tmp[0];

        Info(log::Verbosity::high, text_blue + name_ + ": collected modulo " + tmp[1] + " at " + stmt_name +
                                     " sched dim " + std::to_string(dim));
      }

      if (str_.find("/") != std::string::npos) {
        auto tmp = split_string(str_, "/");
        int divisor = std::stoi(tmp[1]);
        divisions_.push_back(std::make_tuple(stmt_name, dim, divisor));
        str_ = tmp[0];

        Info(log::Verbosity::high, text_blue + name_ + ": collected divisor " + tmp[1] + " at " + stmt_name +
                                     " sched dim " + std::to_string(dim));
      }

      std::vector<std::string> coeff_vec = split_string(str_, ",");

      if (coeff_vec.size() != (nb_vars + nb_params + 1)) {
        Error(name_ + ": at " + stmt_name + ",  nb vars + nb params + 1 is different from size of coefficient vector");
        return;
      }

      unsigned int i, j;
      for (i = 0; i < nb_vars; ++i) {
        CollectSoftConstraintsData(stmt_name, dim, i, isl_var, coeff_vec[i]);
      }

      for (j = i; j < i + nb_params; ++j) {
        CollectSoftConstraintsData(stmt_name, dim, j - i, isl_param, coeff_vec[j]);
      }

      CollectSoftConstraintsData(stmt_name, dim, -1, isl_cst, coeff_vec[j]);
    }
  }

  return;
}

void SchedulingMindTrick::BuildInfluenceList(std::vector<single_data> singles) {
  isl_ctx *const ctx = scop_info_.GetCtx().get();
  influence_list_ = isl_calloc_type(ctx, struct isl_influence_list);

  if (influence_list_ == NULL) {
    return;
  }

  influence_list_->data = isl_calloc_array(ctx, struct isl_influence, singles.size());
  if (influence_list_->data == NULL) {
    return;
  }

  influence_list_->size = singles.size();
  influence_list_->mem = singles.size();

  for (unsigned i = 0; i < singles.size(); ++i) {
    isl_influence *pinf = &influence_list_->data[i];
    single_data data = singles[i];

    pinf->statement_name = strdup(std::get<0>(data).c_str());
    pinf->sched_dim = std::get<1>(data);
    pinf->coef_dim = std::get<2>(data);
    pinf->type = std::get<3>(data);
    pinf->val = std::get<4>(data);

    parse_soft_constraints_log_str_ +=
      "S [" + std::string(pinf->statement_name) + "] : dim=" + std::to_string(pinf->sched_dim);
    parse_soft_constraints_log_str_ +=
      ", coeff val " + std::to_string(pinf->val) + ", coeff type " + std::to_string(pinf->type);
    if (pinf->coef_dim >= 0) {
      parse_soft_constraints_log_str_ += ", coeff dim " + std::to_string(pinf->coef_dim) + "\n";
    } else {
      parse_soft_constraints_log_str_ += " (no coeff dim because coeff type is isl_cst)\n";
    }
  }
}

void SchedulingMindTrick::BuildInfluenceEqualList(std::map<std::string, std::vector<single_data>> linked) {
  isl_ctx *const ctx = scop_info_.GetCtx().get();

  struct isl_influence_equal_list *const list = isl_calloc_type(ctx, struct isl_influence_equal_list);
  if (!list) {
    return;
  }

  std::size_t list_size = 0;
  for (auto l : linked) {
    list_size += l.second.size() - 1;
  }

  struct isl_influence_equal *const data = isl_calloc_array(ctx, struct isl_influence_equal, list_size);
  if (!data) {
    free(list);
    return;
  }

  influence_equal_list_ = list;
  list->mem = list_size;
  list->size = list_size;
  list->data = data;

  int counter = 0;
  for (auto l : linked) {
    const std::vector<single_data> &links = l.second;
    const unsigned count = links.size() - 1;
    for (unsigned i = 0; i < count; ++i) {
      const single_data this_ = links[i];
      const single_data next_ = links[i + 1];
      char *const statement_1 = strdup(std::get<0>(this_).c_str());
      char *const statement_2 = strdup(std::get<0>(next_).c_str());
      const int sched_dim1 = std::get<1>(this_);
      const int sched_dim2 = std::get<1>(next_);
      const int coef_dim1 = std::get<2>(this_);
      const int coef_dim2 = std::get<2>(next_);
      const isl_influence_coeff_type type = std::get<3>(this_);

      isl_influence_equal *const current = data + counter;
      current->statement1 = statement_1;
      current->statement2 = statement_2;
      current->sched_dim1 = sched_dim1;
      current->coef_dim1 = coef_dim1;
      current->sched_dim2 = sched_dim2;
      current->coef_dim2 = coef_dim2;
      current->type = type;

      parse_soft_constraints_log_str_ += "L [" + std::string(statement_1) + "] : dim1 " + std::to_string(sched_dim1);
      parse_soft_constraints_log_str_ += ", ceoff type " + std::to_string(type);
      if (coef_dim1 >= 0 && coef_dim2 >= 0) {
        parse_soft_constraints_log_str_ += ", coeff dim1 " + std::to_string(coef_dim1);
      } else {
        parse_soft_constraints_log_str_ += " (no coeff dim because coeff type is isl_cst)";
      }

      parse_soft_constraints_log_str_ += "; [" + std::string(statement_2) + "] : dim2 " + std::to_string(sched_dim2);
      parse_soft_constraints_log_str_ += ", coeff type " + std::to_string(type);
      if (coef_dim1 >= 0 && coef_dim2 >= 0) {
        parse_soft_constraints_log_str_ += ", coeff dim2 " + std::to_string(coef_dim2) + "\n";
      } else {
        parse_soft_constraints_log_str_ += " (no coeff dim because coeff type is isl_cst)\n";
      }

      counter++;
    }
  }
}

void SchedulingMindTrick::BuildSoftConstraints(void) {
  BuildInfluenceList(singles_);
  BuildInfluenceEqualList(linked_);

  Info(log::Verbosity::high,
       text_blue "Memo: S=Single constraint, L=Linked constraint, type 0=isl_cst, type 1=isl_param, type 2=isl_var");
  Info(log::Verbosity::high, text_blue "Constraints\n" + parse_soft_constraints_log_str_);
}

__isl_give isl_schedule *SchedulingMindTrick::AdjustSchedule(__isl_take isl_schedule *schedule,
                                                             const std::vector<div_mod_data> &modulos,
                                                             const std::vector<div_mod_data> &divisions) {
  // We assume the root's child is a schedule_node_band that contains all target dimensions.
  isl_schedule_node *const root = isl_schedule_get_root(schedule);
  isl_schedule_node *band = isl_schedule_node_get_child(root, 0);

  for (auto adjustment : modulos) {
    const std::string &statement = std::get<0>(adjustment);
    const char *const name = statement.c_str();
    const int dimension = std::get<1>(adjustment);
    const int value = std::get<2>(adjustment);
    band = isl_schedule_node_band_fine_mod(band, name, dimension, value);
  }
  for (auto adjustment : divisions) {
    const std::string &statement = std::get<0>(adjustment);
    const char *const name = statement.c_str();
    const int dimension = std::get<1>(adjustment);
    const int value = std::get<2>(adjustment);
    band = isl_schedule_node_band_fine_scale_down(band, name, dimension, value);
  }

  isl_schedule *const result = isl_schedule_node_get_schedule(band);

  isl_schedule_free(schedule);
  isl_schedule_node_free(band);
  isl_schedule_node_free(root);

  return result;
}

void SchedulingMindTrick::BuildInfluencedSchedule(void) {
  if (!influence_list_ || !influence_equal_list_) return;

  IslInfluenceToggle(true);
  isl_schedule_constraints *const constraints = isl_schedule_constraints_copy(pass_info_.constraints_.get());
  isl_schedule *result =
    akg_isl_schedule_constraints_compute_schedule_influence(constraints, influence_list_, influence_equal_list_);
  IslInfluenceToggle(false);

  if (!result) {
    Warn("Could not influence schedule!");
    return;
  }

  Info(log::Verbosity::low, text_bright_blue "Influenced schedule:\n" + to_block_string(result));
  Info(log::Verbosity::medium, text_cyan "Loop nest:\n" + to_c_code_string(result));

  result = AdjustSchedule(result, modulos_, divisions_);

  Info(log::Verbosity::medium, text_bright_blue "Adjusted schedule:\n" + to_block_string(result));
  Info(log::Verbosity::high, text_cyan "Loop nest:\n" + to_c_code_string(result));

  const std::string &target = scop_info_.user_config_.GetTarget();
  if (target == TARGET_CUDA) {
    // For now, we use another GpuConfig, we do not wish to use the influenced schedule yet.
    result = PrepareMappingOuterBand(result, gpu_info_);
    if (type_ != MindTrickType::manual && (!gpu_info_.block_sizes_.empty() || !gpu_info_.thread_sizes_.empty())) {
      disabled_passes_.insert("GpuDmaAnalysis");
      disabled_passes_.insert("TileOuterBand");
    }

    Info(log::Verbosity::low,
         text_bright_blue "Prepared schedule (PrepareMappingOuterBand):\n" + to_block_string(result));
    Info(log::Verbosity::high, text_cyan "Loop nest:\n" + to_c_code_string(result));
  }

  influenced_schedule_ = result;
}

void SchedulingMindTrick::BuildSuggestedSchedule(void) {
  if (suggested_schedule_string_ == "" && suggested_schedule_vector_.empty()) {
    Info("will not build a suggested schedule (no suggestion available)");
    return;
  }
  if (!domain_) {
    Error("building a suggested schedule requires a domain");
    return;
  }

  const std::string &target = scop_info_.user_config_.GetTarget();

  // First compute the schedule suggestion
  isl_schedule *schedule = ComputeScheduleSuggestion();
  if (target == TARGET_CUDA) {
    // Let us edit the band we created.
    schedule = PrepareMappingOuterBand(schedule, gpu_info_);
  }
  suggested_schedule_ = schedule;
}

__isl_give isl_schedule *SchedulingMindTrick::ComputeScheduleSuggestion(void) {
  // For "suggestions", "computing" the schedule suggestion amounts to converting the input string...

  isl_ctx *const ctx = scop_info_.GetCtx().get();

  isl_multi_union_pw_aff *aff = 0;
  if (suggested_schedule_string_ != "") {
    const char *suggestion = suggested_schedule_string_.c_str();
    aff = isl_multi_union_pw_aff_read_from_str(ctx, suggestion);
  } else if (!suggested_schedule_vector_.empty()) {
    const int count = suggested_schedule_vector_.size();
    const char *const first_string = suggested_schedule_vector_[0].c_str();
    isl_union_pw_aff *const first_aff = isl_union_pw_aff_read_from_str(ctx, first_string);
    isl_union_pw_aff_list *list = isl_union_pw_aff_list_from_union_pw_aff(first_aff);
    for (isl_size i = 1; i < count; ++i) {
      const char *const current_string = suggested_schedule_vector_[i].c_str();
      isl_union_pw_aff *const current_aff = isl_union_pw_aff_read_from_str(ctx, current_string);
      list = isl_union_pw_aff_list_add(list, current_aff);
    }

    isl_space *const domain_space = isl_union_set_get_space(domain_);
    const isl_size params = isl_space_dim(domain_space, isl_dim_param);
    isl_space *space = isl_space_set_alloc(ctx, params, count);
    for (isl_size i = 0; i < params; ++i) {
      isl_id *const name = isl_space_get_dim_id(domain_space, isl_dim_param, i);
      space = isl_space_set_dim_id(space, isl_dim_param, i, name);
    }
    aff = isl_multi_union_pw_aff_from_union_pw_aff_list(space, list);

    isl_space_free(domain_space);
  }

  isl_union_set *const domain = isl_union_set_copy(domain_);
  isl_schedule *result = isl_schedule_from_domain(domain);

  if (aff) {
    result = isl_schedule_insert_partial_schedule(result, aff);
  }

  return result;
}

__isl_give isl_schedule *SchedulingMindTrick::PrepareMappingOuterBand(__isl_take isl_schedule *const schedule,
                                                                      GpuConfig &info) {
  const std::vector<int> &block_dimensions = info.block_dimensions_;
  const std::vector<int> &thread_dimensions = info.thread_dimensions_;
  const std::vector<int> &block_sizes = info.block_sizes_;
  const std::vector<int> &thread_sizes = info.thread_sizes_;

  // We cannot prepare the schedule for MappingOuterBand without any hint.
  if ((block_dimensions.size() == 0 && block_sizes.size() == 0) ||
      (thread_dimensions.size() == 0 && thread_sizes.size() == 0)) {
    return schedule;
  }

  // Check whether block dimensions start at 0.
  if (block_dimensions.size() > 0 && block_dimensions[0] != 0) {
    Error("we do not support (yet?) blocks.x != 0");
    return schedule;
  }

  if (block_dimensions.size() > 0 && block_sizes.size() > 0 && block_dimensions.size() != block_sizes.size()) {
    Warn("you should specify as many block dimensions as sizes");
  }
  if (thread_dimensions.size() > 0 && thread_sizes.size() > 0 && thread_dimensions.size() != thread_sizes.size()) {
    Warn("you should specify as many thread dimensions as sizes");
  }

  // Simple case:
  // - blocks/threads dimensions are consecutive
  // - blocks are mapped to the outermost dimensions
  // - (threads can be anywere as long as they are consecutive...)

  isl_schedule_node *const root = isl_schedule_get_root(schedule);
  isl_schedule_node *band = isl_schedule_node_get_child(root, 0);

  // Retrieve blocks/threads sizes from the schedule if not specified
  if (block_sizes.size() == 0 || thread_sizes.size() == 0) {
    isl_union_map *const map = isl_schedule_node_band_get_partial_schedule_union_map(band);
    isl_union_set *const applied = isl_union_set_apply(isl_union_set_copy(domain_), map);
    isl_union_set *const lexmax = isl_union_set_lexmax(applied);
    if (isl_union_set_isa_set(lexmax)) {
      isl_set *const values = isl_set_from_union_set(lexmax);

      if (info.block_sizes_.size() == 0) {
        info.block_sizes_ = extract_upper_bounds(values, info.block_dimensions_);
      }
      if (info.thread_sizes_.size() == 0) {
        info.thread_sizes_ = extract_upper_bounds(values, info.thread_dimensions_);
      }
      isl_set_free(values);
    } else {
      Warn("can not retrieve blocks/threads sizes");
      isl_union_set_free(lexmax);
    }
  }

  const isl_size size = isl_schedule_node_band_n_member(band);
  // MappingOuterBand looks in permutable bands.
  band = isl_schedule_node_band_set_permutable(band, 1);
  // Enforce coincidence for block and thread dimensions (in between dimensions will not matter)
  for (auto dimension : block_dimensions)
    if (dimension < size) band = isl_schedule_node_band_member_set_coincident(band, dimension, 1);
  for (auto dimension : thread_dimensions)
    if (dimension < size) band = isl_schedule_node_band_member_set_coincident(band, dimension, 1);
  // Explicitly remove coincidence for innermost dimensions (deeper than thread dimensions)
  int innermost_start = 0;
  if (!thread_dimensions.empty())
    innermost_start = 1 + *std::max_element(thread_dimensions.begin(), thread_dimensions.end());
  else if (!block_dimensions.empty())
    innermost_start = 1 + *std::max_element(block_dimensions.begin(), block_dimensions.end());
  for (int dimension = innermost_start; dimension < size; ++dimension)
    band = isl_schedule_node_band_member_set_coincident(band, dimension, 0);
  // Split blocks and threads into separate nodes (especially for reduce operators)
  const int outermost_thread = *std::min_element(thread_dimensions.begin(), thread_dimensions.end());
  band = isl_schedule_node_band_split(band, outermost_thread);
  isl_schedule_node *child = isl_schedule_node_get_child(band, 0);
  child = isl_schedule_node_band_set_permutable(child, 1);

  // Never forget that isl_schedule_node modifications happen on a copy of the schedule tree!
  // Hence we have to get a new isl_schedule from the isl_schedule_node!
  isl_schedule *const result = isl_schedule_node_get_schedule(band);
  // And do not forget to free the initial schedule...
  isl_schedule_free(schedule);

  isl_schedule_node_free(root);
  isl_schedule_node_free(band);
  isl_schedule_node_free(child);

  return result;
}

////////////////////////////////////////////////////////////////////////////////
// MindTrick use
////////////////////////////////////////////////////////////////////////////////

isl::schedule SchedulingMindTrick::Apply(const isl::schedule &sch) {
  // 1. Attempt to influence a schedule.
  // 2. Attempt to build a suggested schedule.
  // 3. See if there is an explicit tree.

  // Attempt to influence a schedule
  if (singles_.empty() && linked_.empty()) {
    Warn(name_ + ": no soft constraints");
  } else {
    Info(log::Verbosity::medium, ": Building soft constraints");
    BuildSoftConstraints();
    BuildInfluencedSchedule();
  }
  // Only attempt to build the full suggestion if the influence failed.
  if (!influenced_schedule_) {
    if (suggested_schedule_string_ == "" && !domain_) {
      Warn(name_ + ": cannot build suggested schedule");
    } else {
      BuildSuggestedSchedule();
    }
  }
  // Could neither build an influenced nor a suggested schedule and no
  // explicit tree is available...
  if (!HasSchedule()) {
    Warn(name_ + ": cannot apply mind trick (no schedule!)");
    return sch;
  }

  // At this point, we know we have a schedule
  GuessGpuConfig();
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
  const char *str = input.c_str();
  return escape(str, c);
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
    stream << "\"" << isl::manage_copy(domain_) << "\"" << sep << std::endl;
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
    stream << "\"" << isl::manage_copy(suggested_schedule_) << "\"" << sep << std::endl;
    stream << indent(1) << key_name("suggested schedule (loop nest)") << "\"" << std::endl;
    stream << to_c_code_string(suggested_schedule_);
    stream << indent(1) << "\"" << sep << std::endl;
  }

  if (influenced_schedule_) {
    stream << indent(1) << key_name("influenced schedule (built)");
    stream << "\"" << isl::manage_copy(influenced_schedule_) << "\"" << sep << std::endl;
    stream << indent(1) << key_name("  influenced schedule (loop nest)") << "\"" << std::endl;
    stream << to_c_code_string(influenced_schedule_);
    stream << indent(1) << "\"" << sep << std::endl;
  }

  if (explicit_tree_) {
    stream << indent(1) << key_name("tree") << "\"";
    stream << isl::manage_copy(explicit_tree_) << "\"" << sep << std::endl;
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

  isl_schedule *const sch = schedule.get();
  const isl::union_set &domain = isl::manage(isl_schedule_get_domain(sch));
  if (domain.isa_set()) {
    std::stringstream stream;
    stream << domain;
    contents["domain"] = picojson::value(stream.str());
  } else {
    const isl::set_list &list = domain.get_set_list();
    const unsigned size = list.size();

    picojson::array domain_json;
    for (unsigned i = 0; i < size - 1; ++i) {
      std::stringstream stream;
      stream << list.at(i);
      const std::string &component = stream.str();
      const picojson::value &current = picojson::value(component);
      domain_json.push_back(current);
    }
    contents["domain"] = picojson::value(domain_json);
  }

  char *const sch_str = isl_schedule_to_str(sch);
  const std::string &pattern_string = escape(sch_str, '"');
  contents["pattern"] = picojson::value(sch_str);
  free(sch_str);

  const picojson::value &trick = picojson::value(contents);
  const std::string &result = trick.serialize();
  log::Info(log::Verbosity::high, "json template:\n" + result);

  return result;
}

////////////////////////////////////////////////////////////////////////////////
// MindTrick metadata
////////////////////////////////////////////////////////////////////////////////

void SchedulingMindTrick::SetName(const std::string &name) { name_ = name; }

std::string SchedulingMindTrick::GetName(void) const { return name_; }

std::string SchedulingMindTrick::GetTarget(void) const { return target_; }

///////////////////////////////////////////////////////////////////////////
// Misc. attributes
///////////////////////////////////////////////////////////////////////////

bool SchedulingMindTrick::NeedsScheduleCheck(void) const { return check_schedule_; }
const air::Map<std::string, air::NodeRef> SchedulingMindTrick::GetAttrs(void) const { return attrs_; }

////////////////////////////////////////////////////////////////////////////////
// GPU Mapping
////////////////////////////////////////////////////////////////////////////////

struct guessed_config {
  std::vector<int> blocks;
  std::vector<int> threads;
};

static inline void isl_set_find_gpu_mapping_values(__isl_keep isl_set *const lexmax, int count,
                                                   const char *const *targets, std::vector<int> &values) {
  for (int i = 0; i < count; ++i) {
    const int position = isl_set_find_dim_by_name(lexmax, isl_dim_set, targets[i]);
    if (position >= 0) {
      isl_val *value = isl_set_plain_get_val_if_fixed(lexmax, isl_dim_set, position);
      // lexmax... bx <= value
      value = isl_val_add_ui(value, 1);
      char *string = isl_val_to_str(value);
      values.push_back(std::stoi(string));

      isl_val_free(value);
      free(string);
    }
  }
}

static inline isl_bool isl_schedule_node_context_guess_gpu_config(__isl_keep isl_schedule_node *const node,
                                                                  struct guessed_config *const config) {
  isl_set *context = isl_schedule_node_context_get_context(node);
  const isl_size set_dims = isl_set_dim(context, isl_dim_set);
  const isl_size param_dims = isl_set_dim(context, isl_dim_param);
  if (set_dims > 0 || param_dims <= 0) {
    LOG(WARNING) << "\033[33m"
                 << "Can not guess blocks/threads from the schedule tree."
                 << "\033[0m";
    isl_set_free(context);
    return isl_bool_true;
  }

  context = isl_set_move_dims(context, isl_dim_set, 0, isl_dim_param, 0, param_dims);
  isl_set *const lexmax = isl_set_lexmax(context);

  const char *const blocks[3] = {"b0", "b1", "b2"};
  const char *const threads[3] = {"t0", "t1", "t2"};
  isl_set_find_gpu_mapping_values(lexmax, sizeof blocks / sizeof blocks[0], blocks, config->blocks);
  isl_set_find_gpu_mapping_values(lexmax, sizeof threads / sizeof threads[0], threads, config->threads);

  isl_set_free(lexmax);
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

  struct guessed_config *config = (struct guessed_config *)user;
  const isl_schedule_node_type type = isl_schedule_node_get_type(node);
  if (type == isl_schedule_node_context && config->blocks.size() == 0 && config->threads.size() == 0) {
    return isl_schedule_node_context_guess_gpu_config(node, config);
  }

  return isl_bool_true;
}

void SchedulingMindTrick::GuessGpuConfig(void) {
  if (gpu_info_.block_sizes_.size() != 0 && gpu_info_.thread_sizes_.size() != 0) {
    return;
  }

  const std::string target = scop_info_.user_config_.GetTarget();
  if (target != TARGET_CUDA) {
    Warn(name_ + ": guessing GPU config for target \'" + target + "\'?!");
  }

  const int saved_verbosity_ = verbosity_;
  verbosity_ = 0;
  const isl::schedule &wrapped_schedule = GetSchedule();
  isl_schedule *const schedule = wrapped_schedule.get();
  verbosity_ = saved_verbosity_;

  // See if blocks/threads information is already available in the schedule
  // tree (usually, an explicit schedule tree).
  struct guessed_config config;
  isl_schedule_foreach_schedule_node_top_down(schedule, isl_extract_gpu_config, (void *)&config);

  if (gpu_info_.block_sizes_.size() == 0) {
    gpu_info_.block_sizes_ = config.blocks;
  }
  if (gpu_info_.thread_sizes_.size() == 0) {
    gpu_info_.thread_sizes_ = config.threads;
  }

  if (static_cast<log::Verbosity>(verbosity_) >= log::Verbosity::medium) {
    Info(name_ + ": guessed blocks: " + GetGpuBlocks());
    Info(name_ + ": guessed threads: " + GetGpuThreads());
  }
}

std::string SchedulingMindTrick::GetGpuBlocks(void) const {
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

std::vector<std::string> SchedulingMindTrick::GetGpuCompilerFlags(void) const { return gpu_info_.compiler_flags_; }

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
      return isl::manage_copy(target);               \
    }                                                \
  } while (0)

  // Priority ordered (although tricks should contain only one kind of schedule):
  select_schedule(explicit_tree_, "using explicit tree");
  select_schedule(suggested_schedule_, "using suggested schedule");
  select_schedule(influenced_schedule_, "using influenced schedule");

#undef select_schedule

  // We should not reach here!
  Error("GetSchedule() was probably called without checking the result of HasSchedule()!");
  isl_ctx *const ctx = scop_info_.GetCtx().get();
  isl_schedule *const dummy_schedule = isl_schedule_read_from_str(ctx, "domain: \"{ S[i]: 0 <= i < 1024 }\"");
  return isl::manage(dummy_schedule);
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
