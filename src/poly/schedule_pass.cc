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

#include "schedule_pass.h"

#include <climits>
#include <fstream>

namespace akg {
namespace ir {
namespace poly {

static constexpr const char *const kEnvStringMsDevPolyScheduler = "MS_DEV_POLY_SCHEDULER";
static constexpr const char *const kEnvStringMsDevPolyTOPSSolver = "MS_DEV_POLYTOPS_SOLVER";
static constexpr const char *const kEnvStringMsDevPolyTOPSVerbosity = "MS_DEV_POLYTOPS_VERBOSITY";
static constexpr const char *const kEnvStringMsDevPolyTOPSCheckSchedules = "MS_DEV_POLYTOPS_CHECK_SCHEDULES";
static constexpr const char *const kEnvStringMsDevPolyTOPSCodeSinking = "MS_DEV_POLYTOPS_CODE_SINKING";
static constexpr const char *const kEnvStringMsDevPolyTOPSConstantToParameter = "MS_DEV_POLYTOPS_CONSTANT_TO_PARAMETER";
static constexpr const char *const kEnvStringMsDevPolyTOPSParameterShifting = "MS_DEV_POLYTOPS_PARAMETER_SHIFTING";
static constexpr const char *const kEnvStringMsDevPolyTOPSPostProcessFullSets =
  "MS_DEV_POLYTOPS_POST_PROCESS_FULL_SETS";
static constexpr const char *const kEnvStringMsDevPolyTOPSPostProcessExtraOuterParallelLoop =
  "MS_DEV_POLYTOPS_POST_PROCESS_EXTRA_OUTER_PARALLEL_LOOP";
static constexpr const char *const kEnvStringMsDevPolyTOPSLargeOuterBounds = "MS_DEV_POLYTOPS_LARGE_OUTER_BOUNDS";
static constexpr const char *const kEnvStringMsDevPolyTOPSEnableSkewing = "MS_DEV_POLYTOPS_ENABLE_SKEWING";
static constexpr const char *const kEnvStringMsDevPolyTOPSEnableParallelSkewingOnly =
  "MS_DEV_POLYTOPS_ENABLE_PARALLEL_SKEWING_ONLY";
static constexpr const char *const kEnvStringMsDevPolyTOPSDumpProblems = "MS_DEV_POLYTOPS_DUMP_PROBLEMS";

isl::schedule_node TileBand(isl::schedule_node node, const isl::multi_val &sizes) {
  isl::ctx ctx = node.ctx();
  int scale_tile;
  int shift_point;

  if (!node.isa<isl::schedule_node_band>()) {
    return node;
  }
  scale_tile = isl_options_get_tile_scale_tile_loops(ctx.get());
  isl_stat status = isl_options_set_tile_scale_tile_loops(ctx.get(), 0);
  CHECK(status == isl_stat_ok);
  shift_point = isl_options_get_tile_shift_point_loops(ctx.get());
  status = isl_options_set_tile_shift_point_loops(ctx.get(), 1);
  CHECK(status == isl_stat_ok);

  isl::schedule_node before_tile = node;
  node = node.as<isl::schedule_node_band>().tile(sizes);

  status = isl_options_set_tile_scale_tile_loops(ctx.get(), scale_tile);
  CHECK(status == isl_stat_ok);
  status = isl_options_set_tile_shift_point_loops(ctx.get(), shift_point);
  CHECK(status == isl_stat_ok);

  return node;
}

size_t CountConsecutiveCoincident(const isl::schedule_node &node) {
  size_t count = 0;
  if (!node.isa<isl::schedule_node_band>()) {
    return count;
  }

  isl::schedule_node_band band_node = node.as<isl::schedule_node_band>();
  while (count < band_node.n_member()) {
    if (!band_node.member_get_coincident(static_cast<int>(count))) {
      break;
    }
    ++count;
  }
  return count;
}

isl::schedule InsertContextNode(const isl::schedule &sch, ScopInfo &scop_info) {
  auto node = sch.root().child(0);
  if (node.isa<isl::schedule_node_context>()) {
    node = node.del();
  }

  // step1. get config
  std::unordered_map<isl::id, int, isl::IslIdIslHash> mapping_ids_with_sizes;
  auto block_cfg = scop_info.user_config_.GetBlockConfig();
  CHECK(block_cfg != nullptr) << "block config is null";

  auto thread_cfg = scop_info.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";

  auto InsertMappingConfig = [&mapping_ids_with_sizes, node](MappingCfg *mapping_cfg) -> void {
    for (size_t i = 0; i < mapping_cfg->bound; ++i) {
      std::pair<std::string, int> pair_i = mapping_cfg->GetAt(i);
      auto id = isl::id(node.ctx(), pair_i.first);
      mapping_ids_with_sizes.insert({id, pair_i.second});
    }
  };

  InsertMappingConfig(block_cfg);
  InsertMappingConfig(thread_cfg);

  auto replace_cfg_map = scop_info.user_config_.GetReplaceConfig();
  for (auto replace_cfg : replace_cfg_map) {
    if (!scop_info.user_config_.GetEnableTensorCoreUsePoly() && replace_cfg.first == WARP_COMPUTE) {
      continue;
    }
    InsertMappingConfig(replace_cfg.second);
  }

  // step2. construct context
  auto space = node.domain().get_space();
  for (auto it = mapping_ids_with_sizes.begin(); it != mapping_ids_with_sizes.end(); ++it) {
    space = space.add_param(it->first);
  }
  isl::set context_set(isl::set::universe(space));
  for (auto it = mapping_ids_with_sizes.begin(); it != mapping_ids_with_sizes.end(); ++it) {
    isl::aff a(isl::aff::param_on_domain(space, it->first));
    context_set = context_set & (isl::aff(a) >= 0) & (isl::aff(a) < it->second);
  }
  scop_info.analysis_result_.RecordContextParams(context_set);
  // step3. insert context
  node = node.insert_context(context_set.from_params());
  return node.get_schedule();
}

isl::union_map DependenceAnalysis(const isl::union_map &sources, const isl::union_map &targets,
                                  const isl::union_map &kills, const isl::union_map &sch) {
  auto access_info = isl::union_access_info(targets);
  access_info = access_info.set_kill(kills);
  access_info = access_info.set_may_source(sources);
  access_info = access_info.set_schedule_map(sch);
  auto union_flow = access_info.compute_flow();
  return union_flow.get_may_dependence();
}

isl::union_map ComputeAllDependences(const isl::schedule &schedule, const isl::union_map &reads_um,
                                     const isl::union_map &writes_um, akg::ir::poly::ScopInfo &scop_info) {
  auto reads = reads_um.domain_factor_domain();
  auto writes = writes_um.domain_factor_domain();
  auto sch = schedule.get_map();

  // RAW
  auto flowDeps = DependenceAnalysis(writes, reads, writes, sch);

  // WAR and WAW
  auto falseDeps = DependenceAnalysis(writes.unite(reads), writes, writes, sch);

#ifdef AKG_USE_POLYTOPS
  constexpr unsigned threshold = 32;
  auto united = flowDeps.unite(falseDeps);
  if (PolyTOPSShouldBeUsed(scop_info) && united.n_map() < threshold) {
    return united;
  } else {
    return united.coalesce();
  }
#else
  return flowDeps.unite(falseDeps).coalesce();
#endif  // AKG_USE_POLYTOPS
}

isl::union_map ComputeRAW(const isl::schedule &schedule, const isl::union_map &reads_um,
                          const isl::union_map &writes_um) {
  auto reads = reads_um.domain_factor_domain();
  auto writes = writes_um.domain_factor_domain();
  auto sch = schedule.get_map();

  // RAW
  return DependenceAnalysis(writes, reads, writes, sch);
}

isl::schedule_node GetOuterBand(const isl::schedule_node &root) {
  auto outer_band = root;

  while (!outer_band.isa<isl::schedule_node_band>()) {
    auto n = outer_band.n_children();
    if (n == 1) {
      outer_band = outer_band.child(0);
      continue;
    } else {
      /*
       * return the node when encountered branching or a leaf
       * an empty band would be inserted elsewhere
       */
      return outer_band;
    }
  }

  return outer_band;
}

bool IsSequenceOrSet(const isl::schedule_node &node) {
  if (node.isa<isl::schedule_node_sequence>()) return true;
  return node.isa<isl::schedule_node_set>();
}

isl::union_map ComputeFilterCopyin(const isl::schedule_node &node, const isl::union_map &ori_reads,
                                   const isl::union_map &ori_writes, const isl::schedule ori_schedule) {
  CHECK(node.isa<isl::schedule_node_filter>()) << "The input should be a filter node!" << std::endl;

  auto filter = node.as<isl::schedule_node_filter>().get_filter();
  auto reads = ori_reads.domain_factor_domain().intersect_domain(filter);
  auto writes = ori_writes.domain_factor_domain().intersect_domain(filter);
  auto uai = isl::union_access_info(reads);
  uai = uai.set_kill(writes);
  uai = uai.set_may_source(writes);
  uai = uai.set_schedule(ori_schedule);
  auto flow = uai.compute_flow();
  auto mayNoSource = flow.get_may_no_source();
  auto copyin = ori_reads.intersect_range(mayNoSource.range());

  return copyin;
}

isl::union_map ComputeFakeCopyin(const isl::schedule &schedule, const isl::union_map &fake_copyin,
                                 const isl::union_map &ori_reads, const isl::union_map &ori_writes) {
  auto root = schedule.get_root();
  auto node = GetOuterBand(root);
  auto result = fake_copyin;

  if (!IsSequenceOrSet(node)) return result;

  auto n = node.n_children();
  for (auto i = 0u; i < n; ++i) {
    auto child = node.child(i);
    auto copyin = ComputeFilterCopyin(child, ori_reads, ori_writes, schedule);
    result = result.unite(copyin);
  }

  return result;
}

isl::schedule_constraints MakeScheduleConstraints(const isl::schedule &schedule, PassInfo &pass_info) {
  auto constraints = isl::schedule_constraints::on_domain(schedule.get_domain());
  constraints = constraints.set_validity(pass_info.dependences_)
                  .set_proximity(pass_info.dependences_)
                  .set_coincidence(pass_info.dependences_);
  return constraints;
}

/*
 * Merge multiple lines of strings into a single-line string
 */
static std::string UndoPrettyPrintSchTree(const std::string &schedule) {
  const char *src = schedule.c_str();
  std::stringstream dst;
  bool in_string = false;
  while (*src != '\0') {
    if (*src == '"') {
      in_string = !in_string;
      if (!in_string) {
        // end of string, find next non-empty char
        const char *next = src + 1;
        while (*next != '\0') {
          char c = *next;
          if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            break;
          }
          ++next;
        }
        if (*next == '"') {
          // multiple consecutive strings, merge them and insert a white space
          dst << " ";
          src = next + 1;
          in_string = true;
          continue;
        }
      }
    }
    dst << *src++;
  }
  return dst.str();
}

bool LoadScheduleTreeFromFile(const std::string &filename, isl::schedule &schedule) {
  std::ifstream new_schedule_file_stream(filename);
  std::string schedule_to_replace_str((std::istreambuf_iterator<char>(new_schedule_file_stream)),
                                      std::istreambuf_iterator<char>());
  schedule_to_replace_str = UndoPrettyPrintSchTree(schedule_to_replace_str);
  if (schedule_to_replace_str == "") {
    return false;
  }
  isl_schedule *ss = isl_schedule_read_from_str(schedule.ctx().get(), schedule_to_replace_str.c_str());
  if (ss != nullptr) {
    schedule = isl::manage(ss);
    return true;
  } else {
    LOG(WARNING) << "Failed to load file " << filename << " to schedule tree, please check syntax of the new schedule.";
    return false;
  }
}

/*
 * Compare and replace schedule hook:
 * Enable users to replace a specific schedule for debugging purpose.
 * If the current schedule is identical to the schedule in OLD_SCHEDULE_FILE,
 * the schedule will be replaced with NEW_SCHEDULE_FILE.
 */
bool ReplaceScheduleTree(isl::schedule &schedule, ScopInfo &info) {
  const std::string OLD_SCHEDULE_FILE = info.AddDumpDir("old_schedule.txt");
  const std::string NEW_SCHEDULE_FILE = info.AddDumpDir("new_schedule.txt");
  // check if two files exist
  char pathBuffOld[PATH_MAX + 1] = {0};
  char pathBuffNew[PATH_MAX + 1] = {0};
  bool should_compare_and_replace = false;
  if (realpath(OLD_SCHEDULE_FILE.c_str(), pathBuffOld) && realpath(NEW_SCHEDULE_FILE.c_str(), pathBuffNew)) {
    FILE *schedule_to_compare = fopen(pathBuffOld, "r");
    FILE *schedule_to_replace = fopen(pathBuffNew, "r");
    should_compare_and_replace = (schedule_to_compare != nullptr && schedule_to_replace != nullptr);
    if (schedule_to_compare != nullptr) {
      int status = fclose(schedule_to_compare);
      if (status != 0) LOG(WARNING) << "Failed to close old_schedule.txt";
    }
    if (schedule_to_replace != nullptr) {
      int status = fclose(schedule_to_replace);
      if (status != 0) LOG(WARNING) << "Failed to close new_schedule.txt";
    }
  }

  if (should_compare_and_replace) {
    std::ifstream old_schedule_file_stream(OLD_SCHEDULE_FILE);
    std::string schedule_to_compare_str((std::istreambuf_iterator<char>(old_schedule_file_stream)),
                                        std::istreambuf_iterator<char>());
    if (CompareSchTreeWithString(schedule_to_compare_str, schedule)) {
      LOG(INFO) << "Current schedule is same as " << OLD_SCHEDULE_FILE << ", replace it with new schedule "
                << NEW_SCHEDULE_FILE;
      CHECK(LoadScheduleTreeFromFile(NEW_SCHEDULE_FILE, schedule));
      return true;
    } else {
      LOG(INFO) << "Current schedule is different from " << OLD_SCHEDULE_FILE << ", not replacing.";
    }
  }
  return false;
}

std::string GetPromotionTensorName(const isl::schedule_node &node, const std::vector<BufferDefInfo> &buffer_def_infos) {
  std::string id_name = "";
  if (!node.isa<isl::schedule_node_band>()) {
    return id_name;
  }
  for (size_t i = 0; i < buffer_def_infos.size(); ++i) {
    auto tensor_id = buffer_def_infos[i].tensor_id;
    isl::union_set id_domain = node.as<isl::schedule_node_band>().get_partial_schedule().domain();
    id_domain = id_domain.unwrap().range();
    id_domain.foreach_set([tensor_id, &id_name](const isl::set &s) -> void {
      std::string node_tensor_name = s.get_tuple_name();
      size_t pos = 0;
      if ((pos = node_tensor_name.find(LOCAL_SUFFIX)) != std::string::npos ||
          (pos = node_tensor_name.find(SHARE_SUFFIX)) != std::string::npos) {
        node_tensor_name = node_tensor_name.erase(pos, node_tensor_name.size() - pos);
      }

      if ((pos = node_tensor_name.find(PROMOTION_INFIX)) != std::string::npos) {
        node_tensor_name = node_tensor_name.erase(pos, node_tensor_name.size() - pos);
      }
      id_name = (node_tensor_name == tensor_id.get_name()) ? node_tensor_name : id_name;
    });

    if (!id_name.empty()) {
      break;
    }
  }
  return id_name;
}

bool IsReadOrWriteTensor(const isl::schedule_node &node, const std::string &read_name, const std::string &write_name) {
  // transform isl::union_set to a vector of isl::set
  if (!node.isa<isl::schedule_node_filter>()) {
    return false;
  }
  isl::union_set uset = node.as<isl::schedule_node_filter>().get_filter();
  std::vector<isl::set> vset;
  uset.foreach_set([&vset](isl::set s) { vset.push_back(s); });

  bool is_all_sets_read_or_write = std::all_of(vset.begin(), vset.end(), [read_name, write_name](isl::set s) {
    auto read_id = isl::id(s.ctx(), read_name);
    auto write_id = isl::id(s.ctx(), write_name);
    return s.get_tuple_id() == read_id || s.get_tuple_id() == write_id;
  });
  return is_all_sets_read_or_write;
}

isl::schedule_node GetCanMappingNode(const isl::schedule_node &node) {
  // It is not allowed multi filter-band pairs below a read/write filter.
  int count_filter_band_pair = 0;
  node.foreach_descendant_top_down([&count_filter_band_pair](const isl::schedule_node &sub_node) -> bool {
    if (sub_node.isa<isl::schedule_node_filter>() && sub_node.n_children() > 0 &&
        sub_node.child(0).isa<isl::schedule_node_band>()) {
      count_filter_band_pair++;
    }
    return true;
  });
  CHECK(count_filter_band_pair == 1) << "multi filter-> band pairs exist in a read/write filter subtree.";

  auto band_node = node.child({0});
  CHECK(band_node.isa<isl::schedule_node_band>()) << "Type of Node must be band.";

  return band_node;
}

#ifdef AKG_USE_POLYTOPS
bool PolyTOPSShouldBeUsed(akg::ir::poly::ScopInfo &scop_info) {
  const auto tol = [](unsigned char c) { return std::tolower(c); };

  bool automatic = false;

  // We check the environment first, it overrides the scop_info.
  // Don't perform the extra checks if we recognize "isl" or "polytops".
  // Otherwise, "auto" mode will let the remainder of the function decide.
  const char *const ms_dev_scheduler = std::getenv(kEnvStringMsDevPolyScheduler);
  if (ms_dev_scheduler) {
    std::string env_string(ms_dev_scheduler);
    std::transform(env_string.begin(), env_string.end(), env_string.begin(), tol);
    if (env_string == "isl") {
      return false;
    } else if (env_string == "polytops") {
      return true;
    } else if (env_string != "auto") {
      // warning? default to never, always, auto?
    } else {
      automatic = true;
    }
  }

  if (!automatic) {
    // If no definitive decision can be taken from the environment we check the scop_info
    std::string enable_polytops = scop_info.user_config_.GetEnablePolyTOPS();
    std::transform(enable_polytops.begin(), enable_polytops.end(), enable_polytops.begin(), tol);
    if (enable_polytops == "never") {
      return false;
    } else if (enable_polytops == "always") {
      return true;
    } else if (enable_polytops != "auto") {
      // warning, default to... never, always, auto?
    }
  }

  // We reach this point if:
  // - the environment did not explicitely specify "isl" nor "polytops"
  // - enable_polytops did not explicitely specify "nerver" nor "always"

  // Explicitly avoid reduce and matmul cases
  if (scop_info.analysis_result_.GetUseGpuReduceLib() ||
      scop_info.analysis_result_.GetOpTemplate() == Template::MATMUL ||
      scop_info.analysis_result_.GetOpTemplate() == Template::REDUCTION ||
      scop_info.analysis_result_.GetOpTemplate() == Template::BITWISE_REDUCTION ||
      !scop_info.analysis_result_.GetReduceTensorInfoMap().empty() || scop_info.user_config_.GetEnableMatmul() ||
      scop_info.user_config_.GetEnableTensorCore() || scop_info.user_config_.GetEnableConvTensorCore()) {
    return false;
  }

  // Check the kernel name
  const std::vector<std::string> unsupported_kernels = {
    "reduce",
    "matmul",
  };
  std::string kernel = scop_info.user_config_.GetKernelName();
  std::transform(kernel.begin(), kernel.end(), kernel.begin(), [](unsigned char c) { return std::tolower(c); });
  for (auto substring : unsupported_kernels) {
    if (kernel.find(substring) != std::string::npos) {
      return false;
    }
  }

  // Inspect read and writes to detect potential reductions
  const isl::union_map reads = scop_info.analysis_result_.GetReads().domain_factor_domain();
  const isl::union_map writes = scop_info.analysis_result_.GetWrites().domain_factor_domain();
  const isl::union_map intersection = reads.intersect(writes);
  if (!intersection.is_empty() && !intersection.is_injective()) {
    return false;
  }

  return true;
}

bool PolyTOPSShouldCheckSchedules(const akg::ir::poly::ScopInfo &scop_info) {
  bool should_check = scop_info.user_config_.GetPolyTOPSCheckSchedules();
  const char *const env_str = std::getenv(kEnvStringMsDevPolyTOPSCheckSchedules);
  if (env_str) {
    const auto tol = [](unsigned char c) { return std::tolower(c); };
    std::string env_string(env_str);
    std::transform(env_string.begin(), env_string.end(), env_string.begin(), tol);
    if (env_string == "true") {
      should_check = true;
    } else if (env_string == "false") {
      should_check = false;
    } else {
      // warn for unrecognized string?
    }
  }

  return should_check;
}

polytops::bin::Options PolyTOPSOptionsInit(const akg::ir::poly::PassInfo &pass_info,
                                           const akg::ir::poly::ScopInfo &scop_info) {
  auto env_to_bool = [](const char *const environment_variable, bool default_value = false) {
    bool result = default_value;
    const char *const env_cstr = std::getenv(environment_variable);
    if (env_cstr) {
      const std::string env_str(env_cstr);
      if (env_str == "true") {
        result = true;
      } else if (env_str == "false") {
        result = false;
      }
    }
    return result;
  };

  // Choose a solver from the ScopInfo
  const std::string solver_string = scop_info.user_config_.GetPolyTOPSSolver();
  polytops::bin::Options::SolverType solver_type = polytops::bin::Options::SolverTypeFromString(solver_string.c_str());
  // Maybe override from the environment
  const char *const ms_dev_polytops_solver = std::getenv(kEnvStringMsDevPolyTOPSSolver);
  if (ms_dev_polytops_solver) {
    const std::string env_string(ms_dev_polytops_solver);
    const polytops::bin::Options::SolverType env_solver_type =
      polytops::bin::Options::SolverTypeFromString(env_string.c_str());
    if (env_solver_type != polytops::bin::Options::SolverType::kNone) {
      solver_type = env_solver_type;
    }
  }
  if (solver_type == polytops::bin::Options::SolverType::kNone) {
    // Select a default if for some reason no valid solver could be selected
    solver_type = polytops::bin::Options::SolverType::kQiuqiIp;
  }

  unsigned long int verbosity = polytops::bin::Options::GetDefaultVerbosity();
  const char *const ms_dev_polytops_verbosity = std::getenv(kEnvStringMsDevPolyTOPSVerbosity);
  if (ms_dev_polytops_verbosity) {
    verbosity = std::stoul(ms_dev_polytops_verbosity);
  }

  bool code_sinking = scop_info.user_config_.GetPolyTOPSCodeSinking() || pass_info.dependences_.is_empty();
  code_sinking = env_to_bool(kEnvStringMsDevPolyTOPSCodeSinking, code_sinking);

  bool constant_to_parameter = scop_info.user_config_.GetPolyTOPSConstantToParameter();
  constant_to_parameter = env_to_bool(kEnvStringMsDevPolyTOPSConstantToParameter, constant_to_parameter);

  bool parameter_shifting = scop_info.user_config_.GetPolyTOPSParameterShifting();
  parameter_shifting = env_to_bool(kEnvStringMsDevPolyTOPSParameterShifting, parameter_shifting);

  bool post_process_full_sets = scop_info.user_config_.GetPolyTOPSPostProcessingFullSets();
  post_process_full_sets = env_to_bool(kEnvStringMsDevPolyTOPSPostProcessFullSets, post_process_full_sets);

  bool post_process_extra_outer_parallel_loop =
    scop_info.user_config_.GetPolyTOPSPostProcessingExtraOuterParallelLoop();
  post_process_extra_outer_parallel_loop =
    env_to_bool(kEnvStringMsDevPolyTOPSPostProcessExtraOuterParallelLoop, post_process_extra_outer_parallel_loop);

  bool large_outer_bounds = scop_info.user_config_.GetPolyTOPSLargeOuterBounds();
  large_outer_bounds = env_to_bool(kEnvStringMsDevPolyTOPSLargeOuterBounds, large_outer_bounds);

  bool enable_skewing = scop_info.user_config_.GetPolyTOPSEnableSkewing();
  enable_skewing = env_to_bool(kEnvStringMsDevPolyTOPSEnableSkewing, enable_skewing);

  bool enable_parallel_skewing_only = scop_info.user_config_.GetPolyTOPSEnableParallelSkewingOnly();
  enable_parallel_skewing_only =
    env_to_bool(kEnvStringMsDevPolyTOPSEnableParallelSkewingOnly, enable_parallel_skewing_only);

  bool dump_problems = scop_info.user_config_.GetPolyTOPSDumpProblems();
  dump_problems = env_to_bool(kEnvStringMsDevPolyTOPSDumpProblems, dump_problems);

  polytops::bin::Options result;
  result.SetSolverType(solver_type);
  result.SetVerbosity(verbosity);
  result.SetCodeSinking(code_sinking);
  result.SetConstantToParameter(constant_to_parameter);
  result.SetParameterShifting(parameter_shifting);
  result.SetFullSetsPostProcessing(post_process_full_sets);
  result.SetExtraParallelOuterLoopPostProcessing(post_process_extra_outer_parallel_loop);
  result.SetEnableLargeOuterBounds(large_outer_bounds);
  result.SetEnableSkewing(enable_skewing);
  result.SetEnableParallelSkewingOnly(enable_parallel_skewing_only);
  result.SetDumpProblems(dump_problems);

  return result;
}

polytops::bin::Hints ExtractDirectivesFromAKG(ScopInfo &scop_info) {
  polytops::bin::Hints hints;

  ForTypeMap directives = scop_info.analysis_result_.GetForTypeMap();
  std::map<std::string, std::vector<int>> serials_directive;
  std::map<std::string, std::vector<int>> vectorials_directive;
  std::map<std::string, std::vector<int>> parallels_directive;
  std::map<std::string, std::vector<int>> reduces_directive;
  for (const auto &[stmt, vloop_directive] : directives) {
    const std::string stmt_string = stmt.get_name();
    for (uint ivd = 0; ivd < vloop_directive.size(); ++ivd) {
      const int i = static_cast<int>(ivd);
      switch (vloop_directive[i]) {
        case ForType::Serial:
          break;
        case ForType::Invariant:
          LOG(INFO) << stmt_string << " invariant_for";
          serials_directive[stmt_string].push_back(i);
          break;
        case ForType::Parallel:
          LOG(INFO) << stmt_string << " parallel";
          parallels_directive[stmt_string].push_back(i);
          break;
        case ForType::Vectorized:
        case ForType::Swizzled:  // treat "Swizzled" like "Vectorized" for the moment
          LOG(INFO) << stmt_string << " vectorized";
          vectorials_directive[stmt_string].push_back(i);
          break;
        case ForType::Reduce:
          LOG(INFO) << stmt_string << " reduce";
          reduces_directive[stmt_string].push_back(i);
          break;
        case ForType::Unrolled:
          LOG(WARNING) << stmt_string << " Do not treat ForType::Unrolled as a directives";
          break;
        default:
          LOG(WARNING) << stmt_string << " Unknow ForType loop";
          break;
      }
    }
  }

  for (const auto &[key, directive] : serials_directive) {
    hints.SetStatementSerials(key.c_str(), directive);
  }
  for (const auto &[key, directive] : vectorials_directive) {
    hints.SetStatementVectorials(key.c_str(), directive);
  }
  for (const auto &[key, directive] : parallels_directive) {
    hints.SetStatementParallels(key.c_str(), directive);
  }
  for (const auto &[key, directive] : reduces_directive) {
    hints.SetStatementReduces(key.c_str(), directive);
  }

  return hints;
}
#endif

}  // namespace poly
}  // namespace ir
}  // namespace akg
