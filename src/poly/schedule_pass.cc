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
/* Reorder filters of a sequence/set node.
 * node: must be a sequence or set node.
 * old_to_new_map: map from original child position to new child position.
 * The caller should make sure that there are no duplicate values.
 */
isl::schedule_node ReorderFilters(const isl::schedule_node &node,
                                  const std::unordered_map<size_t, size_t> &old_to_new_map) {
  auto n_children = node.n_children();
  isl_schedule_tree *old_tree = isl_schedule_node_get_tree(node.get());
  CHECK(old_tree != nullptr);
  isl_schedule_tree *new_tree = isl_schedule_node_get_tree(node.get());
  CHECK(new_tree != nullptr);
  for (auto &it : old_to_new_map) {
    auto old_pos = it.first;
    auto new_pos = it.second;
    CHECK(old_pos < n_children);
    CHECK(new_pos < n_children);
    isl_schedule_tree *old_child = isl_schedule_tree_get_child(old_tree, old_pos);
    CHECK(old_child != nullptr);
    new_tree = isl_schedule_tree_replace_child(new_tree, new_pos, old_child);
    CHECK(new_tree != nullptr);
  }
  static_cast<void>(isl_schedule_tree_free(old_tree));
  isl_schedule_node *new_node = isl_schedule_node_graft_tree(node.copy(), new_tree);
  CHECK(new_node != nullptr);
  return isl::manage(new_node);
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
                                     const isl::union_map &writes_um) {
  auto reads = reads_um.domain_factor_domain();
  auto writes = writes_um.domain_factor_domain();
  auto sch = schedule.get_map();

  // RAW
  auto flowDeps = DependenceAnalysis(writes, reads, writes, sch);

  // WAR and WAW
  auto falseDeps = DependenceAnalysis(writes.unite(reads), writes, writes, sch);

  return flowDeps.unite(falseDeps).coalesce();
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
  isl::schedule_constraints constraints;
  if (pass_info.coincident_) {
    constraints = isl::schedule_constraints::on_domain(schedule.get_domain())
                    .set_coincidence(pass_info.dependences_)  // keep it, check for more cases
                    .set_validity(pass_info.dependences_)
                    .set_proximity(pass_info.dependences_);
  } else {
    constraints = isl::schedule_constraints::on_domain(schedule.get_domain())
                    .set_validity(pass_info.dependences_)
                    .set_proximity(pass_info.dependences_);
  }
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

std::vector<int> GetTileSizeOfLevel(const int member_size, const int dim_size, const std::string &tile_level,
                                    TileSizes tile_sizes, const int count_coincident,
                                    const std::vector<int> &warp_list) {
  std::vector<int> tile_size(member_size, 0);
  for (auto i = 0; i < member_size; ++i) {
    if (i >= dim_size) {
      tile_size[i] = MAX_STRIDE;
      continue;
    }
    // tile_size maybe bigger than dim_num
    if (tile_level == TILE_WITH_C0) {
      tile_size[i] = static_cast<int>(tile_sizes[i].c0_tiling_size);
    } else if (tile_level == TILE_WITH_C1) {
      tile_size[i] = static_cast<int>(tile_sizes[i].c1_tiling_size);
    } else if (tile_level == TILE_WITH_WARP_C1) {
      tile_size[i] = warp_list[i];
    } else if (tile_level == TILE_WITH_LAST_C1) {
      tile_size[i] = static_cast<int>(tile_sizes[tile_sizes.size() - 1 - i].c1_tiling_size);
    } else if (tile_level == TILE_WITH_LAST_C0) {
      tile_size[i] = static_cast<int>(tile_sizes[tile_sizes.size() - 1 - i].c0_tiling_size);
    } else {
      // The tiling size of n and m is warp_number times of c0_tiling_size, which is equivalent to extracting the for
      // loop generated during mapping.This avoids the if condition and facilitates isl_emitter.
      tile_size[i] = (i < count_coincident) ? static_cast<int>(tile_sizes[i].c1_tiling_size)
                                            : static_cast<int>(tile_sizes[i].c0_tiling_size);
    }
  }
  return tile_size;
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

}  // namespace poly
}  // namespace ir
}  // namespace akg
