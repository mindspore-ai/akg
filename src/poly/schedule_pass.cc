/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
}  // namespace poly
}  // namespace ir
}  // namespace akg
