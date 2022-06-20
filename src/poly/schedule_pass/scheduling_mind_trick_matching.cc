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
#ifdef AKG_USE_POLYTOPS

#include "poly/schedule_pass/scheduling_mind_trick.h"

#include "poly/log_util.h"

namespace akg {
namespace ir {
namespace poly {

////////////////////////////////////////////////////////////////////////////////
// Local declarations
////////////////////////////////////////////////////////////////////////////////

static inline isl_schedule_node_type GetScheduleNodeType(const isl::schedule_node &node);
static inline std::vector<isl::schedule_node> CollectScheduleNodes(const isl::schedule &sch, const int verbosity);
static inline bool DomainMatches(const isl::schedule &pattern, const isl::schedule &candidate, const std::string &name_,
                                 const int verbosity);
static inline bool NodesMatch(const isl::schedule_node &pattern_node, const isl::schedule_node &candidate_node,
                              const std::string &name_, const int verbosity);
static inline bool PatternMatches(const isl::schedule &sch, const std::string &pattern_, const std::string &name_,
                                  const int verbosity);

////////////////////////////////////////////////////////////////////////////////
// Patter-matching related in SchedulingMindTrick public API
////////////////////////////////////////////////////////////////////////////////

bool SchedulingMindTrick::Matches(const isl::schedule &sch) const {
  if (operator_ != "" && pattern_ != "") {
    Info(log::Verbosity::medium, "note that this mind trick contains both a pattern and an operator name...");
  }

  if (PatternMatches(sch, pattern_, name_, verbosity_)) {
    Info(log::Verbosity::low, text_green "schedule pattern matches!");
    return true;
  } else if (operator_ == scop_info_.user_config_.GetKernelName()) {
    Info(log::Verbosity::low, text_green "operator name matches!");
    return true;
  } else if (pattern_ == "" && operator_ == "") {
    Warn(log::Verbosity::veryLow, "pattern-free and operator-free mind_trick: matches! (are you sure?)");
    return true;
  } else {
    Info(log::Verbosity::medium, "schedule does not match!");
    return false;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Local definitions
////////////////////////////////////////////////////////////////////////////////

isl_schedule_node_type GetScheduleNodeType(const isl::schedule_node &node) {
  isl_schedule_node_type type = isl_schedule_node_get_type(node.get());
  switch (type) {
    case isl_schedule_node_band:
      return isl_schedule_node_band;
    case isl_schedule_node_context:
      return isl_schedule_node_context;
    case isl_schedule_node_domain:
      return isl_schedule_node_domain;
    case isl_schedule_node_extension:
      return isl_schedule_node_extension;
    case isl_schedule_node_filter:
      return isl_schedule_node_filter;
    case isl_schedule_node_guard:
      return isl_schedule_node_guard;
    case isl_schedule_node_mark:
      return isl_schedule_node_mark;
    case isl_schedule_node_leaf:
      return isl_schedule_node_leaf;
    case isl_schedule_node_sequence:
      return isl_schedule_node_sequence;
    case isl_schedule_node_set:
      return isl_schedule_node_set;
    default:
      assert(false && "cannot convert the node type");
      return isl_schedule_node_leaf;
  }
}

std::vector<isl::schedule_node> CollectScheduleNodes(const isl::schedule &sch, const int verbosity) {
  std::vector<isl::schedule_node> nodes;

  isl::schedule_node root = sch.get_root();
  if (GetScheduleNodeType(root.child(0)) == isl_schedule_node_leaf) {
    log::Info(log::Verbosity::low, "Root node has no children; returning empty vector of schedule nodes");
    return nodes;
  } else {
    auto collect = [&nodes](isl::schedule_node node) {
      nodes.push_back(node);
      return true;
    };
    root.foreach_descendant_top_down(collect);
  }
  return nodes;
}

bool DomainMatches(const isl::schedule &pattern, const isl::schedule &candidate, const std::string &name_,
                   const int verbosity) {
  log::Info(log::Verbosity::high, name_ + ": checking if candidate schedule domain matches...");

  if (pattern.get_domain().is_empty()) {
    log::Warn(log::Verbosity::high,
              name_ + ": pattern's domain is completely unspecified therefore any domain matches");
    return true;
  }
  isl::set_list dom_patt_list = pattern.get_domain().get_set_list();
  isl::set_list dom_can_list = candidate.get_domain().get_set_list();

  if (dom_patt_list.size() != dom_can_list.size()) {
    log::Warn(log::Verbosity::high,
              name_ + ": candidate does not have the same number of statement as pattern. Does not match");
    return false;
  }

  for (unsigned i = 0; i < dom_patt_list.size(); ++i) {
    isl::set patt_i = dom_patt_list.get_at(i);
    isl::id name_i = patt_i.get_tuple_id();
    isl::space space_i = patt_i.get_space();

    isl::set can_j;
    isl::id name_j;
    isl::space space_j;

    bool exists = false;

    for (unsigned j = 0; j < dom_can_list.size(); ++j) {
      can_j = dom_can_list.get_at(j);
      name_j = can_j.get_tuple_id();
      space_j = can_j.get_space();

      if ((name_j.get_name() == name_i.get_name()) && (space_i.is_equal(space_j))) {
        exists = true;

        log::Info(log::Verbosity::high,
                  name_ + ": statement " + space_i.to_str() + " exists; proceeding with further checks...");

        isl::space space_i = patt_i.get_space();
        unsigned dim_num_i = space_i.dim(isl_dim_set);

        for (unsigned k = 0; k < dim_num_i; ++k) {
          // There may be cases where underspecified statements appears in the candidate tree
          // ex:  domain : "{S_0[]}" (variable initizalization or reductions)
          // We must distinguish such cases from that of underspecified patterns (for which we bypass
          // isl_exceptions in order to still capture their specification.

          // The previous tests on spaces should ensure that, at this point of the code,
          // only underspecified patterns can be encountered.
          // But to be very sure about that, the following check eliminates any possibility of attempting
          // pattern matching on a candidate that contains any S[]
          isl_bool dim_id_check = isl_set_has_dim_id(can_j.get(), static_cast<enum isl_dim_type>(isl_dim_set), k);

          if (!dim_id_check) {
            log::Warn(log::Verbosity::high,
                      name_ + ": statement " + name_i.get_name() +
                        " has no dim id. Cannot perform pattern matching on an underspecified candidate");
            return false;
          }

          isl::id dim_name_j =
            isl::manage(isl_set_get_dim_id(can_j.get(), static_cast<enum isl_dim_type>(isl_dim_set), k));

          isl::pw_aff min_j = can_j.dim_min(k);
          isl::pw_aff max_j = can_j.dim_max(k);

          std::stringstream log_prefix_stream;
          log_prefix_stream << name_ << ": candidate bounds for dimension " << k << " of " << name_i.get_name();
          const std::string &log_prefix = log_prefix_stream.str();
          try {
            isl::id dim_name_i =
              isl::manage(isl_set_get_dim_id(patt_i.get(), static_cast<enum isl_dim_type>(isl_dim_set), k));

            isl::pw_aff min_i = patt_i.dim_min(k);
            isl::pw_aff max_i = patt_i.dim_max(k);

            if (dim_name_i.to_str() != "inf") {
              if (!(min_i.is_equal(min_j) && max_i.is_equal(max_j))) {
                log::Warn(log::Verbosity::high,
                          log_prefix + " are not equal to pattern bounds; domain does not match\n");
                return false;
              }
            }
            log::Info(log::Verbosity::high, log_prefix + " match");
          } catch (isl::exception &) {
            log::Info(log::Verbosity::high, log_prefix + " are undefined; any bound matches");
          }
        }
      }
    }
    if (!exists) {
      log::Warn(log::Verbosity::high,
                name_ + ": statement " + space_i.to_str() + " does not exist; domain does not match");
      return false;
    }
  }
  log::Info(log::Verbosity::high, name_ + ": candidate domain matches\n");
  return true;
}

bool NodesMatch(const isl::schedule_node &pattern_node, const isl::schedule_node &candidate_node,
                const std::string &name_, const int verbosity) {
  isl_schedule_node_type pnode_type = GetScheduleNodeType(pattern_node);
  isl_schedule_node_type cnode_type = GetScheduleNodeType(candidate_node);

  // At the first mismatch, return false
  if (pnode_type != cnode_type) {
    return false;
  }

  // If encountering schedule node bands or filters,
  // ensure that their specification is equal
  if (pnode_type == isl_schedule_node_band) {
    isl::multi_union_pw_aff pnode_bsched = isl::manage(isl_schedule_node_band_get_partial_schedule(pattern_node.get()));
    isl::multi_union_pw_aff cnode_bsched =
      isl::manage(isl_schedule_node_band_get_partial_schedule(candidate_node.get()));

    if (!cnode_bsched.plain_is_equal(pnode_bsched)) {
      log::Warn(log::Verbosity::high, name_ + "\' : schedule bands are not equal. Schedule tree does not match");
      return false;
    }
  }

  if (pnode_type == isl_schedule_node_filter) {
    isl::union_set pfilter_info = isl::manage(isl_schedule_node_filter_get_filter(pattern_node.get()));
    isl::union_set cfilter_info = isl::manage(isl_schedule_node_filter_get_filter(candidate_node.get()));

    if (!pfilter_info.is_empty()) {
      isl::set_list plist = pfilter_info.get_set_list();
      isl::set_list clist = cfilter_info.get_set_list();

      for (unsigned i = 0; i < plist.size(); ++i) {
        try {
          isl::id pid = plist.get_at(i).get_tuple_id();

          if (!pid.is_null()) {
            bool exists = false;

            for (unsigned j = 0; j < clist.size(); ++j) {
              isl::id cid = clist.get_at(j).get_tuple_id();

              if (!cid.is_null() && (pid.get_name() == cid.get_name())) {
                exists = true;
              }
            }
            if (!exists) {
              log::Warn(
                log::Verbosity::high,
                "\'" + name_ + "\' candidate filter tuple id does not match with pattern (" + pid.get_name() + ")");
              return false;
            }
          }
        } catch (isl::exception_invalid &) {
          log::Info(log::Verbosity::high, "\'" + name_ + "\' : no tuple id specified, therefore any tuple id matches");
        }
      }
    }
  }

  // Return true if none of the previous tests failed
  return true;
}

bool PatternMatches(const isl::schedule &sch, const std::string &pattern_, const std::string &name_,
                    const int verbosity) {
  if (pattern_ != "") {
    isl::schedule pattern = isl::schedule(sch.ctx(), pattern_);

    if (!DomainMatches(pattern, sch, name_, verbosity)) {
      return false;
    }

    log::Info(log::Verbosity::high, " \'" + name_ + "\' : collecting pattern nodes and candidate nodes...");

    std::vector<isl::schedule_node> pattern_nodes = CollectScheduleNodes(pattern, verbosity);
    std::vector<isl::schedule_node> candidate_nodes = CollectScheduleNodes(sch, verbosity);

    // Iterate according to the number of nodes in the pattern to ensure that
    // the pattern matches at least part of the candidate schedule tree
    if (pattern_nodes.size() == 0) {
      log::Info(
        log::Verbosity::high,
        " \'" + name_ + "\' : no schedule tree pattern is specified, therefore any candidate schedule tree matches");
      return true;
    }

    // The pattern may not be found at the root of the candidate tree.
    // We therefore first search for any nodes index that matches the first node
    // of the pattern to capture starting points for checking if there is a match
    std::vector<unsigned> entry_indexes;
    for (unsigned i = 0; i < candidate_nodes.size(); ++i) {
      if (NodesMatch(pattern_nodes[0], candidate_nodes[i], name_, verbosity)) {
        entry_indexes.push_back(i);
      }
    }
    if (entry_indexes.size() == 0) {
      return false;
    }

    bool match = false;
    for (unsigned k = 0; k < entry_indexes.size(); ++k) {
      unsigned i;
      for (i = 1; i < pattern_nodes.size() - 1; ++i) {
        unsigned j = entry_indexes[k];
        bool check = NodesMatch(pattern_nodes[i], candidate_nodes[j + i], name_, verbosity);
        if (!check) break;
      }

      // With the way we expect patterns to be specified, for now, there
      // can be only one possible match if any. so as soon as we match
      // something, we quit the loop over k.
      if (i == pattern_nodes.size() - 1) {
        match = true;
      }
    }
    if (match == false) {
      log::Warn(log::Verbosity::high, " \'" + name_ + "\' : the candidate schedule tree does not match");
      return false;
    }

    // If the code made it this far, then schedules match
    return true;
  } else {
    return false;
  }
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // AKG_USE_POLYTOPS
