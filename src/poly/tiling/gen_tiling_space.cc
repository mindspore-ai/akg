
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

#include <iostream>
#include <unordered_map>

#include "poly/scop_info.h"
#include "poly/tiling/tiling_analyzer.h"
#include "poly/tiling/tile_space.h"

namespace akg {
namespace ir {
namespace poly {
class TileSpaceCollector {
 public:
  TileSpaceCollector(TilingAnalyzer &analyzer, const int level)
      : space_(make_node<air::TileSpaceNode>()), analyzer_(analyzer), cand_(&analyzer), level_(level) {
    air::runtime::NDArray init_array = air::runtime::NDArray::Empty({}, type, ctx);
    space_->index_table = init_array;
    space_->c1_tile_range_table = init_array;
    space_->c0_tile_range_table = init_array;
    space_->c1_tile_mod_table = init_array;
    space_->c0_tile_mod_table = init_array;
    space_->tiling_candidate = init_array;
    space_->gpu_thread_range_table = init_array;
    space_->gpu_block_range_table = init_array;
    space_->gpu_thread_mod_table = init_array;
    space_->gpu_block_mod_table = init_array;
    if (analyzer_.scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
      cared_info_ = {"index", "C1_range", "C0_range", "C1_mod", "C0_mod", "gpu_thread_range", "gpu_block_range", "gpu_thread_mod", "gpu_block_mod"};
    } else {
      cared_info_ = {"index", "C1_range", "C0_range", "C1_mod", "C0_mod"};
    }
  }
  ~TileSpaceCollector() = default;

  air::TileSpace GetSpace() { return air::TileSpace(space_); }

  void Collect() {
    size_t band_size = analyzer_.RootAxis()->children.size();
    CollectMemLimit();
    CollectSharedAxis(band_size);
    for (size_t i = 0; i < band_size; ++i) {
      result_.emplace_back(std::vector<Result>());
      CollectTileAxisTopDown(i);
      static_cast<void>(ScanDown(0, i));
      LOG(INFO) << "Band = " << i << ", tiling space size: " << result_.back().size();
    }

    if (band_size == 1) {  // fast path
      int tile_size = analyzer_.GetNumOfAxisInBand(0);
      CollectConstraint(tile_size, band_size);
      auto &result = result_[0];
      space_->tiling_candidate = air::runtime::NDArray::Empty(
        {static_cast<int64_t>(result.size()), static_cast<int64_t>(tile_size)}, type, ctx);
      auto spaceTilingDlPack = space_->tiling_candidate.ToDLPack();
      auto ptr = reinterpret_cast<int *>(spaceTilingDlPack->dl_tensor.data);
      for (auto i = 0; i < static_cast<int>(result.size()); ++i) {
        for (int j = 0; j < tile_size; ++j) {
          ptr[i * tile_size + j] = result[i].tile[j];
        }
      }
      delete spaceTilingDlPack;
    } else {
      std::vector<int> band_idx;
      std::vector<std::vector<int>> result;
      int tile_size = 0;
      for (auto &band_result : result_) {
        if (band_result.empty()) continue;
        band_idx.push_back(tile_size);
        tile_size += static_cast<int>(band_result[0].tile.size());
      }
      std::vector<int> tile(tile_size, 0);
      CombineBand(0, band_idx, tile, result);
      CollectConstraint(tile_size, band_size);
      FreeResult();
      space_->tiling_candidate = air::runtime::NDArray::Empty(
        {static_cast<int64_t>(result.size()), static_cast<int64_t>(tile_size)}, type, ctx);
      auto spaceTilingDlPack = space_->tiling_candidate.ToDLPack();
      auto ptr2 = reinterpret_cast<int *>(spaceTilingDlPack->dl_tensor.data);
      for (auto i = 0; i < static_cast<int>(result.size()); ++i) {
        for (int j = 0; j < tile_size; ++j) {
          ptr2[i * tile_size + j] = result[i][j];
        }
      }
      delete spaceTilingDlPack;
    }
  }

  void FreeResult() {
    std::vector<std::vector<Result>> empty;
    std::swap(result_, empty);
  }

  void CollectConstraint(int tile_size, size_t band_size) {
    if (tile_size == 0) return;
    // step 1. collect all axes in all bands
    std::vector<std::vector<TileAxis *>> all_axes;
    for (auto b_idx = 0; b_idx < static_cast<int>(band_size); ++b_idx) {
      std::vector<TileAxis *> band_axes;
      auto CollectTileAxis = [this, b_idx, &band_axes](TileAxis *a) {
        if (a == analyzer_.RootAxis()) return;
        if (a->index == b_idx) {
          band_axes.emplace_back(a);
        }
      };
      analyzer_.ForEachAxisTopDown(CollectTileAxis);
      all_axes.emplace_back(band_axes);
    }

    // step 2. collect cared info from each axis
    for (const auto &con : cared_info_) {
      int length = con.find("mod") != std::string::npos ? 1 : 2;
      auto size = static_cast<int64_t>(tile_size);
      if (con.find("gpu") != std::string::npos) {
        size = std::max<int64_t>(3, size);
      }
      auto array = air::runtime::NDArray::Empty({size, length}, type, ctx);
      auto spaceDlPack = array.ToDLPack();
      auto ptr = reinterpret_cast<int *>(spaceDlPack->dl_tensor.data);
      if (con.find("gpu") != std::string::npos) {
        size_t s = con.find("thread") != std::string::npos ? 0 : 3;
        size_t e = con.find("thread") != std::string::npos ? 3 : 6;
        for (size_t i = s; i < e; ++i) {
          if (length == 1) {
            *ptr++ = analyzer_.binding_spaces_[i].map_mod_;
          } else {
            *ptr++ = analyzer_.binding_spaces_[i].map_min_;
            *ptr++ = analyzer_.binding_spaces_[i].map_extent_;
          }
        }
      } else {
        for (size_t b_idx = 0; b_idx < all_axes.size(); ++b_idx) {
          for (size_t a_idx = 0; a_idx < all_axes[b_idx].size(); ++a_idx) {
            if (con == "index") {
              *ptr++ = b_idx;
              *ptr++ = a_idx;
            } else {
              if (con == "C1_range") {
                TileAxis::Constraint const_cons = all_axes[b_idx][a_idx]->GetConstConstraint(CACHE1);
                *ptr++ = const_cons.tile_min_.as<IntImm>()->value;
                *ptr++ = const_cons.tile_extent_.as<IntImm>()->value;
              } else if (con == "C0_range") {
                TileAxis::Constraint const_cons = all_axes[b_idx][a_idx]->GetConstConstraint(CACHE0);
                *ptr++ = const_cons.tile_min_.as<IntImm>()->value;
                *ptr++ = const_cons.tile_extent_.as<IntImm>()->value;
              } else if (con == "C1_mod") {
                TileAxis::Constraint const_cons = all_axes[b_idx][a_idx]->GetConstConstraint(CACHE1);
                *ptr++ = const_cons.tile_mod_.as<IntImm>()->value;
              } else if (con == "C0_mod") {
                TileAxis::Constraint const_cons = all_axes[b_idx][a_idx]->GetConstConstraint(CACHE0);
                *ptr++ = const_cons.tile_mod_.as<IntImm>()->value;
              }
            }
          }
        }
      }

      if (con == "index") space_->index_table = array;
      if (con == "C1_range") space_->c1_tile_range_table = array;
      if (con == "C0_range") space_->c0_tile_range_table = array;
      if (con == "C1_mod") space_->c1_tile_mod_table = array;
      if (con == "C0_mod") space_->c0_tile_mod_table = array;
      if (con == "gpu_thread_range") space_->gpu_thread_range_table = array;
      if (con == "gpu_block_range") space_->gpu_block_range_table = array;
      if (con == "gpu_thread_mod") space_->gpu_thread_mod_table = array;
      if (con == "gpu_block_mod") space_->gpu_block_mod_table = array;

      delete spaceDlPack;
    }
  }

 private:
  NodePtr<air::TileSpaceNode> space_;

  void CombineBand(size_t band, const std::vector<int> &idx, std::vector<int> &tile,
                   std::vector<std::vector<int>> &combined) {
    auto SharedPrune = [band, &idx, &tile, this](Result &res) -> bool {
      for (size_t i = 0; i < band; ++i) {
        for (size_t s = 0; s < this->is_shared_.size(); ++s) {
          if (tile[idx[i] + s] != res.tile[s]) return true;
        }
      }
      return false;
    };
    if (idx.empty()) return;
    int band_idx = idx[band];
    for (Result &res : result_[band]) {
      if (SharedPrune(res)) continue;
      for (size_t i = 0; i < res.tile.size(); ++i) tile[band_idx + i] = res.tile[i];
      if (band == result_.size() - 1)
        combined.emplace_back(tile);
      else
        CombineBand(band + 1, idx, tile, combined);
    }
  }

  bool ScanDown(size_t axis_idx, size_t band_idx) {
    if (axis_idx == tile_axes_.size()) {
      return AppendCand(band_idx);
    }
    TileAxis *axis = tile_axes_[axis_idx];
    TileAxis::Constraint &cons = axis->c1_constraints;
    const auto tile_min = cons.tile_min_.as<IntImm>();
    const auto tile_mod = cons.tile_mod_.as<IntImm>();
    const auto tile_extent = cons.tile_extent_.as<IntImm>();
    if (tile_min && tile_mod && tile_extent) {
      bool min_tile_ok = false;
      for (int64_t tile = tile_min->value; tile <= tile_extent->value; ++tile) {
        bool break_constraint =
          ((tile != tile_min->value) && (tile != tile_extent->value) && (tile % tile_mod->value != 0)) ||
          (axis->forbid_iso && tile_extent->value % tile != 0);
        if (analyzer_.scop_info_.user_config_.GetPruneTuningSpaceLevel() && break_constraint) {
          continue;
        }
        cand_.UpdateConstTile(axis, tile);
        if (analyzer_.scop_info_.user_config_.GetPruneTuningSpaceLevel() && !cand_.SpaceVerify(axis, CACHE1, band_idx)) {
          continue;
        }
        if (!ScanDown(axis_idx + 1, band_idx)) {
          return min_tile_ok;
        }
        if (!min_tile_ok) min_tile_ok = true;
      }
      return true;
    } else {
      LOG(INFO) << "Contain expr in axis, skip.";
      return false;
    }
  }

  bool AppendCand(size_t band_idx) {
    process_++;

    // check memory constraint
    int64_t mem_sz, align_sz;
    if (analyzer_.scop_info_.user_config_.GetTarget() == TARGET_CCE) {
      std::tie(mem_sz, align_sz) = cand_.MemInfer(MEM_SCOPE_BUFFER, band_idx);
      if (analyzer_.scop_info_.user_config_.GetPruneTuningSpaceLevel() == PRUNE_ALIGNED_MEM_EXCEED && align_sz > mem_limit_[MEM_SCOPE_BUFFER]) {
        return false;
      }
      if (analyzer_.scop_info_.user_config_.GetPruneTuningSpaceLevel() == PRUNE_MEM_EXCEED && mem_sz > mem_limit_[MEM_SCOPE_BUFFER]) {
        return false;
      }

    } else {
      int64_t shared_sz, local_sz;
      std::tie(shared_sz, std::ignore) = cand_.MemInfer(MEM_SCOPE_SHARED, band_idx);
      if (analyzer_.scop_info_.user_config_.GetPruneTuningSpaceLevel() && shared_sz > mem_limit_[MEM_SCOPE_SHARED]) {
        return false;
      }
      std::tie(local_sz, std::ignore) = cand_.MemInfer(MEM_SCOPE_LOCAL, band_idx);
      if (analyzer_.scop_info_.user_config_.GetPruneTuningSpaceLevel() && local_sz > mem_limit_[MEM_SCOPE_LOCAL]) {
        return false;
      }
      mem_sz = std::max(shared_sz, local_sz);
      align_sz = mem_sz;
    }

    // collect tile size
    std::vector<int> tile(tile_axes_.size());
    for (size_t i = 0; i < tile_axes_.size(); ++i) {
      auto tile_val = cand_.GetConstTileVal(tile_axes_[i]);
      tile[i] = tile_val.first;
    }

    auto LargerThan = [&tile](std::vector<int> &other) -> bool {
      for (size_t j = 0; j < tile.size(); ++j) {
        if (tile[j] < other[j]) return false;
      }
      return true;
    };
    auto DumpCand = [&tile, mem_sz, align_sz, this](const std::string &op) {
      if (this->process_ % DUMP_LINE_BREAK_NUM != 0) return;
      std::stringstream ss;
      ss << this->process_ << ": [";
      for (size_t i = 0; i < tile.size(); ++i) {
        ss << tile[i];
        if (i < tile.size() - 1) ss << ",";
      }
      ss << "], mem=(" << mem_sz << ", " << align_sz << "), " << op;
      LOG(INFO) << ss.str();
    };

    // pruning
    if (analyzer_.scop_info_.user_config_.GetTarget() == TARGET_CCE &&
        analyzer_.scop_info_.user_config_.GetPruneTuningSpaceLevel()) {
      for (auto &result : result_.back()) {
        // skip memory align tiling
        if ((mem_sz == result.mem_size) && (align_sz > result.align_size) && (LargerThan(result.tile))) {
          if (level_ >= DUMP_LEVEL_CANDIDATE) {
            DumpCand("skip");
          }
          return true;
        }
        // smaller memory, larger tile, then replace
        if ((mem_sz <= result.mem_size) && (align_sz <= result.align_size) && (LargerThan(result.tile))) {
          if (level_ >= DUMP_LEVEL_CANDIDATE) {
            DumpCand("replace");
          }
          result.tile = std::move(tile);
          result.mem_size = mem_sz;
          result.align_size = align_sz;
          return true;
        }
      }
    }

    // record result
    if (level_ >= DUMP_LEVEL_CANDIDATE) {
      DumpCand("new");
    }
    result_.back().emplace_back(Result{std::move(tile), mem_sz, align_sz});
    return true;
  }

  void CollectMemLimit() {
    if (analyzer_.scop_info_.user_config_.GetTarget() == TARGET_CCE) {
      NpuInfo &d_info = NpuInfo::GetInstance();
      for (auto i = 0; i < MEM_SCOPE_BULK; ++i) {
        this->mem_limit_[i] = d_info.GetMemoryLimitInScope(i);
      }
    } else {
      GpuInfo &gpu_info = GpuInfo::GetInstance();
      for (auto i = 0; i < MEM_SCOPE_BULK; ++i) {
        this->mem_limit_[i] = gpu_info.GetMemoryLimitInScope(i);
      }
    }
  }

  void CollectTileAxisTopDown(int b) {
    auto CollectTileAxis = [this, b](TileAxis *a) {
      if (a == analyzer_.RootAxis()) return;
      if (a->index == b) {
        this->cand_.InsertAxisBack(a);
        this->tile_axes_.emplace_back(a);
      }
    };
    tile_axes_.clear();
    cand_.ResetTileAxis();
    analyzer_.ForEachAxisTopDown(CollectTileAxis);
  }

  void CollectSharedAxis(int band_size) {
    std::vector<TileAxis *> base;
    std::vector<TileAxis *> current;
    int band = 0;
    auto CollectTileAxis = [band, &current, this](TileAxis *a) {
      if (a == analyzer_.RootAxis()) return;
      if (a->index == band) {
        current.emplace_back(a);
      }
    };
    analyzer_.ForEachAxisTopDown(CollectTileAxis);
    base = current;
    is_shared_ = std::vector<bool>(base.size(), false);
    size_t min_tile_size = base.size();
    for (band++; band < band_size; ++band) {
      analyzer_.ForEachAxisTopDown(CollectTileAxis);
      if (current.size() < min_tile_size) min_tile_size = current.size();
      size_t size = std::min(is_shared_.size(), current.size());
      for (size_t i = 0; i < size; ++i) {
        int64_t cur_const_extent = current[i]->GetConstExtent();
        int64_t base_const_extent = base[i]->GetConstExtent();
        if ((current[i]->range_min != base[i]->range_min) || (cur_const_extent != base_const_extent)) {
          size = i;
          break;
        }
      }
      if (size < is_shared_.size()) is_shared_.resize(size);
      if (size == 0) break;
    }
    size_t max_shared_size = min_tile_size / 2;
    if (is_shared_.size() > max_shared_size) is_shared_.resize(max_shared_size);
  }

  TilingAnalyzer &analyzer_;
  TileCandidate cand_;
  int level_{0};
  int64_t mem_limit_[MEM_SCOPE_BULK]{0};
  DLDataType type = {kDLInt, 32, 1};
  DLContext ctx = {kDLCPU, 0};
  std::vector<TileAxis *> tile_axes_;
  std::vector<bool> is_shared_;
  std::unordered_set<std::string> cared_info_;

  struct Result {
    std::vector<int> tile;
    int64_t mem_size;
    int64_t align_size;
  };
  std::vector<std::vector<Result>> result_;
  int process_{0};
};

NodeRef GenerateTilingSpace(const isl::schedule &sch, ScopInfo &scop_info, Stmt body, int dump_level) {
  CHECK(!scop_info.mmu_info_.HasCube()) << "cube op is not supported by auto tiling generator now!";
  TilingAnalyzer analyzer(sch, scop_info, body);
  bool need_tiling = analyzer.Prepare();
  std::stringstream ss;
  ss << body;
  analyzer.GetTileLogger().AppendLog(DO_TUNING, ss);
  if (!analyzer.GetTileLogger().DumpLogFile()) LOG(WARNING) << "Write tiling log fail.";
  TileSpaceCollector collector(analyzer, dump_level);
  if (need_tiling) collector.Collect();
  return collector.GetSpace();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
