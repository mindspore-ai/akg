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
#ifndef POLY_TILE_SPACE_H_
#define POLY_TILE_SPACE_H_

#include <tvm/base.h>
#include <tvm/expr.h>

namespace air {

const std::string kSpaceIndex = "index";
const std::string kSpaceC1Range = "C1_range";
const std::string kSpaceC0Range = "C0_range";
const std::string kSpaceC1Mod = "C1_mod";
const std::string kSpaceC0Mod = "C0_mod";
const std::string kSpaceGpuThreadRange = "gpu_thread_range";
const std::string kSpaceGpuBlockRange = "gpu_block_range";
const std::string kSpaceGpuThreadMod = "gpu_thread_mod";
const std::string kSpaceGpuBlockMod = "gpu_block_mod";

class TileSpaceNode : public Node {
 public:
  air::runtime::NDArray index_table;
  air::runtime::NDArray c1_tile_range_table;
  air::runtime::NDArray c0_tile_range_table;
  air::runtime::NDArray c1_tile_mod_table;
  air::runtime::NDArray c0_tile_mod_table;
  air::runtime::NDArray tiling_candidate;
  air::runtime::NDArray gpu_thread_range_table;
  air::runtime::NDArray gpu_block_range_table;
  air::runtime::NDArray gpu_thread_mod_table;
  air::runtime::NDArray gpu_block_mod_table;

  void VisitAttrs(AttrVisitor *v) {
    v->Visit("index_table", &index_table);
    v->Visit("c1_tile_range_table", &c1_tile_range_table);
    v->Visit("c0_tile_range_table", &c0_tile_range_table);
    v->Visit("c1_tile_mod_table", &c1_tile_mod_table);
    v->Visit("c0_tile_mod_table", &c0_tile_mod_table);
    v->Visit("tiling_candidate", &tiling_candidate);
    v->Visit("gpu_thread_range_table", &gpu_thread_range_table);
    v->Visit("gpu_block_range_table", &gpu_block_range_table);
    v->Visit("gpu_thread_mod_table", &gpu_thread_mod_table);
    v->Visit("gpu_block_mod_table", &gpu_block_mod_table);
  }
  static constexpr const char *_type_key = "TileSpace";
  TVM_DECLARE_NODE_TYPE_INFO(TileSpaceNode, Node);
};

class TileSpace : public NodeRef {
 public:
  TileSpace() {}
  // This constructor involves implicit type conversion so explict key word is not used.
  TileSpace(const ObjectPtr<Object> &n) : NodeRef(n) {}
  ~TileSpace() {}

  inline TileSpaceNode *operator->() const { return static_cast<TileSpaceNode *>(data_.get()); }
};

TVM_REGISTER_NODE_TYPE(TileSpaceNode);

}  // namespace air

#endif  // POLY_TILE_SPACE_H_
