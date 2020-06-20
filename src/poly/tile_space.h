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

namespace ktvm {
class TileSpaceNode : public Node {
 public:
  ktvm::runtime::NDArray index_table;
  ktvm::runtime::NDArray l1_tile_range_table;
  ktvm::runtime::NDArray l0_tile_range_table;
  ktvm::runtime::NDArray l1_tile_mod_table;
  ktvm::runtime::NDArray l0_tile_mod_table;
  ktvm::runtime::NDArray tiling_candidate;

  void VisitAttrs(AttrVisitor *v) {
    v->Visit("index_table", &index_table);
    v->Visit("l1_tile_range_table", &l1_tile_range_table);
    v->Visit("l0_tile_range_table", &l0_tile_range_table);
    v->Visit("l1_tile_mod_table", &l1_tile_mod_table);
    v->Visit("l0_tile_mod_table", &l0_tile_mod_table);
    v->Visit("tiling_candidate", &tiling_candidate);
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

}  // namespace ktvm

#endif  // POLY_TILE_SPACE_H_
