/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef POLY_CUSTOM_TILING_H_
#define POLY_CUSTOM_TILING_H_
#include <tvm/base.h>
#include <tvm/expr.h>
#include <string>

namespace air {
/*!
 * \brief Custom tiling constraints for user specified axis
 *  Users can apply custom tiling by setting dim and related constraint
 *  Or specify the special tile format of tensor to apply default tiling strategy
 *  Use CustomTiling as its container type
 */
class CustomTilingNode : public Node {
 public:
  /*! \brief tile location, chosen from "L1" and "L0"*/
  Expr tile_level;

  /*! \brief custom mode, chosen from "AXIS" and "TENSOR"*/
  Expr tile_mode;

  /*! \brief The specified tensor name for custom tiling */
  Expr tensor_name;

  /*! \brief The dim of tensor for custom tiling*/
  int tile_pos;

  /*! \brief The band for custom tiling*/
  int tile_band;

  /*! \brief The axis for custom tiling*/
  int tile_axis;

  /*! \brief minimal tile factor, greater than 0*/
  Expr tile_min;

  /*! \brief maximal tile factor*/
  Expr tile_max;

  /*! \brief constraint tile factor % tile_mod == 0*/
  Expr tile_mod;

  /*! \brief directly set tile factor to a certain value*/
  Expr tile_factor;

  /*! \brief add tile factor candidate*/
  Expr tile_candidate;

  /*! \brief whether forbid axis to be tiled to isolate block*/
  int forbid_isolate;

  /*! \brief axis info used in dim structure, default is axis's sequence,
   * "H" and "W" may be used in cube op*/
  Expr axis_info;

  /*! \brief the priority of certain axis for tiling:
   * memory will serve axis with smaller priority first*/
  int priority;

  /*! \brief the memory expansion of certain buffer due to alignment
   * or other possible reasons*/
  int expansion;

  /*! \brief the ratio of memory allocation used in auto tiling,
   * default is 0.5 which is reserved for double buffer*/
  double mem_ratio;

  void VisitAttrs(AttrVisitor *v) {
    v->Visit("tile_level", &tile_level);
    v->Visit("tile_mode", &tile_mode);
    v->Visit("tensor_name", &tensor_name);
    v->Visit("tile_pos", &tile_pos);
    v->Visit("tile_band", &tile_band);
    v->Visit("tile_axis", &tile_axis);
    v->Visit("tile_min", &tile_min);
    v->Visit("tile_max", &tile_max);
    v->Visit("tile_mod", &tile_mod);
    v->Visit("tile_factor", &tile_factor);
    v->Visit("tile_candidate", &tile_candidate);
    v->Visit("forbid_isolate", &forbid_isolate);
    v->Visit("axis_info", &axis_info);
    v->Visit("priority", &priority);
    v->Visit("expansion", &expansion);
    v->Visit("mem_ratio", &mem_ratio);
  }

  static constexpr const char *_type_key = "CustomTilingNode";
  TVM_DECLARE_NODE_TYPE_INFO(CustomTilingNode, Node);
};

class CustomTiling : public NodeRef {
 public:
  CustomTiling() {}
  explicit CustomTiling(const ObjectPtr<Object> &n) : NodeRef(n) {}
  ~CustomTiling() {}

  inline CustomTilingNode *operator->() const { return static_cast<CustomTilingNode *>(data_.get()); }
};

TVM_REGISTER_NODE_TYPE(CustomTilingNode);
}  // namespace air

#endif  // POLY_CUSTOM_TILING_H_
