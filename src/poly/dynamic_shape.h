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
#ifndef POLY_DYNAMIC_SHAPE_H
#define POLY_DYNAMIC_SHAPE_H
namespace air {
/*!
 * \brief Dynamic shape node
 *  Users can set attributes for dynamic shape
 */
class DynamicShapeNode : public Node {
 public:
  /*! \brief The specified tensor name with dynamic shape"*/
  std::string tensor_name;
  /*! \brief The dim of tensor with dynamic shape*/
  int pos;
  /*! \brief integer limit of dynamic shape"*/
  int dyn_shape_limit;
  /*! \brief integer poly upper bound of dynamic shape"*/
  int poly_upper_bound;

  void VisitAttrs(AttrVisitor *v) {
    v->Visit("tensor_name", &tensor_name);
    v->Visit("pos", &pos);
    v->Visit("dyn_shape_limit", &dyn_shape_limit);
    v->Visit("poly_upper_bound", &poly_upper_bound);
  }

  static constexpr const char *_type_key = "DynamicShapeNode";
  TVM_DECLARE_NODE_TYPE_INFO(DynamicShapeNode, Node);
};

class DynamicShape : public NodeRef {
 public:
  DynamicShape() {}
  explicit DynamicShape(const ObjectPtr<Object> &n) : NodeRef(n) {}
  ~DynamicShape() {}

  inline DynamicShapeNode *operator->() const { return static_cast<DynamicShapeNode *>(data_.get()); }
};

TVM_REGISTER_NODE_TYPE(DynamicShapeNode);
}  // namespace air

const int default_kernel_h = 3;
const int default_kernel_w = 3;

#endif  // POLY_DYNAMIC_SHAPE_H
