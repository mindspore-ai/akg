/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef AKG_TARGET_INFO_H_
#define AKG_TARGET_INFO_H_

#include <string>

#include <tvm/base.h>
#include <tvm/expr.h>

namespace air {

/*!
 * \brief Memory information of gpu memory region.
 *  Use GpuMemoryInfoNode as its container type
 */
struct GpuMemoryInfoNode : public Node {
  /*! \brief Maximum number of bytes supported in the memory per block */
  int max_bytes_per_block;

  void VisitAttrs(AttrVisitor *v) { v->Visit("max_bytes_per_block", &max_bytes_per_block); }

  static constexpr const char *_type_key = "GpuMemoryInfo";
  TVM_DECLARE_NODE_TYPE_INFO(GpuMemoryInfoNode, Node);
};

/*! \brief Defines memory info */
TVM_DEFINE_NODE_REF(GpuMemoryInfo, GpuMemoryInfoNode);

/*!
 * \brief get memory info given scope
 * \param scope The scope name.
 * \return info The memory info.
 */
TVM_DLL GpuMemoryInfo GetGpuMemoryInfo(const std::string &scope, const std::string &device_type = "");

/*!
 * \brief Compute ability information of gpu.
 *  Use GpuComputeInfoNode as its container type
 */
struct GpuComputeInfoNode : public Node {
  /*! \brief The number of sm per gpu instance */
  int num_sm;

  /*! \brief The proposed number of active blocks per sm */
  int active_blocks_per_sm;

  /*! \brief The minimal number of for-loop size for io-bounded ops */
  int min_elem_for_io_bound;

  void VisitAttrs(AttrVisitor *v) {
    v->Visit("num_sm", &num_sm);
    v->Visit("active_blocks_per_sm", &active_blocks_per_sm);
    v->Visit("min_elem_for_io_bound", &min_elem_for_io_bound);
  }

  static constexpr const char *_type_key = "GpuComputeInfo";
  TVM_DECLARE_NODE_TYPE_INFO(GpuComputeInfoNode, Node);
};

/*! \brief Defines memory info */
TVM_DEFINE_NODE_REF(GpuComputeInfo, GpuComputeInfoNode);

/*!
 * \brief get compute capability info given scope
 * \param scope The scope name.
 * \return info The compute capability info.
 */
TVM_DLL GpuComputeInfo GetGpuComputeInfo(const std::string &scope, const std::string &device_type = "");

}  // namespace air
#endif  // AKG_TARGET_INFO_H_
