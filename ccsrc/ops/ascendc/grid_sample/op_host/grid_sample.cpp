/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "grid_sample_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus GridSampleTiling(gert::TilingContext *context) {
  GridSampleTilingData tiling;
  uint32_t tiling_key{0};
  uint64_t ubSize;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  auto coreNum = ascendcPlatform.GetCoreNum();

  auto input_shape = context->GetInputShape(0)->GetOriginShape();
  auto grid_shape = context->GetInputShape(1)->GetOriginShape();

  int32_t n_in = input_shape.GetDim(0);
  float h_in = static_cast<float>(input_shape.GetDim(1));
  float w_in = static_cast<float>(input_shape.GetDim(2));
  int32_t c_in = input_shape.GetDim(3);

  int32_t h_out = grid_shape.GetDim(1);
  int32_t w_out = grid_shape.GetDim(2);

  tiling.set_h_in(h_in);
  tiling.set_w_in(w_in);
  tiling.set_h_out(h_out);
  tiling.set_w_out(w_out);
  tiling.set_n_in(n_in);
  tiling.set_c_in(c_in);

  context->SetBlockDim(coreNum);
  context->SetTilingKey(tiling_key);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus GridSampleInferShape(gert::InferShapeContext *context) {
  const gert::Shape *input_shape = context->GetInputShape(0);
  const gert::Shape *grid_shape = context->GetInputShape(1);
  gert::Shape *out_shape = context->GetOutputShape(0);
  *out_shape = *grid_shape;
  (*out_shape)[3] = (*input_shape)[3];
  return GRAPH_SUCCESS;
}
static graphStatus GridSampleInferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class GridSample : public OpDef {
 public:
  explicit GridSample(const char *name) : OpDef(name) {
    this->Input("input")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("grid")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .AutoContiguous();
    this->Output("output")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});

    this->SetInferShape(ge::GridSampleInferShape).SetInferDataType(ge::GridSampleInferDataType);
    this->AICore().SetTiling(optiling::GridSampleTiling).AddConfig("ascend310p");
  }
};
OP_ADD(GridSample);
}  // namespace ops
