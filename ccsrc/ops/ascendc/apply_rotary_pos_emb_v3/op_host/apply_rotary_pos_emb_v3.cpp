/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "apply_rotary_pos_emb_v3_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;
static ge::graphStatus ApplyRotaryPosEmbV3Tiling(gert::TilingContext *context) {
  ApplyRotaryPosEmbV3TilingData tiling;
  uint32_t tiling_key{0};
  uint64_t ubSize;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  auto coreNum = ascendcPlatform.GetCoreNum();

  auto query_shape = context->GetInputShape(0)->GetOriginShape();
  auto key_shape = context->GetInputShape(1)->GetOriginShape();
  auto cos_shape = context->GetInputShape(2)->GetOriginShape();

  uint32_t tokens = query_shape.GetDim(0);
  uint32_t query_head_num = query_shape.GetDim(1);
  uint32_t key_head_num = key_shape.GetDim(1);

  uint32_t query_head_dim = query_shape.GetDim(2);
  uint32_t cos_head_dim = cos_shape.GetDim(1);
  uint32_t rotary_dim = cos_head_dim *2;

  uint32_t is_split = (rotary_dim == query_head_dim ? 0: 1);
  tiling.set_queryHeadDim(query_head_dim);
  tiling.set_qHeadNum(query_head_num);
  tiling.set_kHeadNum(key_head_num);
  tiling.set_rotaryDim(rotary_dim);
  tiling.set_qHiddenSize(query_head_num * rotary_dim);
  tiling.set_kHiddenSize(key_head_num * rotary_dim);
  tiling.set_cosHeadDim(cos_head_dim);

  if (tokens < coreNum) {
    coreNum = tokens;
  }
  tiling.set_tokensPerCore(static_cast<uint32_t>(tokens / coreNum));
  tiling.set_tokensTail(tokens % coreNum);
  const uint32_t *layout = context->GetAttrs()->GetAttrPointer<uint32_t>(0);
  const uint32_t *rotaryMode = context->GetAttrs()->GetAttrPointer<uint32_t>(1);
  tiling.set_layout(*layout);
  tiling.set_rotaryMode(*rotaryMode);

  ge::DataType query_type = context->GetInputDesc(0)->GetDataType();
  if (query_type == ge::DataType::DT_FLOAT16) {
    tiling_key = 1;
  }else if(query_type == ge::DataType::DT_FLOAT) {
    tiling_key = 2;
  }
  tiling_key = tiling_key * 10 + is_split;

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
static ge::graphStatus ApplyRotaryPosEmbV3InferShape(gert::InferShapeContext *context) {
  const gert::Shape *query_shape = context->GetInputShape(0);
  const gert::Shape *key_shape = context->GetInputShape(1);
  gert::Shape *out_query_shape = context->GetOutputShape(0);
  gert::Shape *out_key_shape = context->GetOutputShape(1);
  *out_query_shape = *query_shape;
  *out_key_shape = *key_shape;
  return GRAPH_SUCCESS;
}
static graphStatus ApplyRotaryPosEmbV3InferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  context->SetOutputDataType(1, inputDataType);
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ApplyRotaryPosEmbV3 : public OpDef {
 public:
  explicit ApplyRotaryPosEmbV3(const char *name) : OpDef(name) {
    this->Input("query")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("key")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("cos")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("sin")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Output("query")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("key")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("layout").AttrType(OPTIONAL).Int(1);
    this->Attr("rotary_mode").AttrType(OPTIONAL).String("interleave");

    this->SetInferShape(ge::ApplyRotaryPosEmbV3InferShape).SetInferDataType(ge::ApplyRotaryPosEmbV3InferDataType);
    this->AICore().SetTiling(optiling::ApplyRotaryPosEmbV3Tiling).AddConfig("ascend310p");
  }
};
OP_ADD(ApplyRotaryPosEmbV3);
}  // namespace ops
