/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "add_rms_norm_custom_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
constexpr uint32_t kDtypeKeyFp16 = 1;
constexpr uint32_t kDtypeKeyFp32 = 2;
constexpr uint32_t kDtypeKeyBf16 = 3;
constexpr uint32_t kUbFactorB16 = 12288;
constexpr uint32_t kUbFactorB32 = 10240;
constexpr uint32_t kUbFactorB16Cutd = 12096;
constexpr uint32_t kUbFactorB32Cutd = 9696;
constexpr uint32_t kBlockAlignNum = 16;
constexpr size_t kWorkspaceSize = 16 * 1024 * 1024 + 256;

inline int64_t CeilDiv(const int64_t dividend, const int64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  return (dividend + divisor - 1) / divisor;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  AddRmsNormTilingData tiling;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto block_dims = ascendcPlatform.GetCoreNumAiv();
  const float *eps = context->GetAttrs()->GetAttrPointer<float>(0);

  uint32_t row_factor = 64;
  int64_t num_col = 1;
  int64_t num_row = 1;

  auto gamma_shape = context->GetInputShape(2)->GetOriginShape();
  auto gamma_dim = gamma_shape.GetDimNum();
  for (size_t idx = 0; idx < gamma_dim; idx++) {
    num_col = num_col * gamma_shape.GetDim(idx);
  }
  float avg_factor = (num_col == 0) ? 0 : 1.0 / num_col;

  auto x1_shape = context->GetInputShape(0)->GetOriginShape();
  auto x_dim = x1_shape.GetDimNum();
  for (size_t idx = 0; idx < x_dim - gamma_dim; idx++) {
    num_row = num_row * x1_shape.GetDim(idx);
  }

  uint32_t block_factor = 1;
  uint32_t tile_num = CeilDiv(num_row, block_dims * block_factor);
  block_factor *= tile_num;
  uint32_t use_core_num = CeilDiv(num_row, block_factor);

  uint32_t dtype_key;
  uint32_t ub_factor = kUbFactorB16;
  bool is_cast_gamma = false;
  ge::DataType x1_dtype = context->GetInputDesc(0)->GetDataType();
  ge::DataType gamma_dtype = context->GetInputDesc(2)->GetDataType();
  if (x1_dtype == ge::DataType::DT_FLOAT16) {
    dtype_key = kDtypeKeyFp16;
    if (gamma_dtype == ge::DataType::DT_FLOAT) {
      is_cast_gamma = true;
      ub_factor = kUbFactorB32;
    }
  } else if (x1_dtype == ge::DataType::DT_FLOAT) {
    dtype_key = kDtypeKeyFp32;
    ub_factor = kUbFactorB32;
  } else if (x1_dtype == ge::DataType::DT_BF16) {
    dtype_key = kDtypeKeyBf16;
    if (gamma_dtype == ge::DataType::DT_FLOAT) {
      is_cast_gamma = true;
      ub_factor = kUbFactorB32;
    }
  }

  uint32_t split_d = num_col > ub_factor ? 1 : 0;
  if (split_d == 1) {
    ub_factor = ((x1_dtype == ge::DataType::DT_FLOAT) || is_cast_gamma) ? kUbFactorB32Cutd
                                                                        : kUbFactorB16Cutd;
    uint32_t col_tile_num = CeilDiv(num_col, ub_factor);
    ub_factor = CeilDiv(num_col, col_tile_num * kBlockAlignNum) * kBlockAlignNum;
  }

  uint32_t tiling_key = dtype_key * 10 + split_d;
  if (is_cast_gamma) {
    tiling_key = tiling_key + 100;
  }

  tiling.set_num_col(num_col);
  tiling.set_num_row(num_row);
  tiling.set_epsilon(*eps);
  tiling.set_block_factor(block_factor);
  tiling.set_row_factor(row_factor);
  tiling.set_ub_factor(ub_factor);
  tiling.set_avg_factor(avg_factor);

  context->SetBlockDim(use_core_num);
  context->SetTilingKey(tiling_key);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = kWorkspaceSize;
  return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *x1_shape = context->GetInputShape(0);
  gert::Shape *y_shape = context->GetOutputShape(0);
  gert::Shape *x_shape = context->GetOutputShape(2);
  *y_shape = *x1_shape;
  *x_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  context->SetOutputDataType(1, ge::DT_FLOAT);
  context->SetOutputDataType(2, inputDataType);
  return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class AddRmsNormCustom : public OpDef {
public:
  explicit AddRmsNormCustom(const char *name) : OpDef(name) {
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("gamma")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("rstd")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("epsilon").Float();

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
  }
};
OP_ADD(AddRmsNormCustom);
} // namespace ops
